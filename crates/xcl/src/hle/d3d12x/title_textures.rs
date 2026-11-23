//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The first upload is not done when the texture is made (a title makes thousands while it
//! streams – Fortnite's render thread then spent its time in CreatePlacedResourceX and its
//! render command fences ran late): it is pending until the title's next ExecuteCommandLists,
//! where all pending and rewritten textures go in batches on one list each.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch with atomic reset of the reported writes.
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// (device, address, layout, state) of an upload
type Upload = (usize, usize, Arc<xg::Texture>, D3D12_RESOURCE_STATES);

struct Entry {
    device: usize,
    va: usize,
    tex: Arc<xg::Texture>,
    /// the state the texture was created in (uploads leave it there)
    state: D3D12_RESOURCE_STATES,
    /// not uploaded yet (made since the last submit)
    pending: bool,
    /// the title's GPU work wrote it (CopyTextureRegion into it): its CPU view is stale from
    /// then on and no longer uploaded
    gpu_written: bool,
    /// the title's CPU wrote its memory and XCL was told (`note_cpu_write`): uploaded again
    /// before the title's next submit, watched memory or not
    cpu_written: bool,
}

/// textures with `cpu_written` set
static CPU_WRITTEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CPU_WRITE_UPLOADS: AtomicU64 = AtomicU64::new(0);
/// microseconds and textures of those uploads: [as every upload, by rows]
static CPU_WRITE_UPLOAD_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static CPU_WRITE_UPLOADS_BY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// of the uploads by rows: microseconds copying rows, and from the list's close to the GPU's end
static ROWS_COPY_US: AtomicU64 = AtomicU64::new(0);
static ROWS_GPU_US: AtomicU64 = AtomicU64::new(0);

/// textures waiting for their first upload
static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<BTreeMap<usize, Entry>> = Mutex::new(BTreeMap::new());
// The dirty-page scan consumes write-watch state for every submission thread.
// Keep the scan, pending-list update and GPU uploads together.
static REFRESH: Mutex<()> = Mutex::new(());
static TEXTURES_GEN: AtomicU64 = AtomicU64::new(0);
static TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// cached `watch_ranges` keyed by `(sparse::mapped_gen, TEXTURES_GEN, graphics_reservations_gen)`
static WATCH_CACHE: Mutex<Option<((u64, u64, u64), Arc<Vec<(usize, usize)>>)>> = Mutex::new(None);
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES.lock().unwrap().len(), UPLOADS.load(Ordering::Relaxed))
}

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // render targets and depth buffers are the GPU's (their metadata planes, empty planes)
    if tex.planes.len() != 1 || tex.texels().is_none() {
        return Err("a render target / depth buffer".into());
    }
    if ID3D12Resource::from_raw_borrowed(&res).is_none() {
        return Err("no resource".into());
    }
    let e = Entry { device: device as usize, va, tex: Arc::new(tex), state, pending: true, gpu_written: false, cpu_written: false };
    {
        let mut t = TEXTURES.lock().unwrap();
        if let Some(old) = t.insert(res as usize, e) {
            if old.pending {
                PENDING.fetch_sub(1, Ordering::Relaxed);
            }
            if old.cpu_written {
                CPU_WRITTEN.fetch_sub(1, Ordering::Relaxed);
            }
        }
        TEXTURES_LEN.store(t.len(), Ordering::Relaxed);
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
    Ok(())
}

/// The title's GPU work writes texture `res` (a copy into it): on the console the memory then
/// holds what the GPU wrote; here the CPU's view of it is stale, so it is not uploaded again
/// (a CPU write elsewhere on its pages – a neighbour, an allocator's header – would upload the
/// stale view over the copy: Fortnite's font atlas is zeros in memory, its glyphs are copied in
/// by the GPU). The first upload, if still pending, stays: it runs before the title's submit
/// that copies.
pub fn note_gpu_write(res: usize) {
    if let Some(e) = TEXTURES.lock().unwrap().get_mut(&res) {
        if !e.gpu_written {
            e.gpu_written = true;
            GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static GPU_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().iter().filter(|(_, e)| e.va < va + len && va < e.va + e.tex.size() as usize).map(|(&r, e)| (r, e.va)).take(4).collect()
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    // (its staging buffer is let go outside the lock: that release comes back here)
    let stage = STAGES.lock().unwrap().remove(&res);
    drop(stage);
    let removed = {
        let mut t = TEXTURES.lock().unwrap();
        let r = t.remove(&res);
        TEXTURES_LEN.store(t.len(), Ordering::Relaxed);
        r
    };
    if let Some(e) = removed {
        if e.pending {
            PENDING.fetch_sub(1, Ordering::Relaxed);
        }
        if e.cpu_written {
            CPU_WRITTEN.fetch_sub(1, Ordering::Relaxed);
        }
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The title's CPU wrote `[va, va + len)` of its memory in a way XCL knows of (a video decoder
/// of its own filled the frame buffers it was given, `hle::bink`): the textures kept in step
/// that lie over it are uploaded again before the title's next submit. Memory that is not
/// write-watched (a native host's graphics memory, `kernelx::graphics_protect`) has no other
/// way to say so; for watched memory this is the same upload the watch would start. Returns
/// how many textures lie there.
pub fn note_cpu_write(va: usize, len: usize) -> usize {
    // the memory itself, and where a page pool's GPU view shows it: a title with such a pool
    // places its textures at the GPU view's addresses and writes them through the CPU's
    let mut ranges = vec![(va, len)];
    ranges.extend(super::gpu_views::views_of(va, len));
    let mut t = TEXTURES.lock().unwrap();
    let mut n = 0;
    for e in t.values_mut().filter(|e| ranges.iter().any(|&(va, len)| e.va < va + len && va < e.va + e.tex.size() as usize)) {
        n += 1;
        if !e.cpu_written {
            e.cpu_written = true;
            CPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
    n
}

/// Uploads done because the title's CPU writes were noted (`note_cpu_write`).
pub fn cpu_write_uploads() -> u64 {
    CPU_WRITE_UPLOADS.load(Ordering::Relaxed)
}

/// What those uploads took, as text: textures and milliseconds for the ones done as every
/// upload and for the ones done by rows (of which copying rows / the GPU's part).
pub fn cpu_write_upload_times() -> String {
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
    format!(
        "{} as every upload in {} ms, {} by rows in {} ms (rows copied in {} ms, list run and waited for in {} ms)",
        l(&CPU_WRITE_UPLOADS_BY[0]),
        l(&CPU_WRITE_UPLOAD_US[0]) / 1000,
        l(&CPU_WRITE_UPLOADS_BY[1]),
        l(&CPU_WRITE_UPLOAD_US[1]) / 1000,
        l(&ROWS_COPY_US) / 1000,
        l(&ROWS_GPU_US) / 1000
    )
}

/// `XCL_TRACE_VIDEO`: what `[va, va + len)` of the title's memory is here – its unified-memory
/// window, what Windows says of its first page, whether writes to it are watched, and the
/// resources the title placed there.
pub unsafe fn describe_memory(va: usize, len: usize) -> String {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION};
    let mut out: Vec<String> = Vec::new();
    match crate::gpu::uma::get() {
        Some(u) => match u.window_of(va) {
            Some(k) => out.push(format!("unified memory window {k} ({}{})", if u.is_sparse(va) { "sparse" } else { "committed" }, if u.is_title_address(va) { ", the title's" } else { ", not handed out" })),
            None => out.push("outside unified memory".into()),
        },
        None => out.push("no unified memory".into()),
    }
    if crate::hle::kernelx::in_graphics_reservation(va) {
        out.push("in a graphics reservation".into());
    }
    match (super::gpu_views::memory_at(va), super::gpu_views::views_of(va, len)) {
        (Some(at), _) => out.push(format!("a page pool's GPU view of the memory at {at:#x}")),
        (None, views) if !views.is_empty() => out.push(format!("shown in a page pool's GPU view at {:x?}{}", &views[..views.len().min(3)], if views.len() > 3 { format!(" … ({} ranges)", views.len()) } else { String::new() })),
        _ => out.push(format!("in no page pool's GPU view ({} pages mapped in GPU views, physical page here: {:x?})", super::gpu_views::counts().0, crate::hle::kernelx::page_at(va))),
    }
    let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0 {
        out.push(format!("allocation {:#x}, region {:#x}+{:#x}, state {:#x}, protect {:#x}, type {:#x}", mbi.AllocationBase as usize, mbi.BaseAddress as usize, mbi.RegionSize, mbi.State, mbi.Protect, mbi.Type));
    }
    // (no reset: the look before the next submit still sees these writes)
    let mut buf = [std::ptr::null_mut::<c_void>(); 64];
    let (mut n, mut grain) = (buf.len(), 0u32);
    let r = GetWriteWatch(0, (va & !0xFFF) as *const c_void, (len + 0xFFF) & !0xFFF, buf.as_mut_ptr(), &mut n, &mut grain);
    out.push(if r == 0 { format!("write-watched ({n}{} pages written)", if n == buf.len() { "+" } else { "" }) } else { "not write-watched".into() });
    let desc = |r: usize| {
        let p = r as *mut c_void;
        ID3D12Resource::from_raw_borrowed(&p).map(|x| x.GetDesc()).map(|d| format!("{}x{} format {} mips {} flags {:#x}", d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0)).unwrap_or_default()
    };
    {
        let t = TEXTURES.lock().unwrap();
        let over: Vec<String> = t
            .iter()
            .filter(|(_, e)| e.va < va + len && va < e.va + e.tex.size() as usize)
            .take(4)
            .map(|(&r, e)| format!("{r:#x} at {:#x}+{:#x} ({}{}{})", e.va, e.tex.size(), desc(r), if e.pending { ", upload pending" } else { "" }, if e.gpu_written { ", GPU-written" } else { "" }))
            .collect();
        out.push(if over.is_empty() { "no title texture kept in step".into() } else { format!("title texture(s) kept in step: {}", over.join(", ")) });
    }
    let near = |list: Vec<(usize, usize)>| list.into_iter().take(4).map(|(r, a)| format!("{r:#x} at {a:#x} ({})", desc(r))).collect::<Vec<_>>().join(", ");
    let (lo, hi) = (va.saturating_sub(64 << 20), va + len);
    let placed = super::vram::placed_in(lo, hi);
    if !placed.is_empty() {
        out.push(format!("video-memory texture(s) placed from 64 MiB below: {}", near(placed)));
    }
    let own = super::real::title_addresses_in(va.saturating_sub(0x1_0000), hi);
    if !own.is_empty() {
        out.push(format!("resource(s) of their own at title addresses from 64 KiB below: {}", near(own)));
    }
    out.join("; ")
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    TEXTURES.lock().unwrap().get(&res).map(|e| e.va)
}

thread_local! {
    /// XCL's own submits on this thread (texture uploads): they read no title memory on the GPU,
    /// so they do not look for CPU writes first – that look is the title's submits' job
    static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Reusable `GetWriteWatch` output buffer (avoids 128 KiB alloc+zero+free per submit).
    static WATCH_BUF: std::cell::RefCell<Vec<*mut c_void>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable dirty page list for `refresh_once`.
    static DIRTY_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn cached_watch_ranges(u: &crate::gpu::uma::Uma) -> Arc<Vec<(usize, usize)>> {
    let key = (
        super::sparse::mapped_gen(),
        TEXTURES_GEN.load(Ordering::Acquire),
        crate::hle::kernelx::graphics_reservations_gen(),
    );
    {
        let g = WATCH_CACHE.lock().unwrap();
        if let Some((k, ref ranges)) = *g {
            if k == key {
                return Arc::clone(ranges);
            }
        }
    }
    // Recompute the watched ranges (textures kept in step + mapped sparse tiles clipped to
    // write-watched regions) only when the set of textures, tiles, or reservations changed.
    let mut regions = u.watch_regions();
    regions.extend(crate::hle::kernelx::graphics_reservations());
    let mut wanted: Vec<(usize, usize)> = TEXTURES.lock().unwrap().values().map(|e| (e.va & !0xFFF, (e.va + e.tex.size() as usize + 0xFFF) & !0xFFF)).collect();
    wanted.extend(super::sparse::mapped_ranges());
    let computed = Arc::new(watch_ranges(&regions, wanted));
    *WATCH_CACHE.lock().unwrap() = Some((key, Arc::clone(&computed)));
    computed
}

/// Run `f` as XCL's own GPU work: ExecuteCommandLists inside it does not refresh.
pub(super) fn own_work<R>(f: impl FnOnce() -> R) -> R {
    let was = BUSY.with(|b| b.replace(true));
    let r = f();
    BUSY.with(|b| b.set(was));
    r
}

/// Whether the current thread is inside XCL's own internal GPU setup/upload/readback work.
pub(super) fn is_own_work() -> bool {
    BUSY.with(|b| b.get())
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) {
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.get()) {
        return;
    }
    let _refresh = REFRESH.lock().unwrap();
    own_work(|| refresh_once(queue));
}

unsafe fn refresh_once(queue: *mut c_void) {
    let Some(u) = crate::gpu::uma::get() else { return };
    let sparse = u.has_sparse();
    if !sparse && TEXTURES_LEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    let watch_t0 = std::time::Instant::now();
    let mut page = 0x1000u32;
    let ranges = cached_watch_ranges(u);
    let dirty_len = DIRTY_BUF.with(|dcell| {
        let mut dirty = dcell.borrow_mut();
        dirty.clear();
        WATCH_BUF.with(|wcell| {
            let mut buf = wcell.borrow_mut();
            if buf.len() < 16384 {
                buf.resize(16384, std::ptr::null_mut());
            }
            for &(base, size) in ranges.iter() {
                loop {
                    let mut n = buf.len();
                    if GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, size, buf.as_mut_ptr(), &mut n, &mut page) != 0 {
                        break;
                    }
                    dirty.extend(buf[..n].iter().map(|&p| p as usize));
                    if n < buf.len() {
                        break;
                    }
                }
            }
        });
        dirty.sort_unstable();
        dirty.len()
    });
    super::sparse::note_watch(watch_t0.elapsed(), dirty_len);
    if super::present::trace_frame() {
        // every 600th look: what the CPU's view of a few textures holds now (does the title's
        // data reach the address at all, and is the texture still kept in step?)
        static LOOKS: AtomicU64 = AtomicU64::new(0);
        if LOOKS.fetch_add(1, Ordering::Relaxed) % 600 == 599 {
            let t = TEXTURES.lock().unwrap();
            // 16 samples of 4 KiB over the whole texture (GFX10 keeps the small levels first,
            // level 0 last)
            let content = |e: &Entry| -> usize {
                let size = e.tex.size() as usize;
                (0..16usize)
                    .map(|k| e.va + (size.saturating_sub(4096)) * k / 15)
                    .filter(|&a| crate::gpu::shader::readable(a, 4096))
                    .map(|a| std::slice::from_raw_parts(a as *const u8, 4096).iter().filter(|&&b| b != 0).count())
                    .sum()
            };
            let (mut with, mut without) = (0, 0);
            let mut shown = 0;
            for e in t.values() {
                let c = content(e);
                if c > 0 { with += 1 } else { without += 1 }
                if c > 0 && shown < 4 {
                    shown += 1;
                    crate::xlog_info!("frame trace: title texture at {:#x} ({} bytes) has data in memory ({c} of 65536 sampled bytes); GPU-written {}", e.va, e.tex.size(), e.gpu_written);
                }
            }
            crate::xlog_info!("frame trace: title textures: {with} with data in memory, {without} all zero; {dirty_len} dirty pages this look");
        }
    }
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        DIRTY_BUF.with(|dcell| {
            let dirty = dcell.borrow();
            super::sparse::before_execute(q, &dirty, page as usize);
        });
    }
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 && CPU_WRITTEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let mut written: Vec<(ID3D12Resource, Upload)> = Vec::new();
    let todo: Vec<(ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.iter_mut()
            .filter(|(_, e)| {
                if e.cpu_written || e.pending {
                    return true;
                }
                if e.gpu_written {
                    return false;
                }
                let i = dirty.partition_point(|&p| p + page as usize <= e.va);
                dirty.get(i).is_some_and(|&p| p < e.va + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                if e.pending {
                    e.pending = false;
                    PENDING.fetch_sub(1, Ordering::Relaxed);
                }
                let noted = e.cpu_written;
                if noted {
                    e.cpu_written = false;
                    CPU_WRITTEN.fetch_sub(1, Ordering::Relaxed);
                    CPU_WRITE_UPLOADS.fetch_add(1, Ordering::Relaxed);
                }
                let p = res as *mut c_void;
                let item = ID3D12Resource::from_raw_borrowed(&p).map(|r| (r.clone(), (e.device, e.va, Arc::clone(&e.tex), e.state)))?;
                if noted {
                    written.push(item);
                    return None;
                }
                Some(item)
            })
            .collect()
    });
    if !written.is_empty() {
        // textures the title rewrites frame after frame: row by row from its memory where they
        // lie that way, the rest as every other upload
        static TURN: AtomicU64 = AtomicU64::new(0);
        let by_rows = match row_uploads() {
            RowUploads::On => true,
            RowUploads::Off => false,
            RowUploads::Alternate => TURN.fetch_add(1, Ordering::Relaxed) % 2 == 0,
        };
        let t0 = std::time::Instant::now();
        let n = written.len() as u64;
        let rest = if by_rows { own_work(|| upload_rows(written)) } else { written };
        let by_rows = by_rows && rest.is_empty();
        upload_all(rest);
        let k = by_rows as usize;
        CPU_WRITE_UPLOAD_US[k].fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        CPU_WRITE_UPLOADS_BY[k].fetch_add(n, Ordering::Relaxed);
    }
    upload_all(todo);
}

/// `XCL_TITLE_ROW_UPLOADS=0`: textures the title's CPU rewrites are uploaded as every other
/// texture (read, detiled, staged in a buffer made for the upload); `=ab`: every other time, to
/// compare the two ways in one run (`cpu_write_uploads`).
#[derive(Clone, Copy, PartialEq)]
enum RowUploads {
    On,
    Off,
    Alternate,
}

fn row_uploads() -> RowUploads {
    static MODE: std::sync::OnceLock<RowUploads> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XCL_TITLE_ROW_UPLOADS").as_deref().map(str::trim) {
        Ok("0") => RowUploads::Off,
        Ok("ab") => RowUploads::Alternate,
        _ => RowUploads::On,
    })
}

/// The staging buffer of a texture that is uploaded row by row (`upload_rows`): kept for the
/// texture, mapped for good.
struct Stage {
    buffer: ID3D12Resource,
    mapped: usize,
    fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

// (COM pointers of XCL's own objects, used under the map's lock)
unsafe impl Send for Stage {}

/// PC texture → its staging buffer
static STAGES: Mutex<BTreeMap<usize, Stage>> = Mutex::new(BTreeMap::new());

/// What `upload_rows` records with, per device: reused, as each upload waits for the GPU.
struct Recorder {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
}

unsafe impl Send for Recorder {}

static RECORDERS: Mutex<BTreeMap<usize, Recorder>> = Mutex::new(BTreeMap::new());

/// The rows of a texture that is a single linear level (no mips, no array, no volume): (offset
/// of the first row from the texture's address, bytes from a row to the next, bytes of a row,
/// rows).
fn single_level_rows(tex: &xg::Texture) -> Option<(usize, usize, usize, usize)> {
    let s = tex.texels()?;
    if s.depth > 0 || s.array != 1 || s.chain.levels.len() != 1 {
        return None;
    }
    s.chain.linear_rows(0)
}

/// Bytes of the title's memory for an upload: where the CPU has the page a GPU view shows at
/// their address (`gpu_views`), else at the address itself where that is readable. What was
/// last found is kept – a row is a fraction of a page, and asking Windows about an address in
/// a reservation of gigabytes is slow under Wine (it measures the whole region: a row at a
/// time, a 4K frame's planes took 47 ms that way).
struct TitleMemory {
    /// the 64 KiB page last asked for and where its view is read (0: it is in no view, or the
    /// page shown there cannot be read)
    view: (usize, usize, bool),
    /// the region Windows last described for an address itself: [start, end), readable
    direct: (usize, usize, bool),
    /// the same for the place of a view's page
    source: (usize, usize, bool),
}

impl TitleMemory {
    const PAGE: usize = crate::hle::kernelx::PHYS_PAGE_SIZE;

    fn new() -> TitleMemory {
        TitleMemory { view: (usize::MAX, 0, false), direct: (0, 0, false), source: (0, 0, false) }
    }

    /// The region around `a` (kept in `cache`): its end and whether it can be read.
    unsafe fn region(cache: &mut (usize, usize, bool), a: usize) -> (usize, bool) {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
        if a < cache.0 || a >= cache.1 {
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            *cache = if VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 {
                (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize, m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0)
            } else {
                (a & !0xFFF, (a & !0xFFF) + 0x1000, false)
            };
        }
        (cache.1, cache.2)
    }

    /// Where the byte at `at` is read (0: nowhere) and how many bytes from it are in the same
    /// place (at least 1).
    unsafe fn locate(&mut self, at: usize) -> (usize, usize) {
        let page = at & !(Self::PAGE - 1);
        if self.view.0 != page {
            let src = super::gpu_views::memory_at(page);
            // (a view's page is whole: 64 KiB the title mapped at once)
            let readable = src.is_some_and(|s| {
                let (end, ok) = Self::region(&mut self.source, s);
                ok && end >= s + Self::PAGE
            });
            self.view = (page, if readable { src.unwrap_or(0) } else { 0 }, src.is_some());
        }
        if self.view.2 {
            return (if self.view.1 != 0 { self.view.1 + (at - page) } else { 0 }, page + Self::PAGE - at);
        }
        let (end, ok) = Self::region(&mut self.direct, at);
        (if ok { at } else { 0 }, end.min(page + Self::PAGE).max(at + 1) - at)
    }

    /// `n` bytes at `at` to `dst`; what is nowhere stays as `dst` has it. Returns whether all
    /// of it was read.
    unsafe fn read(&mut self, mut at: usize, mut dst: *mut u8, mut n: usize) -> bool {
        let mut all = true;
        while n > 0 {
            let (src, same) = self.locate(at);
            let chunk = n.min(same);
            match src {
                0 => all = false,
                src => std::ptr::copy_nonoverlapping(src as *const u8, dst, chunk),
            }
            at += chunk;
            dst = dst.add(chunk);
            n -= chunk;
        }
        all
    }
}

/// Uploads of textures that are one linear level, their rows copied straight from the title's
/// memory into a staging buffer kept for each (no detiled copy, no buffer made per upload: a
/// video's three planes of a 4K frame are 12 MiB thirty times a second). One list a device,
/// then the wait for the GPU as for every upload. Returns the uploads that cannot go this way.
unsafe fn upload_rows(todo: Vec<(ID3D12Resource, Upload)>) -> Vec<(ID3D12Resource, Upload)> {
    let mut rest = Vec::new();
    let mut memory = TitleMemory::new();
    // the textures recorded: a reference each until the lists have run, and let go only after
    // the locks here are (a last reference's release comes back to `forget`)
    let mut kept: Vec<ID3D12Resource> = Vec::new();
    // the device whose list is being recorded
    let mut open: Option<usize> = None;
    let mut stages = STAGES.lock().unwrap();
    let mut recorders = RECORDERS.lock().unwrap();
    let flush = |open: &mut Option<usize>, recorders: &mut BTreeMap<usize, Recorder>| {
        let Some(device) = open.take() else { return };
        let Some(r) = recorders.get_mut(&device) else { return };
        let d = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { return };
        let t0 = std::time::Instant::now();
        let mut run = || -> Result<(), String> {
            r.list.Close().map_err(|e| e.to_string())?;
            let queue = upload_queue(dev, device)?;
            queue.ExecuteCommandLists(&[Some(r.list.cast().map_err(|e: windows::core::Error| e.to_string())?)]);
            r.value += 1;
            queue.Signal(&r.fence, r.value).map_err(|e| e.to_string())?;
            crate::gpu::fence::wait(&r.fence, r.value, dev, None, 10_000).map_err(|e| e.to_string())
        };
        if let Err(e) = run() {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("title texture: upload by rows failed: {e}");
            }
        }
        ROWS_GPU_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    };
    for (res, (device, va, tex, state)) in todo {
        let key = res.as_raw() as usize;
        let d = device as *mut c_void;
        let (Some(rows), Some(dev)) = (single_level_rows(&tex), ID3D12Device::from_raw_borrowed(&d)) else {
            rest.push((res, (device, va, tex, state)));
            continue;
        };
        let (first, stride, row_bytes, count) = rows;
        // its staging buffer (made at its first upload this way)
        if !stages.contains_key(&key) {
            let rd = res.GetDesc();
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let (mut n, mut bytes, mut total) = (0u32, 0u64, 0u64);
            dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut n), Some(&mut bytes), Some(&mut total));
            let made = if rd.MipLevels != 1 || rd.DepthOrArraySize != 1 || n as usize != count || bytes as usize != row_bytes || total == 0 || total == u64::MAX {
                None
            } else {
                texture::buffer(dev, total).ok().and_then(|buffer| {
                    let mut p = std::ptr::null_mut();
                    buffer.Map(0, None, Some(&mut p)).ok()?;
                    (!p.is_null()).then(|| Stage { buffer, mapped: p as usize, fp })
                })
            };
            match made {
                Some(stage) => {
                    stages.insert(key, stage);
                }
                None => {
                    rest.push((res, (device, va, tex, state)));
                    continue;
                }
            }
        }
        let stage = &stages[&key];
        // the list of this device (another device's is run first)
        if open.is_some_and(|o| o != device) {
            flush(&mut open, &mut recorders);
        }
        if open.is_none() {
            let ready = match recorders.get(&device) {
                Some(r) => r.alloc.Reset().is_ok() && r.list.Reset(&r.alloc, None).is_ok(),
                None => {
                    let made = || -> Result<Recorder, windows::core::Error> {
                        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                        Ok(Recorder { alloc, list, fence, value: 0 })
                    };
                    match made() {
                        Ok(r) => {
                            recorders.insert(device, r);
                            true
                        }
                        Err(_) => false,
                    }
                }
            };
            if !ready {
                rest.push((res, (device, va, tex, state)));
                continue;
            }
            open = Some(device);
        }
        let pitch = stage.fp.Footprint.RowPitch as usize;
        let t0 = std::time::Instant::now();
        for y in 0..count {
            memory.read(va + first + y * stride, (stage.mapped + stage.fp.Offset as usize + y * pitch) as *mut u8, row_bytes.min(pitch));
        }
        ROWS_COPY_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        let list = &recorders[&device].list;
        if state != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[texture::transition(&res, state, D3D12_RESOURCE_STATE_COPY_DEST)]);
        }
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&stage.buffer), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: stage.fp } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        if state != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[texture::transition(&res, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
        UPLOADS.fetch_add(1, Ordering::Relaxed);
        kept.push(res);
    }
    flush(&mut open, &mut recorders);
    drop(recorders);
    drop(stages);
    drop(kept);
    rest
}

/// staging bytes of one batch at most
const BATCH_BYTES: usize = 256 << 20;

/// The uploads `todo`, in batches of up to `BATCH_BYTES` (one list and one wait each).
unsafe fn upload_all(todo: Vec<(ID3D12Resource, Upload)>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    let mut todo = todo.into_iter().peekable();
    while todo.peek().is_some() {
        let mut batch = Vec::new();
        let mut bytes = 0usize;
        let mut device = 0usize;
        while let Some((_, (dev, _, tex, _))) = todo.peek() {
            if !batch.is_empty() && (bytes + tex.size() as usize > BATCH_BYTES || *dev != device) {
                break;
            }
            let (r, (dev, va, tex, state)) = todo.next().unwrap();
            device = dev;
            bytes += tex.size() as usize;
            let subs = subresources(va, &tex);
            if super::present::trace_frame() {
                note_content(va, &r, &subs);
            }
            batch.push((r, subs, state, state));
        }
        let d = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { continue };
        let queue = match upload_queue(dev, device) {
            Ok(q) => q,
            Err(e) => {
                warn(&e);
                continue;
            }
        };
        match own_work(|| texture::upload_batch(dev, &queue, &batch)) {
            Ok(failed) => {
                for (_, e) in failed {
                    warn(&e);
                }
                UPLOADS.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
            Err(e) => warn(&e),
        }
    }
}

/// `XCL_TRACE_FRAME`: how much of a title texture's top level is not zero, each time it is
/// uploaded (the first 48 uploads of textures of 64x64 and more).
fn note_content(va: usize, r: &ID3D12Resource, subs: &[texture::Subresource]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let Some(top) = subs.first() else { return };
    let d = unsafe { r.GetDesc() };
    if d.Width < 64 || d.Height < 64 || N.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let nonzero = top.data.iter().filter(|&&b| b != 0).count();
    crate::xlog_info!(
        "frame trace: title texture at {va:#x} ({}x{} format {}) uploaded: {nonzero} of {} bytes of level 0 not zero",
        d.Width,
        d.Height,
        d.Format.0,
        top.data.len()
    );
}

/// `XCL_TRACE_FRAME`: what the GPU holds in level 0 of title texture `res` (read back on XCL's
/// queue; the texture is taken to be in a shader-resource state) against the CPU's view of the
/// memory now. Each texture once, four at most.
pub unsafe fn probe_gpu_content(res: usize) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&res) || g.len() >= 4 {
            return;
        }
        g.push(res);
    }
    let Some((device, va, tex, gpu_written)) = TEXTURES.lock().unwrap().get(&res).map(|e| (e.device, e.va, Arc::clone(&e.tex), e.gpu_written)) else { return };
    let d = device as *mut c_void;
    let r = res as *mut c_void;
    let (Some(dev), Some(r)) = (ID3D12Device::from_raw_borrowed(&d), ID3D12Resource::from_raw_borrowed(&r)) else { return };
    let rd = r.GetDesc();
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
    dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    let result = own_work(|| -> Result<Vec<u8>, String> {
        let queue = upload_queue(dev, device)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| e.to_string())?;
        let rb = rb.ok_or("no readback buffer")?;
        let barrier = |a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: 0, StateBefore: a, StateAfter: b }),
            },
        };
        let srv = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        list.ResourceBarrier(&[barrier(srv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, srv)]);
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
        texture::wait(dev, &queue)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
        for y in 0..rows as usize {
            out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
        }
        rb.Unmap(0, None);
        Ok(out)
    });
    let cpu = subresources(va, &tex).into_iter().next().map(|s| s.data).unwrap_or_default();
    match result {
        Ok(gpu) => {
            let nonzero = gpu.iter().filter(|&&b| b != 0).count();
            let same = gpu.iter().zip(cpu.iter()).filter(|(a, b)| a == b).count();
            crate::xlog_info!(
                "frame trace: GPU content of title texture {res:#x} at {va:#x} ({}x{} format {}, {} mips, GPU-written {gpu_written}): {nonzero} of {} bytes of level 0 not zero; {same} equal to the CPU view ({} of its bytes not zero); first bytes {:02x?}",
                rd.Width,
                rd.Height,
                rd.Format.0,
                rd.MipLevels,
                gpu.len(),
                cpu.iter().filter(|&&b| b != 0).count(),
                &gpu[..gpu.len().min(16)]
            );
        }
        Err(e) => crate::xlog_warn!("frame trace: GPU content of title texture {res:#x}: {e}"),
    }
}

/// XCL's queue for uploads on `dev`.
unsafe fn upload_queue(dev: &ID3D12Device, device: usize) -> Result<ID3D12CommandQueue, String> {
    let mut q = QUEUES.lock().unwrap();
    if let Some(q) = q.get(&device) {
        return Ok(q.clone());
    }
    let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
    q.insert(device, made.clone());
    Ok(made)
}

/// Ranges closer than this are scanned as one (a gap costs little to scan; a call per texture –
/// thousands of them – a lot)
const WATCH_GAP: usize = 1 << 20;

/// The parts of `wanted` (start, end) inside `regions` (base, size), merged (gaps under
/// `WATCH_GAP` too) and sorted, as (base, size) – each within one region (a write-watch call
/// covers one allocation).
fn watch_ranges(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in wanted {
        match merged.last_mut() {
            Some(last) if a <= last.1.saturating_add(WATCH_GAP) => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut out = Vec::new();
    for (a, b) in merged {
        for &(base, size) in regions {
            let (lo, hi) = (a.max(base), b.min(base + size));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped). An address the title only mapped for the GPU (a page
/// pool's GPU view, `gpu_views`: nothing is there for the CPU) gives what the page mapped
/// there holds where the CPU has it.
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let readable = |m: &MEMORY_BASIC_INFORMATION| m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
    let mut out = vec![0u8; len];
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        if readable(&m) {
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr().add(p - at), end - p);
        } else if super::gpu_views::any() {
            // a 64 KiB page at a time: each has its own place
            const PAGE: usize = crate::hle::kernelx::PHYS_PAGE_SIZE;
            let mut q = p;
            while q < end {
                let n = ((q & !(PAGE - 1)) + PAGE).min(end) - q;
                if let Some(src) = super::gpu_views::memory_at(q) {
                    let mut s: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    if VirtualQuery(src as *const c_void, &mut s, std::mem::size_of_val(&s)) != 0 && readable(&s) && s.BaseAddress as usize + s.RegionSize >= src + n {
                        std::ptr::copy_nonoverlapping(src as *const u8, out.as_mut_ptr().add(q - at), n);
                        VIEW_BYTES.fetch_add(n as u64, Ordering::Relaxed);
                    }
                }
                q += n;
            }
        }
        p = end;
    }
    out
}

/// bytes of textures read through a page pool's GPU view so far (`read_committed`)
static VIEW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes of title textures read through a page pool's GPU view so far.
pub fn view_bytes() -> u64 {
    VIEW_BYTES.load(Ordering::Relaxed)
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        let mem = read_committed(va, tex.size() as usize);
        return c.levels.iter().enumerate().map(|(i, l)| Subresource { depth: l.depth, rows: l.height, row_bytes: l.width * bpe, data: c.detile_volume(&mem, i) }).collect();
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        let slice = read_committed(va + s * c.slice_bytes, c.slice_bytes);
        for (i, l) in c.levels.iter().enumerate() {
            out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(&slice, i, s) });
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Only the wanted parts of the watched regions are scanned, merged, one region each.
    #[test]
    fn watch_ranges_clip_and_merge() {
        let regions = [(0x10000, 0x10000), (0x40000, 0x10000), (0x1000_0000, 0x1000_0000)];
        let wanted = vec![(0x12000, 0x13000), (0x11000, 0x12800), (0x1F000, 0x41000), (0x60000, 0x61000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x11000, 0xF000), (0x40000, 0x10000)]);
        // ranges a gap of under 1 MiB apart: one scan; further apart: two
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1008_0000, 0x1008_1000), (0x1100_0000, 0x1100_1000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x1000_0000, 0x8_1000), (0x1100_0000, 0x1000)]);
    }
}
