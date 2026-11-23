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
    /// its memory is not write-watched (ordinary memory of the title – its heaps, pages it maps
    /// itself; on a host where graphics memory by protection stays the system's, `kernelx`):
    /// CPU writes are found by content (`poll_unwatched`)
    unwatched: bool,
    /// `content_hash` of the memory at the last upload (unwatched textures)
    sent: u64,
    /// the present count when the content last changed (unwatched: it is looked at before
    /// every submit for a while after)
    changed: u64,
    /// its content changed after the first upload (unwatched: a texture the CPU keeps writing
    /// – a glyph cache, a video plane – is looked at every frame from then on)
    dynamic: bool,
    /// the title unmapped it since the last look (unwatched: it is compared before the next
    /// submit – what was written while it was mapped goes up then; a texture the title only
    /// read through the pointer, a target the GPU wrote, is the same as before and stays as it
    /// is). No more than that: a retail game maps its glyph cache once, unmaps it and writes
    /// through the pointer for the rest of the run.
    unmapped: bool,
    /// the title mapped it at least once: one it means to write (unwatched: these have a turn
    /// of their own until their content first changes)
    mapped: bool,
    /// the present count when it was made (unwatched: young textures have a turn of their own)
    made: u64,
    /// the title's CPU wrote its memory and XCL was told (`note_cpu_write`): uploaded again
    /// before the title's next submit, watched memory or not
    cpu_written: bool,
    /// its writer tells XCL of its writes (`note_cpu_write` was called for it at least once: a
    /// video decoder's plane): uploaded when told and only then – it is left out of the
    /// content comparison (`poll_unwatched`), which would read every byte of a plane that
    /// changes every frame, 12 MiB a 4K frame, to learn what the decoder already said
    told: bool,
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
    let size = tex.size() as usize;
    // (a page pool's GPU view is never written at its own addresses: the CPU writes the pages
    // where it has them, so a write watch there, if there is one, says nothing)
    let unwatched = !is_watched(va, size) || super::gpu_views::in_range(va, size);
    if unwatched {
        note_unwatched(res, va, size);
    } else if trace_unwatched() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 300 {
            crate::xlog_info!("title texture trace: texture at {va:#x}, {size} bytes – write-watched, frame {}", frame_now());
        }
    }
    let e = Entry { device: device as usize, va, tex: Arc::new(tex), state, pending: true, gpu_written: false, unwatched, sent: 0, changed: 0, dynamic: false, unmapped: false, mapped: false, made: frame_now(), cpu_written: false, told: false };
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

/// The title unmapped texture `res` (`Map` gave it the texture's place in its memory,
/// `real::res_map`): one it means to write. In memory that is not write-watched it is compared
/// before the next submit and has its turn among the mapped from now on (`pick`).
pub fn note_unmap(res: usize) {
    let mut t = TEXTURES.lock().unwrap();
    let Some(e) = t.get_mut(&res) else { return };
    e.mapped = true;
    e.unmapped = true;
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if trace_unwatched() && (k < 60 || k.is_power_of_two()) {
        crate::xlog_info!("title texture trace: Unmap #{k} of the texture at {:#x} ({} bytes), frame {}", e.va, e.tex.size(), frame_now());
    }
}

/// The traces: what the title's memory of texture `res` holds now – bytes that are not zero
/// among 16 samples of 4 KiB, its content hash and the hash last uploaded.
pub fn memory_summary(res: usize) -> String {
    let Some((va, size, sent, flags)) = TEXTURES.lock().unwrap().get(&res).map(|e| (e.va, e.tex.size() as usize, e.sent, (e.unwatched, e.gpu_written, e.dynamic))) else { return String::new() };
    let nonzero: usize = (0..16usize)
        .map(|k| va + size.saturating_sub(4096) * k / 15)
        .filter(|&a| crate::gpu::shader::readable(a, 4096.min(size)))
        .map(|a| unsafe { std::slice::from_raw_parts(a as *const u8, 4096.min(size)).iter().filter(|&&b| b != 0).count() })
        .sum();
    format!(" [memory: {nonzero} of 65536 sampled bytes not zero, hash {:016x}, sent {sent:016x}, (unwatched, GPU-written, dynamic) {flags:?}]", unsafe { content_hash(va, size) })
}


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
        e.told = true;
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
    // textures in memory that is not write-watched: by content
    let mut todo = poll_unwatched();
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 && CPU_WRITTEN.load(Ordering::Relaxed) == 0 {
        upload_all(todo);
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let mut written: Vec<(ID3D12Resource, Upload)> = Vec::new();
    let first: Vec<(ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.iter_mut()
            .filter(|(_, e)| {
                if e.cpu_written || e.pending {
                    return true;
                }
                if e.gpu_written || e.unwatched {
                    return false;
                }
                let i = dirty.partition_point(|&p| p + page as usize <= e.va);
                dirty.get(i).is_some_and(|&p| p < e.va + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                if e.pending {
                    e.pending = false;
                    PENDING.fetch_sub(1, Ordering::Relaxed);
                    if e.unwatched {
                        // what goes up now is what later looks compare with
                        e.sent = content_hash(e.va, e.tex.size() as usize);
                    }
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
    // (a texture in more than one list goes once: told of in the meantime, or made and changed)
    todo.retain(|(r, _)| !first.iter().chain(written.iter()).any(|(f, _)| f.as_raw() == r.as_raw()));
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
    todo.extend(first);
    upload_all(todo);
}

/// The count of presented frames (the clock of `poll_unwatched`).
fn frame_now() -> u64 {
    super::present::counters().1
}

/// Whether `[va, va + size)` lies in write-watched memory (a unified-memory window or one of the
/// title's graphics reservations): `GetWriteWatch` finds the CPU's writes there.
fn is_watched(va: usize, size: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let inside = |&(b, n): &(usize, usize)| b <= va && va + size <= b + n;
    u.watch_regions().iter().any(inside) || crate::hle::kernelx::graphics_reservations().iter().any(inside)
}

/// `XCL_TRACE_TITLE_TEXTURES=1`: every texture in memory that is not write-watched and every
/// upload after a change of its content is logged (the first 2000 / 600), and every 256th
/// look's totals.
pub(super) fn trace_unwatched() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_TITLE_TEXTURES").is_some())
}

fn note_unwatched(res: *mut c_void, va: usize, size: usize) {
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k == 0 {
        crate::xlog_info!("title texture: textures in memory that is not write-watched (first at {va:#x}, {size} bytes) are compared with what was last uploaded before the title's submits");
    }
    if trace_unwatched() && k < 2000 {
        let d = unsafe { ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()) };
        crate::xlog_info!(
            "title texture trace: #{k} at {va:#x}, {size} bytes, {:?} (dimension, width, height, depth / array, mips, format, flags) – not write-watched, frame {}",
            d.map(|d| (d.Dimension.0, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format.0, d.Flags.0)),
            frame_now()
        );
    }
}

/// A texture is looked at before every submit for this many presented frames after its content
/// last changed (a glyph cache while a page comes up, a streamed texture being filled).
const HOT_FRAMES: u64 = 16;
/// bytes of such textures compared before one submit at most (the rest wait their turn)
const HOT_BYTES: usize = 64 << 20;
/// bytes of the textures that never changed compared before one submit, in turn (at least one
/// texture): one is looked at again after (their bytes / this) submits – with a retail game's
/// front end, 970 MiB in 815 textures, after about 160 frames. There are three turns of this
/// size: the textures the title mapped (it means to write them: a glyph cache is found within
/// a few frames of its first glyphs), those made in the last `YOUNG_FRAMES` frames (one filled
/// after it was made shows sooner), and the rest.
const TURN_BYTES: usize = 2 << 20;
const YOUNG_FRAMES: u64 = 300;
/// A texture of a page pool (placed in the pool's GPU view, read where the CPU has its pages)
/// whose content changed is looked at once a frame for this many presented frames after its
/// last change, then in turn again: what fills these is streaming (a few writes, then none),
/// and a front end has hundreds of MiB of them.
const POOL_FRAMES: u64 = 60;

/// What `pick` needs of a texture in memory that is not write-watched.
#[derive(Clone, Copy)]
struct Seen {
    /// its key in `TEXTURES` (the turns go by it)
    key: usize,
    size: usize,
    dynamic: bool,
    mapped: bool,
    /// unmapped since the last look
    now: bool,
    changed: u64,
    made: u64,
    /// placed in a page pool's GPU view (streamed textures: `POOL_FRAMES`)
    pool: bool,
}

/// The textures of `all` (ascending keys) to compare before this submit, as indices: those
/// just unmapped or whose content changed in the last `HOT_FRAMES` frames (a page pool's: not
/// these – once a frame is what streaming needs); at a frame's first submit (`new_frame`) those
/// that ever changed (a page pool's: in the last `POOL_FRAMES` frames, `poll_unwatched`); and
/// of each turn (`cursors`: the key each stopped at) the next `TURN_BYTES`, at least one texture.
fn pick(all: &[Seen], frame: u64, new_frame: bool, cursors: &mut [usize; 3]) -> Vec<usize> {
    let mut look = Vec::new();
    let mut hot_bytes = 0usize;
    // (the rest, the young, the mapped)
    let mut turns: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, e) in all.iter().enumerate() {
        let turn = if e.mapped {
            2
        } else if frame.saturating_sub(e.made) < YOUNG_FRAMES {
            1
        } else {
            0
        };
        let hot = e.dynamic && !e.pool && frame.saturating_sub(e.changed) < HOT_FRAMES;
        if e.now || hot || (e.dynamic && new_frame) {
            if e.now || hot_bytes + e.size <= HOT_BYTES {
                hot_bytes += e.size;
                look.push(i);
            } else {
                turns[turn].push(i);
            }
        } else if !e.dynamic {
            turns[turn].push(i);
        }
    }
    for (turn, cursor) in turns.iter().zip(cursors.iter_mut()) {
        let start = turn.partition_point(|&i| all[i].key <= *cursor);
        let mut bytes = 0usize;
        for k in 0..turn.len() {
            let i = turn[(start + k) % turn.len()];
            if k > 0 && bytes + all[i].size > TURN_BYTES {
                break;
            }
            bytes += all[i].size;
            *cursor = all[i].key;
            look.push(i);
        }
    }
    look
}

/// A hash of `len` bytes at `at` for telling a change: four chains of 64-bit words (each step a
/// bijection of its chain, so a change within one word always shows). The bytes are the ones an
/// upload reads (`read_committed`): where the CPU has the page a GPU view shows at an address,
/// else at the address itself; what is nowhere (not committed, not readable, a view with no
/// page) counts by its length only – an upload reads it as zero.
unsafe fn content_hash(at: usize, len: usize) -> u64 {
    content_hash_in(&mut TitleMemory::new(), at, len)
}

#[cfg(test)]
pub(super) unsafe fn content_hash_for_tests(at: usize, len: usize) -> u64 {
    content_hash(at, len)
}

/// `content_hash` with the places of memory found so far (`memory`): one look at many textures
/// asks Windows about a region once.
unsafe fn content_hash_in(memory: &mut TitleMemory, at: usize, len: usize) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut l = [K, K.rotate_left(13), K.rotate_left(29), K.rotate_left(47)];
    let mut p = at;
    while p < at + len {
        let (src, same) = memory.locate(p);
        let n = same.min(at + len - p);
        if src != 0 {
            let b = std::slice::from_raw_parts(src as *const u8, n);
            let mut chunks = b.chunks_exact(32);
            for c in &mut chunks {
                for (i, lane) in l.iter_mut().enumerate() {
                    let w = u64::from_le_bytes(c[8 * i..8 * i + 8].try_into().unwrap());
                    *lane = (*lane ^ w).wrapping_mul(K).rotate_left(29);
                }
            }
            for (i, &x) in chunks.remainder().iter().enumerate() {
                l[i & 3] = (l[i & 3] ^ x as u64).wrapping_mul(K).rotate_left(29);
            }
        } else {
            l[0] = (l[0] ^ n as u64 ^ 0x5555).wrapping_mul(K).rotate_left(29);
        }
        p += n;
    }
    let mut h = l[0];
    for x in &l[1..] {
        h = (h.rotate_left(23) ^ x).wrapping_mul(K);
    }
    h ^ h >> 31
}

/// looks of `poll_unwatched` that compared something, the bytes they hashed, the time that took
/// and the uploads that followed a change
static LOOKS: AtomicU64 = AtomicU64::new(0);
static HASHED: AtomicU64 = AtomicU64::new(0);
static NANOS: AtomicU64 = AtomicU64::new(0);
static CHANGED: AtomicU64 = AtomicU64::new(0);

/// What keeping title textures in step cost since this was last asked (the frame rate's log
/// line asks every 300 frames), as text: the content comparison (looks, MiB hashed, ms, uploads
/// after a change), all uploads through the common path (MiB, ms) and the uploads by rows of
/// textures whose writer tells of its writes (textures, ms).
pub fn cost_since_last() -> String {
    static LAST: Mutex<[u64; 8]> = Mutex::new([0; 8]);
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let now = [l(&LOOKS), l(&HASHED), l(&NANOS), l(&CHANGED), l(&UPLOAD_BYTES), l(&UPLOAD_NANOS), l(&CPU_WRITE_UPLOADS_BY[1]), l(&CPU_WRITE_UPLOAD_US[1])];
    let mut last = LAST.lock().unwrap();
    let d: Vec<u64> = now.iter().zip(last.iter()).map(|(n, o)| n.saturating_sub(*o)).collect();
    *last = now;
    format!(
        "{} looks compared {} MiB in {} ms ({} changed); uploads {} MiB in {} ms; {} by rows in {} ms",
        d[0],
        d[1] >> 20,
        d[2] / 1_000_000,
        d[3],
        d[4] >> 20,
        d[5] / 1_000_000,
        d[6],
        d[7] / 1000
    )
}

/// Textures in memory that is not write-watched whose content differs from what was last
/// uploaded. Those that changed in the last `HOT_FRAMES` frames are compared before every
/// submit (what a list of this submit samples is complete by now – on the console the GPU reads
/// the same memory); those that ever changed after their first upload before the first submit
/// of every frame; the others in turn (`pick`). Left out: textures whose writer tells XCL of its
/// writes (`note_cpu_write`) – they go up when told.
unsafe fn poll_unwatched() -> Vec<(ID3D12Resource, Upload)> {
    static LAST_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static UNWATCHED: AtomicU64 = AtomicU64::new(0);
    static CURSORS: Mutex<[usize; 3]> = Mutex::new([0; 3]);
    static TOLD: AtomicU64 = AtomicU64::new(0);
    let frame = frame_now();
    let new_frame = LAST_FRAME.swap(frame, Ordering::Relaxed) != frame;
    // (texture, address, bytes, hash last sent)
    let (look, count, total): (Vec<(usize, usize, usize, u64)>, usize, usize) = {
        let mut t = TEXTURES.lock().unwrap();
        let mut all: Vec<Seen> = Vec::new();
        let mut places: Vec<(usize, u64)> = Vec::new();
        let mut told = 0u64;
        for (&res, e) in t.iter_mut() {
            let now = std::mem::take(&mut e.unmapped);
            if !e.unwatched || e.pending || e.gpu_written {
                continue;
            }
            if e.told {
                told += 1;
                continue;
            }
            let size = e.tex.size() as usize;
            let pool = super::gpu_views::in_range(e.va, size);
            if pool && e.dynamic && frame.saturating_sub(e.changed) >= POOL_FRAMES {
                // streamed in and quiet since: back to its turn
                e.dynamic = false;
            }
            all.push(Seen { key: res, size, dynamic: e.dynamic, mapped: e.mapped, now, changed: e.changed, made: e.made, pool });
            places.push((e.va, e.sent));
        }
        let picked = pick(&all, frame, new_frame, &mut CURSORS.lock().unwrap());
        TOLD.store(told, Ordering::Relaxed);
        (picked.into_iter().map(|i| (all[i].key, places[i].0, all[i].size, places[i].1)).collect(), all.len(), all.iter().map(|e| e.size).sum())
    };
    UNWATCHED.store(count as u64, Ordering::Relaxed);
    if look.is_empty() {
        return Vec::new();
    }
    let t0 = std::time::Instant::now();
    let mut changed: Vec<(usize, u64)> = Vec::new();
    let mut hashed = 0usize;
    let mut memory = TitleMemory::new();
    for &(res, va, size, sent) in &look {
        let h = content_hash_in(&mut memory, va, size);
        hashed += size;
        if h != sent {
            changed.push((res, h));
        }
    }
    HASHED.fetch_add(hashed as u64, Ordering::Relaxed);
    NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let looks = LOOKS.fetch_add(1, Ordering::Relaxed) + 1;
    if trace_unwatched() && looks % 256 == 0 {
        crate::xlog_info!(
            "title texture trace: {looks} looks at textures in unwatched memory ({count} textures, {} KiB; {} more told of by their writer and not compared): {} MiB compared in {} ms, {} uploads after a change; all uploads {} MiB in {} ms; frame {frame}",
            total >> 10,
            TOLD.load(Ordering::Relaxed),
            HASHED.load(Ordering::Relaxed) >> 20,
            NANOS.load(Ordering::Relaxed) / 1_000_000,
            CHANGED.load(Ordering::Relaxed),
            UPLOAD_BYTES.load(Ordering::Relaxed) >> 20,
            UPLOAD_NANOS.load(Ordering::Relaxed) / 1_000_000
        );
    }
    if changed.is_empty() {
        return Vec::new();
    }
    let mut t = TEXTURES.lock().unwrap();
    changed
        .into_iter()
        .filter_map(|(res, h)| {
            let e = t.get_mut(&res)?;
            if e.gpu_written {
                return None;
            }
            e.sent = h;
            e.changed = frame;
            e.dynamic = true;
            let k = CHANGED.fetch_add(1, Ordering::Relaxed);
            let p = res as *mut c_void;
            let r = ID3D12Resource::from_raw_borrowed(&p)?.clone();
            if trace_unwatched() && (k < 600 || k.is_power_of_two()) {
                let d = r.GetDesc();
                crate::xlog_info!("title texture trace: content of the texture at {:#x} ({}x{} format {}, {} bytes) changed – upload #{k}, frame {frame}", e.va, d.Width, d.Height, d.Format.0, e.tex.size());
            }
            Some((r, (e.device, e.va, Arc::clone(&e.tex), e.state)))
        })
        .collect()
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
    if todo.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    UPLOAD_BYTES.fetch_add(todo.iter().map(|(_, u)| u.2.size()).sum::<u64>(), Ordering::Relaxed);
    upload_batches(todo, &N);
    UPLOAD_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// bytes of title memory uploaded as textures and the time that took (detiling, the copy, the
/// wait for the GPU)
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_NANOS: AtomicU64 = AtomicU64::new(0);

unsafe fn upload_batches(todo: Vec<(ID3D12Resource, Upload)>, warned: &AtomicU64) {
    let warn = |e: &str| {
        if warned.fetch_add(1, Ordering::Relaxed) < 4 {
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

    fn seen(key: usize, size: usize) -> Seen {
        Seen { key, size, dynamic: false, mapped: false, now: false, changed: 0, made: 0, pool: false }
    }

    /// Textures that never changed are all looked at within about (their bytes / `TURN_BYTES`)
    /// submits, a texture larger than a turn included.
    #[test]
    fn every_unchanged_texture_gets_its_turn() {
        let all: Vec<Seen> = (0..200).map(|k| seen(0x1000 + k, if k == 7 { 5 << 20 } else { 300 << 10 })).collect();
        let total: usize = all.iter().map(|e| e.size).sum();
        let mut cursors = [0usize; 3];
        let mut seen_at = vec![0usize; all.len()];
        let looks = total / TURN_BYTES + all.len();
        let mut picked = 0;
        for look in 1..=looks {
            for i in pick(&all, 1000, look % 3 == 1, &mut cursors) {
                if seen_at[i] == 0 {
                    seen_at[i] = look;
                    picked += 1;
                }
            }
            if picked == all.len() {
                break;
            }
        }
        assert_eq!(picked, all.len(), "all looked at");
        let last = *seen_at.iter().max().unwrap();
        // (a submit takes whole textures: at least a turn less one of them)
        assert!(last <= total / (TURN_BYTES - (300 << 10)) + 2, "after {last} submits for {} MiB", total >> 20);
        // one submit compares one turn's bytes (or its one large texture)
        let n: usize = pick(&all, 1000, false, &mut cursors).iter().map(|&i| all[i].size).sum();
        assert!(n <= TURN_BYTES.max(5 << 20));
    }

    /// A texture whose content changed is looked at before every submit for `HOT_FRAMES`
    /// frames, then before each frame's first submit; one the title mapped is found in its own
    /// turn however many others wait; the young have theirs.
    #[test]
    fn changed_mapped_and_young_textures_come_first() {
        let mut all: Vec<Seen> = (0..500).map(|k| seen(0x1000 + k, 2 << 20)).collect();
        all[300].dynamic = true;
        all[300].changed = 990;
        all[450].mapped = true; // a glyph cache: mapped once, never changed so far
        all[451].mapped = true;
        all[20].made = 900; // made 100 frames ago
        let mut cursors = [0usize; 3];
        let frame = 1000;
        // hot: every submit
        for new_frame in [true, false, false] {
            assert!(pick(&all, frame, new_frame, &mut cursors).contains(&300));
        }
        // the two mapped textures alternate: each within two submits
        let a = pick(&all, frame, false, &mut cursors);
        let b = pick(&all, frame, false, &mut cursors);
        assert!(a.contains(&450) != a.contains(&451));
        assert!(a.contains(&450) != b.contains(&450));
        // the young one: every submit (it is the only one in its turn)
        assert!(a.contains(&20) && b.contains(&20));
        // no longer hot: the first submit of a frame only
        let later = frame + HOT_FRAMES + 50;
        assert!(pick(&all, later, true, &mut cursors).contains(&300));
        assert!(!pick(&all, later, false, &mut cursors).contains(&300));
        // a changed texture is not in a turn as well
        assert_eq!(pick(&all, later, true, &mut cursors).iter().filter(|&&i| i == 300).count(), 1);
        // one just unmapped: before this submit, whatever else waits, and once
        all[77].now = true;
        assert_eq!(pick(&all, later, false, &mut cursors).iter().filter(|&&i| i == 77).count(), 1);
    }

    /// A page pool's texture that changed (streamed in) is looked at once a frame, not before
    /// every submit; one that never changed has its turn like any other.
    #[test]
    fn a_pools_changed_texture_is_looked_at_once_a_frame() {
        let mut all: Vec<Seen> = (0..500).map(|k| seen(0x1000 + k, 2 << 20)).collect();
        for e in all.iter_mut() {
            e.pool = true;
        }
        all[300].dynamic = true;
        all[300].changed = 999;
        let mut cursors = [0usize; 3];
        assert!(pick(&all, 1000, true, &mut cursors).contains(&300));
        assert!(!pick(&all, 1000, false, &mut cursors).contains(&300));
        assert!(!pick(&all, 1000, false, &mut cursors).contains(&300));
        assert!(pick(&all, 1001, true, &mut cursors).contains(&300));
        // the others: one turn's bytes a submit
        let n = pick(&all, 1001, false, &mut cursors).len();
        assert_eq!(n, 1);
    }

    /// More changed textures than `HOT_BYTES`: the rest are looked at in their turn.
    #[test]
    fn hot_textures_over_the_limit_wait_their_turn() {
        let mut all: Vec<Seen> = (0..40).map(|k| seen(0x1000 + k, 4 << 20)).collect();
        for e in all.iter_mut() {
            e.dynamic = true;
            e.changed = 100;
        }
        let mut cursors = [0usize; 3];
        let mut got = vec![false; all.len()];
        for _ in 0..40 {
            let p = pick(&all, 101, false, &mut cursors);
            assert!(p.iter().map(|&i| all[i].size).sum::<usize>() <= HOT_BYTES + (4 << 20));
            p.into_iter().for_each(|i| got[i] = true);
        }
        assert!(got.iter().all(|&g| g));
    }

    /// The content hash tells a change of one byte anywhere and reads what it is given.
    #[test]
    fn content_hash_tells_a_one_byte_change() {
        let mut m = vec![0u8; 3 * 4096 + 37];
        for (i, b) in m.iter_mut().enumerate() {
            *b = (i * 31 >> 3) as u8;
        }
        let h = unsafe { content_hash(m.as_ptr() as usize, m.len()) };
        assert_eq!(h, unsafe { content_hash(m.as_ptr() as usize, m.len()) });
        for at in [0usize, 1, 31, 32, 4095, 4096, 8191, m.len() - 38, m.len() - 1] {
            m[at] ^= 0x10;
            assert_ne!(h, unsafe { content_hash(m.as_ptr() as usize, m.len()) }, "byte {at}");
            m[at] ^= 0x10;
        }
        assert_ne!(h, unsafe { content_hash(m.as_ptr() as usize, m.len() - 1) });
    }

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
