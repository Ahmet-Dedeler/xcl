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
    /// the title unmapped it since the last upload: what it wrote while it was mapped goes up
    /// before its next submit. (No more than that: a retail game maps its glyph cache once,
    /// unmaps it and writes through the pointer for the rest of the run.)
    unmapped: bool,
    /// the title mapped it at least once: one it means to write (unwatched: these have a turn
    /// of their own until their content first changes)
    mapped: bool,
    /// the present count when it was made (unwatched: young textures have a turn of their own)
    made: u64,
}

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
    let unwatched = !is_watched(va, size);
    if unwatched {
        note_unwatched(res, va, size);
    } else if trace_unwatched() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 300 {
            crate::xlog_info!("title texture trace: texture at {va:#x}, {size} bytes – write-watched, frame {}", frame_now());
        }
    }
    let e = Entry { device: device as usize, va, tex: Arc::new(tex), state, pending: true, gpu_written: false, unwatched, sent: 0, changed: 0, dynamic: false, unmapped: false, mapped: false, made: frame_now() };
    {
        let mut t = TEXTURES.lock().unwrap();
        if let Some(old) = t.insert(res as usize, e) {
            if old.pending {
                PENDING.fetch_sub(1, Ordering::Relaxed);
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

/// textures unmapped since their last upload
static UNMAPPED: AtomicU64 = AtomicU64::new(0);

/// The title unmapped texture `res` (`Map` gave it the texture's place in its memory,
/// `real::res_map`): what it wrote there is uploaded before its next submit, whether the memory
/// is write-watched or not.
pub fn note_unmap(res: usize) {
    let mut t = TEXTURES.lock().unwrap();
    let Some(e) = t.get_mut(&res) else { return };
    e.mapped = true;
    if !e.unmapped {
        e.unmapped = true;
        UNMAPPED.fetch_add(1, Ordering::Relaxed);
    }
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

/// The textures unmapped since their last upload (a reference each).
unsafe fn take_unmapped() -> Vec<(ID3D12Resource, Upload)> {
    if UNMAPPED.load(Ordering::Relaxed) == 0 {
        return Vec::new();
    }
    let mut t = TEXTURES.lock().unwrap();
    t.iter_mut()
        .filter(|(_, e)| e.unmapped)
        .filter_map(|(&res, e)| {
            e.unmapped = false;
            UNMAPPED.fetch_sub(1, Ordering::Relaxed);
            if e.pending || e.gpu_written {
                return None; // (pending: its first upload takes the memory as it is now)
            }
            let p = res as *mut c_void;
            ID3D12Resource::from_raw_borrowed(&p).map(|r| (r.clone(), (e.device, e.va, Arc::clone(&e.tex), e.state)))
        })
        .collect()
}

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().iter().filter(|(_, e)| e.va < va + len && va < e.va + e.tex.size() as usize).map(|(&r, e)| (r, e.va)).take(4).collect()
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
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
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
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
    // textures in memory that is not write-watched: by content; those the title unmapped
    let mut todo = poll_unwatched();
    todo.extend(take_unmapped());
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 {
        upload_all(todo);
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let first: Vec<(ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.iter_mut()
            .filter(|(_, e)| {
                if e.pending {
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
                let p = res as *mut c_void;
                ID3D12Resource::from_raw_borrowed(&p).map(|r| (r.clone(), (e.device, e.va, Arc::clone(&e.tex), e.state)))
            })
            .collect()
    });
    // (a texture in both lists goes once)
    todo.retain(|(r, _)| !first.iter().any(|(f, _)| f.as_raw() == r.as_raw()));
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

/// What `pick` needs of a texture in memory that is not write-watched.
#[derive(Clone, Copy)]
struct Seen {
    /// its key in `TEXTURES` (the turns go by it)
    key: usize,
    size: usize,
    dynamic: bool,
    mapped: bool,
    changed: u64,
    made: u64,
}

/// The textures of `all` (ascending keys) to compare before this submit, as indices: those
/// whose content changed in the last `HOT_FRAMES` frames; at a frame's first submit
/// (`new_frame`) those that ever changed; and of each turn (`cursors`: the key each stopped at)
/// the next `TURN_BYTES`, at least one texture.
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
        let hot = e.dynamic && frame.saturating_sub(e.changed) < HOT_FRAMES;
        if hot || (e.dynamic && new_frame) {
            if hot_bytes + e.size <= HOT_BYTES {
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
/// bijection of its chain, so a change within one word always shows). Pages that are not
/// committed or not readable count by their length only (`read_committed` reads them as zero).
unsafe fn content_hash(at: usize, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut l = [K, K.rotate_left(13), K.rotate_left(29), K.rotate_left(47)];
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        if end <= p {
            break;
        }
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0;
        if readable {
            let b = std::slice::from_raw_parts(p as *const u8, end - p);
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
            l[0] = (l[0] ^ (end - p) as u64 ^ 0x5555).wrapping_mul(K).rotate_left(29);
        }
        p = end;
    }
    let mut h = l[0];
    for x in &l[1..] {
        h = (h.rotate_left(23) ^ x).wrapping_mul(K);
    }
    h ^ h >> 31
}

/// Textures in memory that is not write-watched whose content differs from what was last
/// uploaded. Those that changed in the last `HOT_FRAMES` frames are compared before every
/// submit (what a list of this submit samples is complete by now – on the console the GPU reads
/// the same memory); those that ever changed after their first upload before the first submit
/// of every frame; the others in turn (`pick`).
unsafe fn poll_unwatched() -> Vec<(ID3D12Resource, Upload)> {
    static LAST_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static UNWATCHED: AtomicU64 = AtomicU64::new(0);
    static CURSORS: Mutex<[usize; 3]> = Mutex::new([0; 3]);
    static LOOKS: AtomicU64 = AtomicU64::new(0);
    static HASHED: AtomicU64 = AtomicU64::new(0);
    static NANOS: AtomicU64 = AtomicU64::new(0);
    static CHANGED: AtomicU64 = AtomicU64::new(0);
    let frame = frame_now();
    let new_frame = LAST_FRAME.swap(frame, Ordering::Relaxed) != frame;
    // (texture, address, bytes, hash last sent)
    let (look, count, total): (Vec<(usize, usize, usize, u64)>, usize, usize) = {
        let t = TEXTURES.lock().unwrap();
        let mut all: Vec<Seen> = Vec::new();
        let mut places: Vec<(usize, u64)> = Vec::new();
        for (&res, e) in t.iter() {
            if !e.unwatched || e.pending || e.gpu_written {
                continue;
            }
            all.push(Seen { key: res, size: e.tex.size() as usize, dynamic: e.dynamic, mapped: e.mapped, changed: e.changed, made: e.made });
            places.push((e.va, e.sent));
        }
        let picked = pick(&all, frame, new_frame, &mut CURSORS.lock().unwrap());
        (picked.into_iter().map(|i| (all[i].key, places[i].0, all[i].size, places[i].1)).collect(), all.len(), all.iter().map(|e| e.size).sum())
    };
    UNWATCHED.store(count as u64, Ordering::Relaxed);
    if look.is_empty() {
        return Vec::new();
    }
    let t0 = std::time::Instant::now();
    let mut changed: Vec<(usize, u64)> = Vec::new();
    let mut hashed = 0usize;
    for &(res, va, size, sent) in &look {
        let h = content_hash(va, size);
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
            "title texture trace: {looks} looks at textures in unwatched memory ({count} textures, {} KiB): {} MiB compared in {} ms, {} uploads after a change; all uploads {} MiB in {} ms; frame {frame}",
            total >> 10,
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
/// has only some of its pages mapped).
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut out = vec![0u8; len];
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr().add(p - at), end - p);
        }
        p = end;
    }
    out
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
        Seen { key, size, dynamic: false, mapped: false, changed: 0, made: 0 }
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
