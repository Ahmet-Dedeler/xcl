//! The title's Bink 2 player (`Bink2Scarlett.dll`, shipped with the game and run as it is).
//!
//! A title that decodes on the CPU gives Bink its own frame buffers
//! (`BinkRegisterFrameBuffers`): the planes of textures it locked once, which Bink then writes
//! for every frame of the video with no further call into D3D. On the console the GPU samples
//! that memory. Here the calls that fill the planes are wrapped (the title's imports of the DLL
//! go through `wrap`), so the textures over the planes are uploaded again after each decoded
//! frame (`d3d12x::title_textures::note_cpu_write`) – whether or not their memory is
//! write-watched, and also when the title has the textures at another address than the one
//! it writes them through (a page pool's GPU view, `d3d12x::gpu_views`).
//!
//! `XCL_BINK_WRAP=0` (`resolver`): the DLL's functions unwrapped. `XCL_BINK_PLANES=all`: every
//! registered set of planes counts as written after a decode, not only the current one.
//! `XCL_TRACE_VIDEO=1`: the frame buffers as registered (where their memory is, which textures
//! lie over it) and what they hold while the video plays.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// `XCL_TRACE_VIDEO=1`
pub fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_VIDEO").is_some())
}

/// One plane of a frame buffer: its memory, bytes a row, rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub at: usize,
    pub pitch: usize,
    pub rows: usize,
}

impl Plane {
    pub fn bytes(&self) -> usize {
        self.pitch * self.rows
    }
}

/// The frame buffers of one open video: `frames[k]` are the planes of buffer `k` (Y, cR, cB, A,
/// H – those the video has).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Buffers {
    pub frames: Vec<Vec<Plane>>,
    /// `FrameNum`: the buffer the current frame is in
    pub current: usize,
}

/// most frame buffers a video has in this DLL (the title's own copy of the structure, at +0x58
/// of its player object, ends where the next member starts at +0x160: room for three)
const MAX_FRAMES: usize = 3;

/// `BINKFRAMEBUFFERS` as this DLL has it (from the title's code that fills it, image
/// +0x442f760): TotalFrames, YABufferWidth, YABufferHeight, cRcBBufferWidth, cRcBBufferHeight,
/// FrameNum (6 × 4 bytes), then per frame five planes of {Buffer (8), Allocate (4), BufferPitch
/// (4)} in the order Y, cR, cB, A, H.
pub fn parse(words: &[u8]) -> Option<Buffers> {
    let u = |o: usize| words.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    let q = |o: usize| words.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize);
    let total = u(0)?;
    if total == 0 || total > MAX_FRAMES {
        return None;
    }
    let (ya_h, c_h) = (u(8)?, u(16)?);
    let mut frames = Vec::new();
    for f in 0..total {
        let mut planes = Vec::new();
        for p in 0..5 {
            let o = 24 + (f * 5 + p) * 16;
            let (at, pitch) = (q(o)?, u(o + 12)?);
            if at != 0 && pitch != 0 {
                // Y and A (and H) are full size, cR and cB half
                planes.push(Plane { at, pitch, rows: if p == 1 || p == 2 { c_h } else { ya_h } });
            }
        }
        frames.push(planes);
    }
    Some(Buffers { frames, current: u(20)? })
}

/// Size of a `BINKFRAMEBUFFERS` with every frame.
const FRAME_BUFFERS_BYTES: usize = 24 + MAX_FRAMES * 5 * 16;

/// (HBINK, the title's `BINKFRAMEBUFFERS` it registered, the frame buffers in it then)
static OPEN: Mutex<Vec<(usize, usize, Buffers)>> = Mutex::new(Vec::new());

/// the DLL's own functions (what the wrappers call)
static REAL: [AtomicUsize; 6] = [const { AtomicUsize::new(0) }; 6];
const REGISTER: usize = 0;
const DO_FRAME: usize = 1;
const ASYNC_WAIT: usize = 2;
const CLOSE: usize = 3;
const ASYNC_MULTI: usize = 4;
const NEXT_FRAME: usize = 5;

/// The title's import `name` of the Bink DLL, at `real`: the address it gets instead (a
/// wrapper), or None for the function itself.
pub fn wrap(name: &str, real: usize) -> Option<usize> {
    let (slot, f) = match name {
        "BinkRegisterFrameBuffers" => (REGISTER, register_frame_buffers as usize),
        "BinkDoFrame" => (DO_FRAME, do_frame as usize),
        "BinkDoFrameAsyncWait" => (ASYNC_WAIT, do_frame_async_wait as usize),
        "BinkDoFrameAsyncMulti" => (ASYNC_MULTI, do_frame_async_multi as usize),
        "BinkNextFrame" => (NEXT_FRAME, next_frame as usize),
        "BinkClose" => (CLOSE, close as usize),
        _ => return None,
    };
    REAL[slot].store(real, Ordering::Release);
    Some(f)
}

/// The frame buffers the title registered for `bink`, with the buffer the current frame is in
/// now: Bink keeps the pointer to the registered structure and moves its `FrameNum` on with
/// every decode (the DLL's `BinkGetFrameBuffersInfo` gives 0 there throughout – seen live; the
/// title reads its own structure too). None: no frame buffers were registered for it (the
/// DLL's own are not the title's textures).
unsafe fn buffers_of(bink: usize) -> Option<Buffers> {
    let (set, mut registered) = OPEN.lock().unwrap().iter().find(|e| e.0 == bink).map(|e| (e.1, e.2.clone()))?;
    if crate::gpu::shader::readable(set, 24) {
        let w = |o: usize| ((set + o) as *const u32).read_unaligned() as usize;
        let (total, current) = (w(0), w(20));
        if total == registered.frames.len() && current < total {
            registered.current = current;
        }
    }
    Some(registered)
}

/// `XCL_BINK_PLANES=all`
fn all_planes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_BINK_PLANES").is_ok_and(|v| v.trim().eq_ignore_ascii_case("all")))
}

/// `void BinkRegisterFrameBuffers(HBINK, BINKFRAMEBUFFERS*)`
unsafe extern "system" fn register_frame_buffers(bink: usize, set: *const u8) {
    let f: unsafe extern "system" fn(usize, *const u8) = std::mem::transmute(REAL[REGISTER].load(Ordering::Acquire));
    f(bink, set);
    if set.is_null() || !crate::gpu::shader::readable(set as usize, FRAME_BUFFERS_BYTES) {
        return;
    }
    let Some(b) = parse(std::slice::from_raw_parts(set, FRAME_BUFFERS_BYTES)) else { return };
    if trace() {
        let w = |o: usize| (set.add(o) as *const u32).read_unaligned();
        crate::xlog_info!(
            "video trace: BinkRegisterFrameBuffers({bink:#x}): {} frame buffer(s), Y/A {}x{}, cR/cB {}x{}, frame {}; frame {} presented",
            w(0), w(4), w(8), w(12), w(16), w(20),
            crate::hle::d3d12x::present::counters().1
        );
        let window = trace_title_window();
        for (k, planes) in b.frames.iter().enumerate() {
            for p in planes {
                crate::xlog_info!("video trace:   buffer {k} plane at {:#x}, pitch {}, {} rows: {}", p.at, p.pitch, p.rows, crate::hle::d3d12x::title_textures::describe_memory(p.at, p.bytes()));
                // the same plane in the title's other view of its resource memory
                if let Some([cpu, size, gpu]) = window.filter(|w| w[0] != w[2] && p.at >= w[0] && p.at - w[0] < w[1]) {
                    let other = p.at - cpu + gpu;
                    crate::xlog_info!("video trace:   … in the GPU view ({size:#x} bytes) at {other:#x}: {}", crate::hle::d3d12x::title_textures::describe_memory(other, p.bytes()));
                }
            }
        }
    }
    let mut g = OPEN.lock().unwrap();
    g.retain(|e| e.0 != bink);
    g.push((bink, set as usize, b));
}

/// The title's resource memory window (image +0x9e1dac0: CPU view, size, GPU view): its
/// texture lock (image +0x5121690) turns an address in the GPU view into one in the CPU view.
unsafe fn trace_title_window() -> Option<[usize; 3]> {
    let m = crate::modules::main_module()?;
    let at = m.base + 0x9e1_dac0;
    if m.size <= 0x9e1_dad8 || !crate::gpu::shader::readable(at, 24) {
        return None;
    }
    let w = (at as *const [usize; 3]).read_unaligned();
    crate::xlog_info!("video trace: the title's resource memory: CPU view {:#x}, {:#x} bytes, GPU view {:#x}", w[0], w[1], w[2]);
    Some(w)
}

/// The planes Bink just decoded into were written by the CPU.
unsafe fn decoded(bink: usize, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let Some(b) = buffers_of(bink) else {
        if trace() && n < 8 {
            crate::xlog_info!("video trace: {what}({bink:#x}) #{n}: no frame buffers registered");
        }
        return;
    };
    // the buffer the frame was decoded into (the others hold earlier frames, as they were)
    let mut textures = 0;
    for (k, planes) in b.frames.iter().enumerate() {
        if k == b.current || b.current >= b.frames.len() || all_planes() {
            for p in planes {
                textures += crate::hle::d3d12x::title_textures::note_cpu_write(p.at, p.bytes());
            }
        }
    }
    if trace() && (n < 12 || n % 60 == 0) {
        // what the planes hold: bytes that are not zero among 4 KiB samples
        let content: Vec<String> = b
            .frames
            .iter()
            .map(|planes| {
                planes
                    .iter()
                    .map(|p| {
                        let (mut nz, mut seen) = (0usize, 0usize);
                        for k in 0..8usize {
                            let a = p.at + (p.bytes().saturating_sub(4096)) * k / 7;
                            if crate::gpu::shader::readable(a, 4096) {
                                seen += 4096;
                                nz += std::slice::from_raw_parts(a as *const u8, 4096).iter().filter(|&&x| x != 0).count();
                            }
                        }
                        format!("{nz}/{seen}")
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        crate::xlog_info!(
            "video trace: {what}({bink:#x}) #{n}, frame {} presented: current buffer {}, bytes not zero in the planes' samples: [{}]; {textures} title texture(s) over the written planes; uploads after such writes so far: {}",
            crate::hle::d3d12x::present::counters().1,
            b.current,
            content.join("] ["),
            crate::hle::d3d12x::title_textures::cpu_write_upload_times()
        );
        if n % 60 == 0 {
            crate::xlog_info!("video trace: page views: {}", crate::hle::d3d12x::gpu_views::summary());
        }
    }
}

/// `S32 BinkDoFrame(HBINK)`: decodes the current frame into the frame buffers.
unsafe extern "system" fn do_frame(bink: usize) -> i32 {
    let f: unsafe extern "system" fn(usize) -> i32 = std::mem::transmute(REAL[DO_FRAME].load(Ordering::Acquire));
    let r = f(bink);
    decoded(bink, "BinkDoFrame");
    r
}

/// videos whose decode was started on Bink's threads and not waited for yet
static ASYNC: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `S32 BinkDoFrameAsyncMulti(HBINK, U32* threads, S32 count)`: starts the decode on Bink's
/// threads.
unsafe extern "system" fn do_frame_async_multi(bink: usize, threads: *const u32, count: i32) -> i32 {
    let f: unsafe extern "system" fn(usize, *const u32, i32) -> i32 = std::mem::transmute(REAL[ASYNC_MULTI].load(Ordering::Acquire));
    let r = f(bink, threads, count);
    if trace() {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 6 || n % 120 == 0 {
            crate::xlog_info!("video trace: BinkDoFrameAsyncMulti({bink:#x}, {count} thread(s)) #{n} → {r}; frame {} presented", crate::hle::d3d12x::present::counters().1);
        }
    }
    if r != 0 {
        let mut g = ASYNC.lock().unwrap();
        if !g.contains(&bink) {
            g.push(bink);
        }
    }
    r
}

/// `S32 BinkDoFrameAsyncWait(HBINK, S32 microseconds)`: nonzero once the decode started with
/// `BinkDoFrameAsync*` is done.
unsafe extern "system" fn do_frame_async_wait(bink: usize, us: i32) -> i32 {
    let f: unsafe extern "system" fn(usize, i32) -> i32 = std::mem::transmute(REAL[ASYNC_WAIT].load(Ordering::Acquire));
    let r = f(bink, us);
    if r != 0 {
        let was = {
            let mut g = ASYNC.lock().unwrap();
            let n = g.len();
            g.retain(|&b| b != bink);
            g.len() != n
        };
        if was {
            decoded(bink, "BinkDoFrameAsyncWait");
        }
    }
    r
}

/// `void BinkNextFrame(HBINK)`: moves on to the next frame. A title waits for a decode it
/// started on Bink's threads before this; one that did not gets its planes noted here (late
/// rather than never).
unsafe extern "system" fn next_frame(bink: usize) {
    let f: unsafe extern "system" fn(usize) = std::mem::transmute(REAL[NEXT_FRAME].load(Ordering::Acquire));
    f(bink);
    let was = {
        let mut g = ASYNC.lock().unwrap();
        let n = g.len();
        g.retain(|&b| b != bink);
        g.len() != n
    };
    if was {
        decoded(bink, "BinkNextFrame");
    }
}

/// `void BinkClose(HBINK)`
unsafe extern "system" fn close(bink: usize) {
    OPEN.lock().unwrap().retain(|e| e.0 != bink);
    ASYNC.lock().unwrap().retain(|&b| b != bink);
    if trace() {
        crate::xlog_info!("video trace: BinkClose({bink:#x}); frame {} presented", crate::hle::d3d12x::present::counters().1);
    }
    let f: unsafe extern "system" fn(usize) = std::mem::transmute(REAL[CLOSE].load(Ordering::Acquire));
    f(bink)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_buffers_are_read_as_the_title_fills_them() {
        let mut w = vec![0u8; FRAME_BUFFERS_BYTES];
        let put32 = |w: &mut Vec<u8>, o: usize, v: u32| w[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let put64 = |w: &mut Vec<u8>, o: usize, v: u64| w[o..o + 8].copy_from_slice(&v.to_le_bytes());
        for (o, v) in [(0, 2), (4, 1920), (8, 1088), (12, 960), (16, 544), (20, 1)] {
            put32(&mut w, o, v);
        }
        // frame 0: Y, cR, cB; frame 1: Y only (the others not allocated); the current frame in 1
        for (i, (at, pitch)) in [(0x1000_0000u64, 1920u32), (0x1100_0000, 960), (0x1200_0000, 960)].into_iter().enumerate() {
            put64(&mut w, 24 + i * 16, at);
            put32(&mut w, 24 + i * 16 + 8, 1);
            put32(&mut w, 24 + i * 16 + 12, pitch);
        }
        put64(&mut w, 24 + 5 * 16, 0x2000_0000);
        put32(&mut w, 24 + 5 * 16 + 12, 2048);
        let b = parse(&w).unwrap();
        assert_eq!(b.frames.len(), 2);
        assert_eq!(b.frames[0], [Plane { at: 0x1000_0000, pitch: 1920, rows: 1088 }, Plane { at: 0x1100_0000, pitch: 960, rows: 544 }, Plane { at: 0x1200_0000, pitch: 960, rows: 544 }]);
        assert_eq!(b.frames[1], [Plane { at: 0x2000_0000, pitch: 2048, rows: 1088 }]);
        assert_eq!(b.frames[0][0].bytes(), 1920 * 1088);
        assert_eq!(b.current, 1);
        // three frames (what the title's own structure has room for)
        put32(&mut w, 0, 3);
        put64(&mut w, 24 + 10 * 16, 0x3000_0000);
        put32(&mut w, 24 + 10 * 16 + 12, 2048);
        let b = parse(&w).unwrap();
        assert_eq!((b.frames.len(), b.frames[2].clone()), (3, vec![Plane { at: 0x3000_0000, pitch: 2048, rows: 1088 }]));
        // no frames, or more than the DLL has: not a frame buffer set
        put32(&mut w, 0, 0);
        assert!(parse(&w).is_none());
        put32(&mut w, 0, 4);
        assert!(parse(&w).is_none());
        // cut short: not read past its end
        put32(&mut w, 0, 3);
        assert!(parse(&w[..FRAME_BUFFERS_BYTES - 8]).is_none());
    }
}
