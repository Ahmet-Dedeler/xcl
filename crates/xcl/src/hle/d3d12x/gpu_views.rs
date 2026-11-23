//! The GPU's own view of a title's page pool.
//!
//! A title with a page pool (RegisterPagePoolX, device #74) maps the pool's 64 KiB pages twice:
//! for the CPU with XMemMapPhysicalPagesScatter, and for the GPU with the command queue's page
//! mapping call (#31 of the newer layout) – at *another* address range, where it places its
//! resources. A retail game reserves two ranges of the same size for its resource memory: one
//! the CPU reads and writes (pages mapped with the kernel call), one only the GPU sees (pages
//! mapped with the queue call, the same pool page at the same offset), and creates every
//! streamed texture – and the planes its video decoder writes – at an address in the second.
//! Its texture lock turns such an address into the CPU's (image +0x5121690).
//!
//! Windows has one address for a page here (`kernelx::map_pages`), and the GPU range is never
//! mapped for the CPU: a texture placed there read as zeros for good (the game's intro video
//! was flat green – three planes of zeros – while its decoder filled the pages through the
//! CPU's range). So the queue call's mappings are kept: GPU-view page → the pool page shown
//! there, named by the physical page that was at its place in the pool
//! (`kernelx::page_of_view`). `title_textures` reads a texture's memory through it
//! (`memory_at`) and finds the textures over memory the CPU wrote (`views_of`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::hle::kernelx::PHYS_PAGE_SIZE as PAGE;

#[derive(Default)]
struct Views {
    /// GPU-view page address → the physical page (its number) mapped there
    at: BTreeMap<usize, usize>,
    /// physical page → the GPU-view page addresses showing it
    of: HashMap<usize, Vec<usize>>,
}

impl Views {
    fn unmap(&mut self, view: usize) {
        if let Some(page) = self.at.remove(&view) {
            if let Some(v) = self.of.get_mut(&page) {
                v.retain(|&a| a != view);
                if v.is_empty() {
                    self.of.remove(&page);
                }
            }
        }
    }

    fn map(&mut self, view: usize, page: usize) {
        self.unmap(view);
        self.at.insert(view, page);
        self.of.entry(page).or_default().push(view);
    }
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);
/// pages mapped in GPU views now (0: no title here maps any – nothing is looked up)
static MAPPED: AtomicUsize = AtomicUsize::new(0);
/// lowest and one past the highest GPU-view address ever mapped (a quick "not there")
static LOW: AtomicUsize = AtomicUsize::new(usize::MAX);
static HIGH: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static UNKNOWN: AtomicU64 = AtomicU64::new(0);

/// `XCL_TRACE_PAGE_VIEWS=1`: what the queue's page mapping calls ask for is counted – how far
/// each mapped GPU-view page is from where the CPU has its page (one distance for all of them:
/// the GPU's range mirrors the CPU's), pages mapped that the CPU has nowhere, views mapped
/// again to another page – and logged with the calls' lines (`summary`).
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_PAGE_VIEWS").is_some())
}

#[derive(Default)]
struct Stats {
    /// GPU-view address minus the CPU's address of the page, at the time of the call → pages
    distances: HashMap<isize, u64>,
    /// pages mapped whose physical page has no memory anywhere
    nowhere: u64,
    /// views that showed another page when they were mapped / the same page again
    replaced: u64,
    again: u64,
    /// views unmapped that showed a page / that showed none
    unmapped: u64,
    unmapped_empty: u64,
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

/// `XCL_TRACE_PAGE_VIEWS`: the counts so far, as text.
pub fn summary() -> String {
    let g = STATS.lock().unwrap();
    let Some(s) = g.as_ref() else { return String::from("page views are not traced (XCL_TRACE_PAGE_VIEWS)") };
    let mut d: Vec<(isize, u64)> = s.distances.iter().map(|(&k, &v)| (k, v)).collect();
    d.sort_by_key(|e| std::cmp::Reverse(e.1));
    let shown: Vec<String> = d.iter().take(4).map(|&(k, v)| format!("{v} page(s) at {k:#x}")).collect();
    format!(
        "{} distance(s) between a GPU-view page and the CPU's address of its page: {}; {} mapped with no memory anywhere; {} view(s) mapped again to another page, {} to the same one; {} unmapped, {} unmapped that showed nothing; {} mapped now",
        d.len(),
        shown.join(", "),
        s.nowhere,
        s.replaced,
        s.again,
        s.unmapped,
        s.unmapped_empty,
        MAPPED.load(Ordering::Relaxed)
    )
}

/// One range of a batch: `count` pages from the batch's running address are mapped to the pool
/// pages from `pool_page` (flags 1) or unmapped (flags 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub flags: u32,
    pub pool_page: u32,
    pub count: u32,
}

/// The mappings a batch at GPU-view address `address` asks for: (GPU-view page address, the
/// pool page shown there or None for "unmapped"), in order. Ranges follow one another from
/// the batch's address; a range with other flags only moves on.
pub fn batch_mappings(address: usize, ranges: &[Range]) -> Vec<(usize, Option<u32>)> {
    let mut out = Vec::new();
    let mut at = address & !(PAGE - 1);
    for r in ranges {
        for i in 0..r.count as usize {
            match r.flags {
                1 => out.push((at + i * PAGE, Some(r.pool_page + i as u32))),
                2 => out.push((at + i * PAGE, None)),
                _ => {}
            }
        }
        at += r.count as usize * PAGE;
    }
    out
}

/// most ranges of one batch / batches of one call that are read (a call that asks for more is
/// not one of these)
const MAX_RANGES: usize = 1 << 20;
const MAX_BATCHES: usize = 1 << 16;

/// Command queue #31 of the newer layout (the name is XCL's): update the GPU's page mappings
/// of a page pool – `(batch count, batches, page pool address, pages in the pool, 0)`.
///
/// * `n`, `batches`: `n` records of 0x18 bytes – { GPU-view address (u64), range count (u32),
///   pointer to the ranges (at +0x10) }.
/// * a range is 12 bytes – { flags (u32): 1 = map, 2 = unmap; pool page (u32): the index of a
///   64 KiB page in the pool, for a map; page count (u32) }. A batch's ranges follow one
///   another from its address.
/// * `pool`: the address the pool was registered with (RegisterPagePoolX, device #74), where
///   the title first mapped all of its pages in order: pool page `k` is the physical page that
///   was mapped at `pool + k * 64 KiB`.
/// * `arg`: the pool's size in pages (the high part of its byte size); not used here.
///
/// From a retail game's caller (image +0xf2c560; it builds one map range a page and one unmap
/// range a run) and its calls in a run: pool 0xd20000000, 0x185c0 pages (the 99,776 pages it
/// allocates), batches at 0x7440000000 and up – its GPU range starts at 0x7000000000, its CPU
/// range at 0x720000000, and a batch address minus the GPU range's start is the offset
/// XMemMapPhysicalPagesScatter got in the CPU range in the call before. The result is not
/// looked at.
pub(super) unsafe fn update(n: u32, batches: *const u8, pool: usize, arg: u32) {
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    // `XCL_GPU_PAGE_MAPPINGS=0`: the calls are not kept – a GPU view is read by the pairing of
    // the two reservations alone (`kernelx::mirrors`), to compare the two
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("XCL_GPU_PAGE_MAPPINGS").is_ok_and(|v| v.trim() == "0")) {
        if call == 0 {
            crate::xlog_info!("D3D12.X: command queue #31 (page mappings of pool {pool:#x}): not kept (XCL_GPU_PAGE_MAPPINGS=0)");
        }
        return;
    }
    let readable = crate::gpu::shader::readable;
    if n == 0 || n as usize > MAX_BATCHES || pool == 0 || !pool.is_multiple_of(PAGE) || !readable(batches as usize, n as usize * 0x18) {
        if UNKNOWN.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: command queue #31 ({n} batch(es) at {batches:p}, pool {pool:#x}, {arg:#x}): not page mappings as XCL knows them – not done");
        }
        return;
    }
    // what is asked, then the pool pages' physical pages (the kernel's lock), then the table
    let mut asked: Vec<(usize, Option<u32>)> = Vec::new();
    for k in 0..n as usize {
        let b = batches.add(k * 0x18);
        let (address, count, ranges) = ((b as *const u64).read_unaligned() as usize, (b.add(8) as *const u32).read_unaligned() as usize, (b.add(0x10) as *const usize).read_unaligned());
        if count == 0 {
            continue;
        }
        if count > MAX_RANGES || !readable(ranges, count * 12) {
            if UNKNOWN.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: command queue #31: batch {k} at {address:#x} has {count} range(s) at {ranges:#x} that cannot be read – left out");
            }
            continue;
        }
        let ranges: Vec<Range> = (0..count)
            .map(|i| {
                let [flags, pool_page, count] = ((ranges + i * 12) as *const [u32; 3]).read_unaligned();
                Range { flags, pool_page, count }
            })
            .collect();
        asked.extend(batch_mappings(address, &ranges));
    }
    let resolved: Vec<(usize, Option<usize>)> = asked
        .iter()
        .map(|&(view, pool_page)| (view, pool_page.and_then(|p| crate::hle::kernelx::page_of_view(pool + p as usize * PAGE))))
        .collect();
    // (the trace's distances: asked of the kernel before the table is locked)
    let places: Vec<Option<usize>> = if trace() { resolved.iter().map(|&(_, page)| page.and_then(crate::hle::kernelx::page_address)).collect() } else { Vec::new() };
    let (mut mapped, mut unmapped, mut lost) = (0usize, 0usize, 0usize);
    {
        let mut g = VIEWS.lock().unwrap();
        let v = g.get_or_insert_with(Views::default);
        if trace() {
            let mut st = STATS.lock().unwrap();
            let st = st.get_or_insert_with(Stats::default);
            for (&(view, page), place) in resolved.iter().zip(&places) {
                let shown = v.at.get(&view).copied();
                match (page, place) {
                    (Some(page), place) => {
                        match shown {
                            Some(old) if old == page => st.again += 1,
                            Some(_) => st.replaced += 1,
                            None => {}
                        }
                        match place {
                            Some(at) => *st.distances.entry(view as isize - *at as isize).or_default() += 1,
                            None => st.nowhere += 1,
                        }
                    }
                    (None, _) if shown.is_some() => st.unmapped += 1,
                    (None, _) => st.unmapped_empty += 1,
                }
            }
        }
        for (&(view, page), &(_, pool_page)) in resolved.iter().zip(&asked) {
            match page {
                Some(page) => {
                    v.map(view, page);
                    mapped += 1;
                    LOW.fetch_min(view, Ordering::Relaxed);
                    HIGH.fetch_max(view + PAGE, Ordering::Relaxed);
                }
                None => {
                    v.unmap(view);
                    if pool_page.is_some() {
                        lost += 1;
                    } else {
                        unmapped += 1;
                    }
                }
            }
        }
        MAPPED.store(v.at.len(), Ordering::Relaxed);
    }
    if call < 4 || call.is_power_of_two() || lost != 0 && UNKNOWN.fetch_add(1, Ordering::Relaxed) < 8 {
        let first = asked.first().map(|&(view, p)| format!("; first: {view:#x} → pool page {p:x?}")).unwrap_or_default();
        crate::xlog_info!(
            "D3D12.X: command queue #31 (page mappings of pool {pool:#x}, call #{call}, {n} batch(es), {arg:#x}): {mapped} page(s) mapped in the GPU's view, {unmapped} unmapped{}; {} mapped there now{first}",
            if lost != 0 { format!(", {lost} name a pool page the title has no memory at") } else { String::new() },
            MAPPED.load(Ordering::Relaxed)
        );
        if trace() {
            crate::xlog_info!("D3D12.X: page views after call #{call}: {}", summary());
        }
    }
}

/// Whether any page is mapped in a GPU view (else nothing here applies).
pub fn any() -> bool {
    MAPPED.load(Ordering::Relaxed) != 0
}

/// Whether `[addr, addr + len)` touches the addresses GPU views were mapped at so far (from the
/// lowest to the highest one): memory the CPU never writes at these addresses.
pub fn in_range(addr: usize, len: usize) -> bool {
    any() && addr < HIGH.load(Ordering::Relaxed) && addr + len > LOW.load(Ordering::Relaxed)
}

/// The physical page shown at the GPU-view page of `addr`.
fn page_at(addr: usize) -> Option<usize> {
    if !any() || addr < LOW.load(Ordering::Relaxed) || addr >= HIGH.load(Ordering::Relaxed) {
        return None;
    }
    VIEWS.lock().unwrap().as_ref()?.at.get(&(addr & !(PAGE - 1))).copied()
}

/// What the queue's page mappings say of `addr`: None – no page is mapped at its GPU-view page
/// (or it is no GPU-view address); Some(place) – one is, and `place` is where the CPU has the
/// memory the GPU sees at `addr` (None: a page with no memory yet).
pub fn lookup(addr: usize) -> Option<Option<usize>> {
    let page = page_at(addr)?;
    Some(crate::hle::kernelx::page_address(page).map(|a| a + (addr & (PAGE - 1))))
}

/// Where the CPU has the memory the GPU sees at `addr` (an address in a GPU view): the place
/// of the page mapped there, at `addr`'s offset in it. None: not a GPU-view address, nothing
/// mapped there, or a page with no memory yet.
pub fn memory_at(addr: usize) -> Option<usize> {
    lookup(addr).flatten()
}

/// The GPU-view ranges (address, bytes) that show the CPU memory `[addr, addr + len)` – pages
/// the title mapped there and, with the queue call, in a GPU view. In address order, neighbours
/// joined.
pub fn views_of(addr: usize, len: usize) -> Vec<(usize, usize)> {
    if !any() || len == 0 {
        return Vec::new();
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut p = addr & !(PAGE - 1);
    while p < addr + len {
        // (the kernel's lock, then the table's: never one inside the other)
        if let Some(page) = crate::hle::kernelx::page_at(p) {
            let views: Vec<usize> = VIEWS.lock().unwrap().as_ref().and_then(|v| v.of.get(&page).cloned()).unwrap_or_default();
            let (from, to) = (addr.max(p) - p, (addr + len).min(p + PAGE) - p);
            out.extend(views.into_iter().map(|v| (v + from, to - from)));
        }
        p += PAGE;
    }
    out.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(out.len());
    for (a, n) in out {
        match joined.last_mut() {
            Some(l) if l.0 + l.1 == a => l.1 += n,
            _ => joined.push((a, n)),
        }
    }
    joined
}

/// (pages mapped in GPU views now, calls of the queue's page mapping method)
pub fn counts() -> (usize, u64) {
    (MAPPED.load(Ordering::Relaxed), CALLS.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_follow_one_another_from_the_batch_address() {
        let a = 0x7440_0000_0000usize;
        // two single pages (as the title maps: one range a page), then three unmapped, then a
        // range of flags XCL does not know (skipped, but it takes its pages), then one mapped
        let ranges = [
            Range { flags: 1, pool_page: 7, count: 1 },
            Range { flags: 1, pool_page: 3, count: 1 },
            Range { flags: 2, pool_page: 0, count: 3 },
            Range { flags: 9, pool_page: 1, count: 2 },
            Range { flags: 1, pool_page: 40, count: 2 },
        ];
        assert_eq!(
            batch_mappings(a + 0x123, &ranges),
            [
                (a, Some(7)),
                (a + PAGE, Some(3)),
                (a + 2 * PAGE, None),
                (a + 3 * PAGE, None),
                (a + 4 * PAGE, None),
                (a + 7 * PAGE, Some(40)),
                (a + 8 * PAGE, Some(41)),
            ]
        );
        assert!(batch_mappings(a, &[]).is_empty());
    }

    #[test]
    fn a_view_shows_one_page_and_a_page_knows_its_views() {
        let mut v = Views::default();
        v.map(0x1_0000, 16);
        v.map(0x2_0000, 16);
        v.map(0x3_0000, 32);
        assert_eq!(v.of[&16], [0x1_0000, 0x2_0000]);
        // another page at a view: the first one no longer shows there
        v.map(0x1_0000, 32);
        assert_eq!((v.of[&16].clone(), v.of[&32].clone()), (vec![0x2_0000], vec![0x3_0000, 0x1_0000]));
        v.unmap(0x2_0000);
        assert!(!v.of.contains_key(&16) && v.at.len() == 2);
        v.unmap(0x2_0000);
        assert_eq!(v.at.len(), 2);
    }

    /// The whole path on real pages: a pool mapped at its address, a page of it mapped where
    /// the CPU uses it, the queue call naming it by its pool index for a GPU-view address –
    /// the GPU view reads what the CPU wrote, and the CPU range is found from the write.
    #[test]
    fn a_gpu_view_reads_the_page_the_cpu_wrote() {
        use crate::hle::kernelx::{AllocateTitlePhysicalPages, FreeTitlePhysicalPages, MapTitlePhysicalPages};
        use std::ffi::c_void;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            if crate::gpu::uma::get().is_none() {
                return;
            }
            let mut n = 4usize;
            let mut pages = [0usize; 4];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let reserve = |pages: usize| VirtualAlloc(std::ptr::null(), pages * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            let (pool, cpu, gpu) = (reserve(4), reserve(8), reserve(8));
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 4, 0, 4, pages.as_mut_ptr()) as usize, pool);
            // pool page 2 at virtual page 5 of both ranges
            let mut one = [pages[2]];
            assert_eq!(MapTitlePhysicalPages((cpu + 5 * PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, cpu + 5 * PAGE);
            let ranges = [[1u32, 2, 1]];
            let batch: [usize; 3] = [gpu + 5 * PAGE, 1, ranges.as_ptr() as usize];
            update(1, batch.as_ptr() as *const u8, pool, 4);
            // (a texture over virtual pages 4..7 of the GPU's range, as the content comparison sees it)
            let hash = || crate::hle::d3d12x::title_textures::memory_hash_for_tests(gpu + 4 * PAGE + 0x30, 3 * PAGE - 0x50);
            let before = hash();
            assert_eq!(before, hash());
            *((cpu + 5 * PAGE + 0x40) as *mut u32) = 0xB14B_0002;
            let written = hash();
            assert_ne!(before, written, "a write through the CPU's range shows in the comparison of the GPU's");
            let at = memory_at(gpu + 5 * PAGE + 0x40).expect("the GPU view has the page");
            assert_eq!((at, *(at as *const u32)), (cpu + 5 * PAGE + 0x40, 0xB14B_0002));
            assert_eq!(memory_at(gpu + 4 * PAGE), None, "nothing mapped at the page before");
            assert_eq!(views_of(cpu + 5 * PAGE + 0x100, 0x200), [(gpu + 5 * PAGE + 0x100, 0x200)]);
            assert_eq!(views_of(cpu + 4 * PAGE, 3 * PAGE), [(gpu + 5 * PAGE, PAGE)]);
            // unmapped in the GPU's view: nothing there any more
            let ranges = [[2u32, 0, 1]];
            let batch: [usize; 3] = [gpu + 5 * PAGE, 1, ranges.as_ptr() as usize];
            update(1, batch.as_ptr() as *const u8, pool, 4);
            assert_eq!(memory_at(gpu + 5 * PAGE + 0x40), None);
            assert!(views_of(cpu + 5 * PAGE, PAGE).is_empty());
            assert_ne!(written, hash(), "a page unmapped from the view shows too");
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 4, pages.as_mut_ptr()), 1);
            for r in [pool, cpu, gpu] {
                VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }
}
