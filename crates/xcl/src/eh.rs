//! Exception-handling support for manually mapped images.
//!
//! * `.pdata` is registered with `RtlAddFunctionTable` so the unwinder finds our frames.
//! * `RtlPcToFileHeader` is hooked: x64 C++ `throw` (`_CxxThrowException`) and the
//!   catch matching code use it to find the image base of the thrower's `ThrowInfo`;
//!   the real function only knows modules on the Windows loader list.
//! * An unhandled-exception filter logs crashes with module+RVA and writes the report.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::System::Diagnostics::Debug::{
    RtlAddFunctionTable, SetUnhandledExceptionFilter, EXCEPTION_POINTERS, IMAGE_RUNTIME_FUNCTION_ENTRY,
};

use crate::hooks;
use crate::isa;
use crate::modules;

pub fn register_function_table(base: usize, rva: u32, size: u32) -> bool {
    let count = size / std::mem::size_of::<IMAGE_RUNTIME_FUNCTION_ENTRY>() as u32;
    let ok = unsafe {
        RtlAddFunctionTable((base + rva as usize) as *const IMAGE_RUNTIME_FUNCTION_ENTRY, count, base as u64)
    };
    if !ok {
        crate::xlog_warn!("RtlAddFunctionTable({base:#x}) failed");
    }
    ok
}

static ORIG_PC_TO_FILE_HEADER: AtomicUsize = AtomicUsize::new(0);
type PcToFileHeader = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> *mut c_void;

unsafe extern "system" fn hk_rtl_pc_to_file_header(pc: *mut c_void, base_out: *mut *mut c_void) -> *mut c_void {
    if let Some(b) = modules::image_base_for_addr(pc as usize) {
        if !base_out.is_null() {
            *base_out = b as *mut c_void;
        }
        return b as *mut c_void;
    }
    let orig: PcToFileHeader = std::mem::transmute(ORIG_PC_TO_FILE_HEADER.load(Ordering::Acquire));
    orig(pc, base_out)
}

pub fn install_pc_hook() -> Result<(), String> {
    unsafe {
        hooks::hook(
            "ntdll.dll",
            "RtlPcToFileHeader",
            hk_rtl_pc_to_file_header as usize,
            &ORIG_PC_TO_FILE_HEADER,
        )
    }
}

/// The title's unhandled-exception filter (0: none), called after XCL's report.
static TITLE_FILTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The title sets its filter: kept behind XCL's; returns the title's previous one.
pub fn set_title_filter(f: usize) -> usize {
    TITLE_FILTER.swap(f, Ordering::AcqRel)
}

unsafe extern "system" fn unhandled_filter(info: *const EXCEPTION_POINTERS) -> i32 {
    crate::hle::d3d12x::real::log_device_removed();
    if !info.is_null() && !(*info).ExceptionRecord.is_null() {
        let rec = &*(*info).ExceptionRecord;
        let addr = rec.ExceptionAddress as usize;
        let mut extra = String::new();
        if rec.ExceptionCode as u32 == 0xC000_0005 && rec.NumberParameters >= 2 {
            let op = match rec.ExceptionInformation[0] {
                0 => "read",
                1 => "write",
                8 => "execute",
                _ => "?",
            };
            extra = format!(" ({op} at address {:#x})", rec.ExceptionInformation[1]);
        }
        // an instruction this host does not run and XCL does not emulate (`unsupported_insn_handler`): its
        // bytes name it without a second run
        if rec.ExceptionCode as u32 == 0xC000_001D && crate::gpu::shader::readable(addr, 1) {
            let mut buf = [0u8; 16];
            let hex: Vec<String> = insn_bytes(addr, &mut buf).iter().map(|b| format!("{b:02x}")).collect();
            extra = format!(" (code bytes: {})", hex.join(" "));
        }
        let ctx = (*info).ContextRecord;
        let regs = if ctx.is_null() {
            String::new()
        } else {
            let c = &*ctx;
            // registers that point at objects: the module of their vtable (which object it is)
            let obj = |v: u64| -> String {
                let p = v as usize;
                if p < 0x10000 || !crate::gpu::shader::readable(p, 8) {
                    return String::new();
                }
                let vt = (p as *const usize).read_unaligned();
                let d = modules::describe_address(vt);
                if d.contains('+') { format!(" (vtable {d})") } else { String::new() }
            };
            let regs = [("rax", c.Rax), ("rbx", c.Rbx), ("rcx", c.Rcx), ("rdx", c.Rdx), ("rsi", c.Rsi), ("rdi", c.Rdi), ("rbp", c.Rbp), ("r8", c.R8), ("r9", c.R9), ("r12", c.R12), ("r13", c.R13), ("r14", c.R14), ("r15", c.R15), ("rsp", c.Rsp)];
            let callers = stack_callers(c.Rsp as usize);
            regs.iter().map(|(n, v)| format!(" {n}={v:#x}{}", obj(*v))).collect::<String>() + &if callers.is_empty() { String::new() } else { format!("; on the stack: {callers}") }
        };
        {
            use std::sync::atomic::Ordering::Relaxed;
            let v = &crate::hle::d3d12x::list_x::LAST_INDEX_VIEW;
            if v[0].load(Relaxed) != 0 {
                extra += &format!(" (last index buffer view: {:#x}, {} bytes; the title's {:#x})", v[0].load(Relaxed), v[1].load(Relaxed), v[2].load(Relaxed));
            }
        }
        let msg = format!(
            "unhandled exception {:#010x} in {}{extra}{regs}",
            rec.ExceptionCode as u32,
            modules::describe_address(addr)
        );
        crate::xlog_error!("{msg}");
        crate::report::write_final(Some(msg));
        crate::log::flush();
    }
    match TITLE_FILTER.load(Ordering::Acquire) {
        0 => 0, // EXCEPTION_CONTINUE_SEARCH
        f => std::mem::transmute::<usize, unsafe extern "system" fn(*const EXCEPTION_POINTERS) -> i32>(f)(info),
    }
}

static ORIG_NT_TERMINATE: AtomicUsize = AtomicUsize::new(0);
static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// ntdll!NtTerminateProcess: the process ending (ExitProcess, abort()'s _exit(3),
/// TerminateProcess of itself) is logged once – status, the calls that led there and, when the
/// status is not 0, every other thread – and the log written out. abort() leaves no other
/// trace: the buffered log lines went with the process.
unsafe extern "system" fn hk_nt_terminate_process(h: isize, status: i32) -> i32 {
    if (h == 0 || h == -1) && !ENDING.swap(true, Ordering::AcqRel) {
        let probe = 0usize;
        let (tid, stack) = (windows_sys::Win32::System::Threading::GetCurrentThreadId(), stack_callers_n(&probe as *const usize as usize, 16));
        if status != 0 {
            crate::xlog_error!("process ends with status {status:#x} (thread {tid}); on the stack: {stack}");
            dump_threads();
        } else {
            crate::xlog_info!("process ends (thread {tid}); on the stack: {stack}");
        }
        crate::log::flush();
    }
    let orig: unsafe extern "system" fn(isize, i32) -> i32 = std::mem::transmute(ORIG_NT_TERMINATE.load(Ordering::Acquire));
    orig(h, status)
}

pub fn install_crash_handler() {
    unsafe {
        if let Err(e) = hooks::hook("ntdll.dll", "NtTerminateProcess", hk_nt_terminate_process as usize, &ORIG_NT_TERMINATE) {
            crate::xlog_warn!("{e}");
        }
        SetUnhandledExceptionFilter(Some(unhandled_filter));
        install_invalid_parameter_handler();
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(0, Some(cpp_exception_logger));
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(null_call_logger));
        if std::env::var_os("XCL_ISA_REDIRECT").is_some_and(|v| v == "0") {
            REDIRECT.store(false, Ordering::Relaxed);
        }
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(unsupported_insn_handler));
        // `XCL_DUMP_THREADS_AT=40,90`: every thread's call stack that many seconds into the
        // run – what a title that neither crashes nor moves on is waiting for
        if let Ok(at) = std::env::var("XCL_DUMP_THREADS_AT") {
            let mut times: Vec<u64> = at.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            times.sort_unstable();
            let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
                let start = std::time::Instant::now();
                for t in times {
                    let due = std::time::Duration::from_secs(t);
                    if let Some(wait) = due.checked_sub(start.elapsed()) {
                        std::thread::sleep(wait);
                    }
                    crate::xlog_info!("thread stacks {t} s into the run (XCL_DUMP_THREADS_AT):");
                    dump_threads();
                    // `XCL_PEEK=140a210d81:1,14a210d60:8`: title globals by their address in the
                    // image as linked (hex) and size in bytes, read at the same moment
                    if let (Ok(list), Some(m)) = (std::env::var("XCL_PEEK"), modules::main_module()) {
                        for item in list.split(',') {
                            let mut it = item.trim().split(':');
                            let (Some(a), n) = (it.next().and_then(|a| u64::from_str_radix(a, 16).ok()), it.next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(8)) else { continue };
                            let at = m.base + (a as usize).wrapping_sub(0x1_4000_0000);
                            if !m.contains(at) || !m.contains(at + n.min(8) - 1) {
                                continue;
                            }
                            let mut v = 0u64;
                            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, &mut v as *mut u64 as *mut u8, n.min(8)) };
                            crate::xlog_info!("peek {a:#x} ({n} B) = {v:#x}");
                        }
                    }
                    crate::log::flush();
                }
            });
        }
        if std::env::var_os("XCL_TRACE_EXCEPTIONS").is_some() {
            windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(first_chance_logger));
        }
    }
}

/// `XCL_TRACE_EXCEPTIONS=1`: the first 24 exceptions raised anywhere, as they are raised (code,
/// address, thread, parameters, the return addresses on the stack) – for a thread that ends
/// in nested exceptions, where the unhandled-exception report never comes.
unsafe extern "system" fn first_chance_logger(info: *mut EXCEPTION_POINTERS) -> i32 {
    static N: AtomicUsize = AtomicUsize::new(0);
    let rec = &*(*info).ExceptionRecord;
    let ctx = &*(*info).ContextRecord;
    let code = rec.ExceptionCode as u32;
    // thread names, debug output, XCL's own write-watch arming, single steps
    // … and illegal instructions: the ones XCL emulates would use up the budget, the others
    // end in the unhandled-exception report
    if matches!(code, 0x406D_1388 | 0x4001_0006 | 0x4001_000A | 0xE0C1_A7E1 | 0x8000_0004 | 0xC000_001D) {
        return 0;
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 24 {
        let params: Vec<String> = (0..(rec.NumberParameters as usize).min(4)).map(|k| format!("{:#x}", rec.ExceptionInformation[k])).collect();
        crate::xlog_warn!(
            "exception #{n} {code:#010x} at {} on thread {}, parameters [{}], rsp {:#x}; on the stack: {}",
            modules::describe_address(rec.ExceptionAddress as usize),
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            params.join(", "),
            ctx.Rsp,
            stack_callers(ctx.Rsp as usize)
        );
        crate::log::flush();
    }
    0
}

/// Length of the ModRM operand bytes (ModRM, SIB, displacement) of a memory operand.
unsafe fn modrm_len(p: *const u8) -> Option<usize> {
    let m = *p;
    let (md, rm) = (m >> 6, m & 7);
    if md == 3 {
        return None;
    }
    let mut n = 1;
    if rm == 4 {
        n += 1;
        if md == 0 && *p.add(1) & 7 == 5 {
            n += 4;
        }
    } else if md == 0 && rm == 5 {
        n += 4;
    }
    n += match md {
        1 => 1,
        2 => 4,
        _ => 0,
    };
    Some(n)
}

/// `CONTEXT` as the register file of `isa::exec`.
struct Regs(*mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT);

impl Regs {
    /// Rax..R15 lie in the record in the hardware's register order.
    unsafe fn gpr_at(&self, i: u8) -> *mut u64 {
        std::ptr::addr_of_mut!((*self.0).Rax).add(i as usize & 15)
    }
}

impl isa::Cpu for Regs {
    fn gpr(&self, i: u8) -> u64 {
        unsafe { *self.gpr_at(i) }
    }
    fn set_gpr(&mut self, i: u8, v: u64) {
        unsafe { *self.gpr_at(i) = v }
    }
    fn rip(&self) -> u64 {
        unsafe { (*self.0).Rip }
    }
    fn set_rip(&mut self, v: u64) {
        unsafe { (*self.0).Rip = v }
    }
    fn flags(&self) -> u32 {
        unsafe { (*self.0).EFlags }
    }
    fn set_flags(&mut self, v: u32) {
        unsafe { (*self.0).EFlags = v }
    }
    fn xmm(&self, i: u8) -> (u64, u64) {
        let x = unsafe { (*self.0).Anonymous.FltSave.XmmRegisters[i as usize & 15] };
        (x.Low, x.High as u64)
    }
    fn set_xmm(&mut self, i: u8, v: (u64, u64)) {
        let x = unsafe { &mut (*self.0).Anonymous.FltSave.XmmRegisters[i as usize & 15] };
        (x.Low, x.High) = (v.0, v.1 as i64);
    }
    fn random(&mut self) -> Option<u64> {
        #[link(name = "advapi32")]
        extern "system" {
            /// RtlGenRandom
            fn SystemFunction036(buffer: *mut c_void, len: u32) -> u8;
        }
        let mut v = 0u64;
        (unsafe { SystemFunction036(&mut v as *mut u64 as *mut c_void, 8) } != 0).then_some(v)
    }
}

/// Up to 16 code bytes at `ip`, not reading into a following page that is not mapped.
unsafe fn insn_bytes(ip: usize, buf: &mut [u8; 16]) -> &[u8] {
    let in_page = 0x1000 - (ip & 0xFFF);
    let n = if in_page >= 16 || crate::gpu::shader::readable(ip + in_page, 1) { 16 } else { in_page };
    std::ptr::copy_nonoverlapping(ip as *const u8, buf.as_mut_ptr(), n);
    &buf[..n]
}

/// Helpers that faulting sites are redirected to (`isa::thunk`): 64 KiB within reach of a
/// `call rel32`, allocated at the first use. The first half holds the ADX helpers, one slot
/// per operand combination, the second half one helper per INSERTQ/CLZERO site.
/// 0: not allocated yet, 1: no room near the code – every use is then emulated.
static HELPERS: AtomicUsize = AtomicUsize::new(0);
static HELPERS_USED: AtomicUsize = AtomicUsize::new(isa::ADX_SLOT * isa::ADX_SLOTS);
const HELPERS_SIZE: usize = 0x1_0000;
/// Serialises site patching; `XCL_ISA_REDIRECT=0` turns it off (every use then traps).
static PATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static REDIRECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Free address space for the helpers: the nearest 64 KiB above `near`, else below.
unsafe fn alloc_near(near: usize) -> usize {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RESERVE, PAGE_EXECUTE_READWRITE};
    const REACH: usize = 0x7000_0000;
    let query = |at: usize| -> Option<(usize, usize, bool)> {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        (VirtualQuery(at as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0)
            .then(|| (mbi.BaseAddress as usize, (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize), mbi.State == MEM_FREE))
    };
    let take = |p: usize| VirtualAlloc(p as *const _, HELPERS_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) as usize;
    let mut at = near & !0xFFFF;
    for _ in 0..4096 {
        let Some((_, end, free)) = query(at) else { break };
        if at - (near & !0xFFFF) > REACH {
            break;
        }
        if free && at + HELPERS_SIZE <= end {
            let p = take(at);
            if p != 0 {
                return p;
            }
        }
        let next = end.saturating_add(0xFFFF) & !0xFFFF;
        if next <= at {
            break;
        }
        at = next;
    }
    let mut at = near & !0xFFFF;
    for _ in 0..4096 {
        if at <= 0x2_0000 || (near & !0xFFFF) - at > REACH {
            break;
        }
        let Some((base, _, free)) = query(at - 1) else { break };
        let p = at - HELPERS_SIZE;
        if free && p >= base {
            let p = take(p);
            if p != 0 {
                return p;
            }
        }
        let next = base & !0xFFFF;
        if next >= at {
            break;
        }
        at = next;
    }
    0
}

/// Writes `call helper` over the site at `site` (whose bytes were `code` when it trapped).
/// `true`: the site is to be run again – redirected now, or changed by another thread.
unsafe fn redirect(site: usize, code: &[u8], t: &isa::Thunk, helper: &[u8; isa::THUNK_MAX]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let n = t.site_len as usize;
    let Some(code) = code.get(..n) else { return false };
    if !REDIRECT.load(Ordering::Relaxed) || !(5..=8).contains(&n) {
        return false;
    }
    while PATCHING.swap(true, Ordering::Acquire) {
        std::hint::spin_loop();
    }
    let done = (|| {
        if std::slice::from_raw_parts(site as *const u8, n) != code {
            return true;
        }
        let mut pool = HELPERS.load(Ordering::Relaxed);
        if pool == 0 {
            pool = alloc_near(site).max(1);
            HELPERS.store(pool, Ordering::Relaxed);
        }
        let used = HELPERS_USED.load(Ordering::Relaxed);
        let at = match t.slot {
            _ if pool == 1 => return false,
            Some(s) => pool + (s as usize % isa::ADX_SLOTS) * isa::ADX_SLOT,
            None if used + isa::THUNK_MAX <= HELPERS_SIZE => pool + used,
            None => return false,
        };
        let Some(call) = isa::call_site(site as u64, t.site_len, at as u64) else { return false };
        // eight bytes in one store where the page allows it: no thread runs half a call
        let span = if site & 0xFFF <= 0xFF8 { 8 } else { n };
        let mut old = 0u32;
        if VirtualProtect(site as *const _, span, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        // fresh memory is zero and no helper starts with a zero byte: an ADX slot is written once
        if *(at as *const u8) == 0 {
            std::ptr::copy_nonoverlapping(helper.as_ptr(), at as *mut u8, (t.len as usize).min(isa::THUNK_MAX));
            FlushInstructionCache(GetCurrentProcess(), at as *const _, isa::THUNK_MAX);
            if t.slot.is_none() {
                HELPERS_USED.store(used + isa::THUNK_MAX, Ordering::Relaxed);
            }
        }
        if span == 8 {
            let mut b = (site as *const [u8; 8]).read_unaligned();
            for (d, s) in b.iter_mut().zip(call).take(n) {
                *d = s;
            }
            (site as *mut u64).write_unaligned(u64::from_le_bytes(b));
        } else {
            std::ptr::copy_nonoverlapping(call.as_ptr(), site as *mut u8, n);
        }
        VirtualProtect(site as *const _, span, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), site as *const _, n);
        true
    })();
    PATCHING.store(false, Ordering::Release);
    done
}

/// One log line per instruction kind, and one when emulation by exception turns out to be
/// frequent: each such trap costs tens of microseconds.
unsafe fn note_insn(kind: usize, what: &str, redirected: bool, ip: usize) {
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    static EMULATED: AtomicUsize = AtomicUsize::new(0);
    if SEEN.fetch_or(1 << kind, Ordering::Relaxed) & (1 << kind) == 0 {
        let how = if redirected { "each site calls a helper after its first use" } else { "emulated at each use" };
        crate::xlog_warn!("CPU: {what} is not available on this host – {how} (first at {})", modules::describe_address(ip));
    }
    if !redirected {
        let n = EMULATED.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 10_000 || n == 1_000_000 {
            crate::xlog_warn!("CPU: {n} instructions emulated by exception so far (now {what} at {}) – this is slow", modules::describe_address(ip));
        }
    }
}

/// Zen 2 instructions a host CPU or translator does not run (Rosetta 2: ADX, RDSEED, SSE4a,
/// CLWB, CLZERO, MONITORX/MWAITX), raised as illegal instructions. A site of five bytes or more is redirected
/// to a helper in host code (ADCX/ADOX, INSERTQ, CLZERO in the zeroing loops), so it traps
/// once; the others are emulated on the context at every use (`isa`). CLWB only hints the
/// cache: the site is patched to NOPs. A lone CLZERO zeroes the 64-byte line at RAX here;
/// MONITORX/MWAITX become a yield.
unsafe extern "system" fn unsupported_insn_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    static SITES: AtomicUsize = AtomicUsize::new(0);
    let rec = &*(*info).ExceptionRecord;
    let ctx = &mut *(*info).ContextRecord;
    if rec.ExceptionCode as u32 != 0xC000_001D {
        return 0;
    }
    let ip = ctx.Rip as *const u8;
    let mut buf = [0u8; 16];
    let code = insn_bytes(ip as usize, &mut buf);
    // already a call to a helper: another thread redirected the site while this one was on
    // its way here with the old instruction. Run it again.
    if let (Some(0xE8), Some(rel)) = (code.first(), code.get(1..5)) {
        let to = (ip as usize + 5).wrapping_add(i32::from_le_bytes([rel[0], rel[1], rel[2], rel[3]]) as usize);
        let pool = HELPERS.load(Ordering::Relaxed);
        if pool > 1 && (pool..pool + HELPERS_SIZE).contains(&to) {
            return -1;
        }
    }
    // likewise a CLWB site that became NOPs: a NOP does not raise this exception
    if code.first() == Some(&0x90) {
        return -1;
    }
    let mut helper = [0u8; isa::THUNK_MAX];
    if let Some(t) = isa::thunk(code, &mut helper) {
        // a site that takes a call is never emulated: with CLZERO the call also covers the
        // next instruction, where an emulated thread would return to
        if redirect(ip as usize, code, &t, &helper) {
            note_insn(t.op as usize, t.op.name(), true, ip as usize);
            return -1;
        }
    }
    if let Some(insn) = isa::decode(code) {
        if isa::exec(&insn, &mut Regs(ctx)) {
            note_insn(8 + insn.op as usize, insn.op.name(), false, ip as usize);
            return -1;
        }
    }
    // MONITORX (0F 01 FA) arms a wait on the line at RAX, MWAITX (0F 01 FB) waits for a write
    // to it, a timeout or any other event: returning at once is within its contract, and the
    // title's one wait loop tests its condition again. The thread yields instead.
    if code.starts_with(&[0x0F, 0x01, 0xFA]) || code.starts_with(&[0x0F, 0x01, 0xFB]) {
        if code.get(2) == Some(&0xFB) {
            windows_sys::Win32::System::Threading::SwitchToThread();
        }
        ctx.Rip += 3;
        note_insn(15, "MONITORX/MWAITX", false, ip as usize);
        return -1;
    }
    // CLZERO: 0F 01 FC
    if *ip == 0x0F && *ip.add(1) == 0x01 && *ip.add(2) == 0xFC {
        std::ptr::write_bytes((ctx.Rax as usize & !63) as *mut u8, 0, 64);
        ctx.Rip += 3;
        if SITES.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!("CPU: CLZERO is not available on this host – emulated at each use (first at {})", modules::describe_address(ip as usize));
        }
        return -1;
    }
    // CLWB m8: 66 [REX] 0F AE /6
    let mut k = 0;
    if *ip != 0x66 {
        return 0;
    }
    k += 1;
    if *ip.add(k) & 0xF0 == 0x40 {
        k += 1;
    }
    if *ip.add(k) != 0x0F || *ip.add(k + 1) != 0xAE || (*ip.add(k + 2) >> 3) & 7 != 6 {
        return 0;
    }
    let Some(m) = modrm_len(ip.add(k + 2)) else { return 0 };
    let len = k + 2 + m;
    let mut old = 0u32;
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};
    if VirtualProtect(ip as *const _, len, PAGE_EXECUTE_READWRITE, &mut old) != 0 {
        std::ptr::write_bytes(ip as *mut u8, 0x90, len);
        VirtualProtect(ip as *const _, len, old, &mut old);
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(windows_sys::Win32::System::Threading::GetCurrentProcess(), ip as *const _, len);
    }
    ctx.Rip += len as u64;
    let n = SITES.fetch_add(1, Ordering::Relaxed);
    if n == 0 {
        crate::xlog_warn!("CPU: CLWB is not available on this host – each site becomes NOPs when first reached (first at {})", modules::describe_address(ip as usize));
    }
    -1
}

/// A retail title's stall watchdog (GTA VI 1.0.2.1, exe+0x2215b91..): a thread walks 64
/// heartbeat slots and, when one is older than its limit, calls a hook (which stores to
/// address 0 and returns) and then stores to address 0 itself – a deliberate crash that sends
/// a report. A host that is slow for a moment (shader compiles, other programs) trips it
/// although nothing hangs. Both stores are recognised by their bytes and the watchdog goes
/// on as if it had found no stale slot (`XCL_TITLE_WATCHDOG=1`: the crash stays).
unsafe fn title_watchdog_resume(ctx: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    let rip = ctx.Rip as usize;
    if rip < 0x1_0000 || *KEEP.get_or_init(|| std::env::var("XCL_TITLE_WATCHDOG").is_ok_and(|v| v.trim() == "1")) || !crate::gpu::shader::readable(rip.wrapping_sub(0x42), 0x4E) {
        return false;
    }
    let at = |a: usize, n: usize| std::slice::from_raw_parts(a as *const u8, n);
    // mov dword ptr [0], imm32
    if at(rip, 7) != [0xC7, 0x04, 0x25, 0, 0, 0, 0] {
        return false;
    }
    // the hook: `mov dword ptr [0], 0x2ef1e94b; ret`
    if at(rip + 7, 5) == [0x4B, 0xE9, 0xF1, 0x2E, 0xC3] {
        ctx.Rip += 11;
        return true;
    }
    // the watchdog: `mov r14, rsi; mov ebx, ebp` 0x29 bytes before, `mov r15, [rip+..]; mov r14, rsi;
    // mov ebp, 0x1e` (the path taken when no slot is stale) 0x42 bytes before
    if at(rip + 7, 5) == [0, 0, 0, 0, 0xCC] && at(rip - 0x29, 5) == [0x49, 0x89, 0xF6, 0x89, 0xEB] && at(rip - 0x42, 3) == [0x4C, 0x8B, 0x3D] && at(rip - 0x3B, 8) == [0x49, 0x89, 0xF6, 0xBD, 0x1E, 0, 0, 0] {
        let n = SEEN.fetch_add(1, Ordering::Relaxed);
        if n < 4 || n.is_power_of_two() {
            crate::xlog_warn!("the title's stall watchdog fired (#{n}: a thread's heartbeat is older than its limit) – its deliberate crash is skipped and the title goes on (XCL_TITLE_WATCHDOG=1 keeps the crash)");
        }
        ctx.Rsi = ctx.R14;
        ctx.Rip -= 0x42;
        return true;
    }
    false
}

/// A call or jump through a null (or near-null) pointer: logged with the return address and
/// the registers, since the faulting address itself says nothing about the caller.
unsafe extern "system" fn null_call_logger(info: *mut EXCEPTION_POINTERS) -> i32 {
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    let rec = &*(*info).ExceptionRecord;
    if rec.ExceptionCode as u32 == 0xC000_0005 && title_watchdog_resume(&mut *(*info).ContextRecord) {
        return -1; // EXCEPTION_CONTINUE_EXECUTION
    }
    let ctx = &*(*info).ContextRecord;
    if rec.ExceptionCode as u32 == 0xC000_0005 && ctx.Rip < 0x1_0000 && SEEN.fetch_add(1, Ordering::Relaxed) < 4 {
        let ret = if crate::gpu::shader::readable(ctx.Rsp as usize, 8) { (ctx.Rsp as *const usize).read_unaligned() } else { 0 };
        crate::xlog_error!(
            "call through a null pointer (rip {:#x}): return address {}, rax={:#x} rcx={:#x} rdx={:#x} r8={:#x} r9={:#x} r10={:#x} r11={:#x} rbx={:#x} rsi={:#x} rdi={:#x}; on the stack: {}",
            ctx.Rip, modules::describe_address(ret), ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.Rbx, ctx.Rsi, ctx.Rdi,
            stack_callers(ctx.Rsp as usize)
        );
        crate::log::flush();
    }
    0
}

/// MSVC's x64 `__security_check_cookie` (vcruntime): `cmp rcx, [__security_cookie]; jne fail;
/// rol rcx, 16; test cx, 0xFFFF; jne restore; ret; restore: ror rcx, 16; fail: jmp
/// __report_gsfailure` – fixed but for the cookie's address (`None`) and the jump. The
/// console's vcruntime has `bnd` prefixes (F2) on the branches and the return.
const GS_CHECKS: [&[Option<u8>]; 2] = [
    &[Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None, Some(0x75), Some(0x10), Some(0x48), Some(0xC1), Some(0xC1), Some(0x10), Some(0x66), Some(0xF7), Some(0xC1), Some(0xFF), Some(0xFF), Some(0x75), Some(0x01), Some(0xC3), Some(0x48), Some(0xC1), Some(0xC9), Some(0x10), Some(0xE9)],
    &[
        Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None, Some(0xF2), Some(0x75), Some(0x12), Some(0x48), Some(0xC1), Some(0xC1), Some(0x10), Some(0x66), Some(0xF7), Some(0xC1), Some(0xFF), Some(0xFF), Some(0xF2), Some(0x75), Some(0x02),
        Some(0xF2), Some(0xC3), Some(0x48), Some(0xC1), Some(0xC9), Some(0x10), Some(0xE9),
    ],
];

/// The offset of `__security_check_cookie`'s jump to `__report_gsfailure` (E9 rel32) in
/// `code` (None: not there).
pub fn find_gs_check(code: &[u8]) -> Option<usize> {
    for p in GS_CHECKS {
        if code.len() < p.len() + 4 {
            continue;
        }
        if let Some(i) = (0..code.len() - p.len() - 4).find(|&i| code[i] == 0x48 && p.iter().enumerate().all(|(k, b)| b.map_or(true, |b| code[i + k] == b))) {
            return Some(i + p.len() - 1);
        }
    }
    None
}

/// Executable memory `[at, at + len)` of a title image: its `__report_gsfailure` (the target of
/// `__security_check_cookie`'s failure jump) is redirected to a report that names the
/// function whose stack cookie did not match (the return address on the stack), then ends
/// the process with the same code (0xC0000409) – a stack-cookie failure is a `__fastfail`,
/// which no exception filter sees, and often means a function of XCL's wrote more into a
/// title's buffer than the title expects.
pub unsafe fn hook_gs_failure(at: usize, len: usize, image: &str) {
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};
    let code = std::slice::from_raw_parts(at as *const u8, len);
    let Some(i) = find_gs_check(code) else { return };
    let jmp = at + i;
    let rel = (jmp as *const u8).add(1).cast::<i32>().read_unaligned();
    let target = (jmp as isize + 5 + rel as isize) as usize;
    // thunk: rdx = [rsp] (the checked function's return address), then the report
    let t = crate::stubs::alloc_code(32);
    let mut c: Vec<u8> = vec![0x48, 0x8B, 0x14, 0x24, 0x48, 0xB8];
    c.extend_from_slice(&(gs_failure as usize).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xE0]);
    std::ptr::copy_nonoverlapping(c.as_ptr(), t, c.len());
    // __report_gsfailure: mov rax, thunk; jmp rax
    let mut j: Vec<u8> = vec![0x48, 0xB8];
    j.extend_from_slice(&(t as usize).to_le_bytes());
    j.extend_from_slice(&[0xFF, 0xE0]);
    let mut old = 0u32;
    if VirtualProtect(target as *const std::ffi::c_void, j.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
        return;
    }
    std::ptr::copy_nonoverlapping(j.as_ptr(), target as *mut u8, j.len());
    VirtualProtect(target as *const std::ffi::c_void, j.len(), old, &mut old);
    crate::xlog_info!("{image}: stack-cookie failures are reported (__report_gsfailure at {target:#x})");
}

unsafe extern "system" fn gs_failure(cookie: usize, ret: usize) {
    let here = 0usize;
    crate::xlog_error!(
        "stack cookie check failed (cookie {cookie:#x}) in the function returning to {} – its stack frame was overwritten; on the stack: {}",
        modules::describe_address(ret),
        stack_callers(&here as *const usize as usize)
    );
    crate::log::flush();
    windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), 0xC000_0409);
}

/// First-chance C++ exceptions (0xE06D7363), the first 16: type, what() and where they were
/// thrown. One nobody catches ends the process through std::terminate → abort → __fastfail
/// (0xC0000409), which leaves no other trace (a static initializer in a title DLL's DllMain).
/// The C string at `p`, at most `max` bytes, read only as far as the memory is readable (a
/// vectored handler faulting on an unmapped byte after a title's string would turn the
/// exception it looks at into a fatal one).
unsafe fn bounded_cstr(p: usize, max: usize) -> String {
    let mut bytes = Vec::new();
    let mut a = p;
    while bytes.len() < max {
        // readability is checked page by page: at the start and at every page crossed
        if (a == p || a & 0xFFF == 0) && !crate::gpu::shader::readable(a, 1) {
            break;
        }
        let b = *(a as *const u8);
        if b == 0 {
            break;
        }
        bytes.push(b);
        a += 1;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

unsafe extern "system" fn cpp_exception_logger(info: *mut EXCEPTION_POINTERS) -> i32 {
    const CPP: i32 = 0xE06D_7363_u32 as i32;
    const THREAD_NAME: i32 = 0x406D_1388;
    if info.is_null() || (*info).ExceptionRecord.is_null() || (*info).ContextRecord.is_null() {
        return 0;
    }
    let rec = &*(*info).ExceptionRecord;
    // the debugger's thread-name exception (THREADNAME_INFO: 0x1000, name, thread id, flags):
    // the name goes to SetThreadDescription, where tools – and XCL's thread dump – see it
    if rec.ExceptionCode == THREAD_NAME && rec.NumberParameters >= 3 && rec.ExceptionInformation[0] == 0x1000 {
        let (name, tid) = (rec.ExceptionInformation[1], rec.ExceptionInformation[2] as u32);
        if name != 0 {
            let n: Vec<u16> = bounded_cstr(name, 256).encode_utf16().chain([0]).collect();
            if n.len() > 1 {
                name_thread(tid, &n);
            }
        }
        return 0;
    }
    if rec.ExceptionCode != CPP || rec.NumberParameters < 4 {
        return 0;
    }
    // thrown by the title's code (its image holds the throw info), not by XCL's libraries
    if !crate::modules::mapped_modules().iter().any(|m| m.base == rec.ExceptionInformation[3]) {
        return 0;
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return 0;
    }
    let (object, throw_info, base) = (rec.ExceptionInformation[1], rec.ExceptionInformation[2], rec.ExceptionInformation[3]);
    let rd32 = |a: usize| if crate::gpu::shader::readable(a, 4) { (a as *const i32).read_unaligned() } else { 0 };
    // ThrowInfo { attributes, unwind, forward compat, catchable type array } (image-relative)
    let mut name = String::from("?");
    if throw_info != 0 && base != 0 {
        let cta = rd32(throw_info + 12);
        if cta > 0 {
            let first = rd32(base + cta as usize + 4);
            if first > 0 {
                let td = rd32(base + first as usize + 4);
                if td > 0 {
                    let p = base + td as usize + 16;
                    if crate::gpu::shader::readable(p, 1) {
                        name = bounded_cstr(p, 256);
                    }
                }
            }
        }
    }
    // std::exception { vfptr, const char* what, … } – for other thrown types the second word is
    // whatever the object holds: read as a string only where it points at readable memory
    let mut what = String::new();
    if object != 0 && crate::gpu::shader::readable(object + 8, 8) {
        let s = (object as *const usize).add(1).read_unaligned();
        if s != 0 {
            what = bounded_cstr(s, 200);
        }
    }
    let ctx = &*(*info).ContextRecord;
    crate::xlog_warn!("C++ exception {name} \"{what}\" thrown; on the stack: {}", stack_callers(ctx.Rsp as usize));
    0
}

/// The UCRT's invalid-parameter handler for the whole process (titles and their DLLs share the
/// host's ucrtbase.dll): without one a CRT function given a bad argument ends the process with
/// `__fastfail` (0xC0000409), which no exception filter sees – the log then just stops. With
/// it the call is logged with the code that made it; functions that can return an error then
/// return EINVAL, the `_noreturn` ones still end the process.
unsafe fn install_invalid_parameter_handler() {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    let h = GetModuleHandleA(b"ucrtbase.dll\0".as_ptr());
    if h.is_null() {
        return;
    }
    let Some(set) = GetProcAddress(h, b"_set_invalid_parameter_handler\0".as_ptr()) else { return };
    type Handler = unsafe extern "C" fn(*const u16, *const u16, *const u16, u32, usize);
    let set: unsafe extern "C" fn(Option<Handler>) -> Option<Handler> = std::mem::transmute(set);
    set(Some(invalid_parameter));
}

unsafe extern "C" fn invalid_parameter(_expr: *const u16, _function: *const u16, _file: *const u16, _line: u32, _reserved: usize) {
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 16 {
        let here = 0usize;
        let callers = stack_callers(&here as *const usize as usize);
        crate::xlog_warn!("CRT: invalid parameter (the CRT would end the process with 0xC0000409); called from: {callers}");
    }
}

/// Return addresses into loaded modules (not the system's D3D12 / ntdll / kernel ones) among
/// the first 512 stack slots from `rsp`: the calls that led to a crash (heuristic: values on
/// the stack that point into code).
unsafe fn stack_callers(rsp: usize) -> String {
    stack_callers_n(rsp, 6)
}

/// `stack_callers` with up to `max` callers.
unsafe fn stack_callers_n(rsp: usize, max: usize) -> String {
    let mut callers = Vec::new();
    for k in 0..512usize {
        let at = rsp + 8 * k;
        if !crate::gpu::shader::readable(at, 8) {
            break;
        }
        let d = modules::describe_address((at as *const usize).read_unaligned());
        if d.contains('+') && !d.starts_with("D3D12") && !d.starts_with("ntdll") && !d.starts_with("KERNEL") && !d.ends_with("+0x0") {
            callers.push(d);
            if callers.len() == max {
                break;
            }
        }
    }
    callers.join(", ")
}

#[link(name = "kernel32")]
extern "system" {
    fn GetThreadDescription(thread: *mut std::ffi::c_void, description: *mut *mut u16) -> i32;
    fn LocalFree(mem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}

/// A thread's description (SetThreadDescription – engines name their threads).
unsafe fn thread_name(h: *mut std::ffi::c_void) -> String {
    let mut p: *mut u16 = std::ptr::null_mut();
    if GetThreadDescription(h, &mut p) < 0 || p.is_null() {
        return String::new();
    }
    let n = (0..256).take_while(|&i| *p.add(i) != 0).count();
    let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
    LocalFree(p as *mut _);
    s
}

#[link(name = "kernel32")]
extern "system" {
    fn SetThreadDescription(thread: *mut std::ffi::c_void, description: *const u16) -> i32;
}

/// Thread `tid` (u32::MAX: the current one) is called `name` (UTF-16, terminated).
unsafe fn name_thread(tid: u32, name: &[u16]) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, OpenThread, THREAD_SET_LIMITED_INFORMATION};
    if tid == u32::MAX || tid == windows_sys::Win32::System::Threading::GetCurrentThreadId() {
        SetThreadDescription(GetCurrentThread(), name.as_ptr());
        return;
    }
    let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
    if !h.is_null() {
        SetThreadDescription(h, name.as_ptr());
        CloseHandle(h);
    }
}

/// The return addresses of the frames from `ctx` on (its RIP first), unwound with the
/// functions' unwind data (registered for the title's images too); up to `n`.
unsafe fn unwind(ctx: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT, n: usize) -> Vec<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER};
    let mut out = Vec::new();
    for _ in 0..n {
        let rip = ctx.Rip;
        if rip == 0 {
            break;
        }
        out.push(rip as usize);
        let mut base = 0u64;
        let f = RtlLookupFunctionEntry(rip, &mut base, std::ptr::null_mut());
        if f.is_null() {
            // a leaf function: the return address is at RSP
            if !crate::gpu::shader::readable(ctx.Rsp as usize, 8) {
                break;
            }
            ctx.Rip = (ctx.Rsp as *const u64).read_unaligned();
            ctx.Rsp += 8;
        } else {
            let mut data = std::ptr::null_mut();
            let mut frame = 0u64;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, rip, f, ctx, &mut data, &mut frame, std::ptr::null_mut());
        }
    }
    out
}

/// `XCL_THREAD_DUMP_AT`: every other thread of the process stopped for a moment – where it is
/// (RIP) and the calls on its stack that lead there, in the log.
pub fn dump_threads() {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};
    unsafe {
        let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let mut lines = Vec::new();
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            if te.th32OwnerProcessID == pid && te.th32ThreadID != me {
                let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, te.th32ThreadID);
                if !h.is_null() {
                    // CONTEXT must be 16-byte aligned
                    #[repr(align(16))]
                    struct Aligned(CONTEXT);
                    let mut c: Aligned = std::mem::zeroed();
                    c.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
                    // only the context is taken while the thread is suspended: the unwinding
                    // (RtlLookupFunctionEntry), the module lookup (loader lock), the name and the
                    // formatting (heap) take locks the suspended thread may hold (#106)
                    let got = SuspendThread(h) != u32::MAX && {
                        let ok = GetThreadContext(h, &mut c.0) != 0;
                        ResumeThread(h);
                        ok
                    };
                    if got {
                        let frames: Vec<String> = unwind(&mut c.0, 14).into_iter().map(modules::describe_address).collect();
                        lines.push(format!("thread {} '{}': {}", te.th32ThreadID, thread_name(h), frames.join(" < ")));
                    }
                    CloseHandle(h);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
        crate::xlog_info!("thread dump ({} threads):
{}", lines.len(), lines.join("
"));
        crate::log::flush();
    }
}

// ---------------------------------------------------------------------------------------
// write watchpoint (diagnostics): who overwrites a given 8 bytes
// ---------------------------------------------------------------------------------------

static WATCHED: AtomicUsize = AtomicUsize::new(0);
static WATCH_HITS: AtomicUsize = AtomicUsize::new(0);

/// the exception `watch_writes` raises: the handler arms the raising thread at once
const WATCH_ARM: u32 = 0xE0C1_A7E1;

unsafe extern "system" fn watch_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    const SINGLE_STEP: u32 = 0x8000_0004;
    let rec = &*(*info).ExceptionRecord;
    let ctx = &mut *(*info).ContextRecord;
    let addr = WATCHED.load(Ordering::Acquire);
    if rec.ExceptionCode as u32 == WATCH_ARM {
        ctx.Dr0 = addr as u64;
        // L0, RW0 = 01 (writes), LEN0 = 10 (8 bytes)
        ctx.Dr7 = (ctx.Dr7 & !0xF_0003) | 1 | (0b01 << 16) | (0b10 << 18);
        ctx.ContextFlags |= windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_DEBUG_REGISTERS_AMD64;
        return -1;
    }
    if rec.ExceptionCode as u32 != SINGLE_STEP || addr == 0 || ctx.Dr6 & 1 == 0 {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    ctx.Dr6 = 0;
    if WATCH_HITS.fetch_add(1, Ordering::Relaxed) < 32 {
        let callers = stack_callers(ctx.Rsp as usize);
        crate::xlog_warn!(
            "watch: {:#x} now {:#x}, written by the instruction before {} (thread {}); on the stack: {}",
            addr,
            (addr as *const u64).read_volatile(),
            modules::describe_address(ctx.Rip as usize),
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            callers
        );
    }
    -1 // EXCEPTION_CONTINUE_EXECUTION
}

/// Arm a hardware write watchpoint on the 8 bytes at `addr` in every thread of the process
/// (threads created later are armed as they appear, for a minute); each write is logged
/// with the writing code's address. `XCL_WATCH_DEVICE=1`: on the title's D3D12 device.
pub fn watch_writes(addr: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, GetThreadContext, SetThreadContext, CONTEXT, CONTEXT_DEBUG_REGISTERS_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    if WATCHED.swap(addr, Ordering::AcqRel) != 0 {
        return;
    }
    unsafe {
        AddVectoredExceptionHandler(1, Some(watch_handler));
        // the calling thread now (the others below, as the arming loop reaches them)
        windows_sys::Win32::System::Diagnostics::Debug::RaiseException(WATCH_ARM, 0, 0, std::ptr::null());
    }
    crate::xlog_warn!("watch: write watchpoint on {addr:#x}");
    std::thread::spawn(move || {
        let me = unsafe { GetCurrentThreadId() };
        let mut armed = std::collections::HashSet::new();
        let start = std::time::Instant::now();
        while start.elapsed().as_secs() < 60 {
            unsafe {
                let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let mut e: THREADENTRY32 = std::mem::zeroed();
                e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
                let mut ok = Thread32First(snap, &mut e) != 0;
                while ok {
                    if e.th32OwnerProcessID == GetCurrentProcessId() && e.th32ThreadID != me && armed.insert(e.th32ThreadID) {
                        let h = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, 0, e.th32ThreadID);
                        if !h.is_null() {
                            // CONTEXT must be 16-byte aligned
                            #[repr(align(16))]
                            struct Aligned(CONTEXT);
                            let mut c: Aligned = std::mem::zeroed();
                            c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                            if SuspendThread(h) != u32::MAX {
                                if GetThreadContext(h, &mut c.0) != 0 {
                                    c.0.Dr0 = addr as u64;
                                    // L0, RW0 = 01 (writes), LEN0 = 10 (8 bytes)
                                    c.0.Dr7 = (c.0.Dr7 & !0xF_0003) | 1 | (0b01 << 16) | (0b10 << 18);
                                    SetThreadContext(h, &c.0);
                                }
                                ResumeThread(h);
                            } else {
                                armed.remove(&e.th32ThreadID);
                            }
                            windows_sys::Win32::Foundation::CloseHandle(h);
                        }
                    }
                    ok = Thread32Next(snap, &mut e) != 0;
                }
                windows_sys::Win32::Foundation::CloseHandle(snap);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    });
}

// ---------------------------------------------------------------------------------------
// call probes (diagnostics): `XCL_PROBE=rva,…` – what the title's code at those addresses
// is called with
// ---------------------------------------------------------------------------------------

/// probed addresses (DR1..DR3; DR0 stays the write watchpoint's)
static PROBES: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static PROBE_HITS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
/// the last logged values of each probe (a hit that repeats them is only counted)
static PROBE_LAST: Mutex<[u64; 3]> = Mutex::new([0; 3]);
static PROBE_LINES: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

unsafe fn probe_bytes(at: u64, n: usize) -> Option<Vec<u8>> {
    (at > 0x10000 && crate::gpu::shader::readable(at as usize, n)).then(|| std::slice::from_raw_parts(at as *const u8, n).to_vec())
}

/// the exception `arm_current_thread` raises: the handler arms the raising thread
const PROBE_ARM: u32 = 0xE0C1_A7E0;

/// The probes in the calling thread at once (XCL's own short-lived threads that run title
/// callbacks end before the arming loop sees them).
pub fn arm_current_thread() {
    if PROBES[0].load(Ordering::Acquire) != 0 {
        unsafe { windows_sys::Win32::System::Diagnostics::Debug::RaiseException(PROBE_ARM, 0, 0, std::ptr::null()) };
    }
}

unsafe fn arm_context(c: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) {
    for k in 0..3 {
        let a = PROBES[k].load(Ordering::Acquire) as u64;
        if a == 0 {
            continue;
        }
        match k {
            0 => c.Dr1 = a,
            1 => c.Dr2 = a,
            _ => c.Dr3 = a,
        }
        // L(k+1) enabled; RW = 00 (execution), LEN = 00
        let i = k as u32 + 1;
        c.Dr7 = (c.Dr7 & !(0xF_u64 << (16 + 4 * i)) & !(3_u64 << (2 * i))) | (1 << (2 * i));
    }
}

unsafe extern "system" fn probe_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    const SINGLE_STEP: u32 = 0x8000_0004;
    let rec = &*(*info).ExceptionRecord;
    let ctx = &mut *(*info).ContextRecord;
    if rec.ExceptionCode as u32 == PROBE_ARM {
        arm_context(ctx);
        ctx.ContextFlags |= windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_DEBUG_REGISTERS_AMD64;
        return -1;
    }
    if rec.ExceptionCode as u32 != SINGLE_STEP {
        return 0;
    }
    let Some(k) = (0..3).find(|&k| ctx.Dr6 & (2 << k) != 0 && PROBES[k].load(Ordering::Relaxed) == ctx.Rip as usize) else { return 0 };
    ctx.Dr6 = 0;
    // resume flag: the instruction runs without breaking again
    ctx.EFlags |= 0x1_0000;
    let n = PROBE_HITS[k].fetch_add(1, Ordering::Relaxed) + 1;
    // at a function's entry: arguments 5 and 6 above the return address and the home area
    let stack = |i: u64| probe_bytes(ctx.Rsp + 8 * i, 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let (s5, s6) = (stack(5), stack(6));
    let at_r8 = probe_bytes(ctx.R8, 24);
    let at_rcx = probe_bytes(ctx.Rcx, 48);
    // values that tell calls apart (pointers left out: they differ per call)
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&(ctx.Rax as u32, ctx.Rdx as u32 & 0xFF, ctx.R9 as u32, s5 as u8, s6 as u8, &at_r8, &at_rcx), &mut h);
    let key = std::hash::Hasher::finish(&h);
    let new = {
        let mut g = PROBE_LAST.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::replace(&mut g[k], key) != key
    };
    if (new || n.is_power_of_two()) && PROBE_LINES[k].fetch_add(1, Ordering::Relaxed) < 200 {
        crate::xlog_info!(
            "probe {}: hit #{n} (thread {}): rcx {:#x} rdx {:#x} r8 {:#x} r9 {:#x} rax {:#x}, stack args 5/6 {s5:#x} {s6:#x}; at r8 {:x?}; at rcx {:x?}; from {}",
            modules::describe_address(ctx.Rip as usize),
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            ctx.Rcx,
            ctx.Rdx,
            ctx.R8,
            ctx.R9,
            ctx.Rax,
            at_r8,
            at_rcx,
            modules::describe_address(stack(0) as usize)
        );
    }
    -1
}

/// `XCL_PROBE=rva,…` (hex, up to three, in the title's exe): hardware breakpoints on those
/// instructions in every thread; each hit with new argument values is logged (registers,
/// stack arguments 5 and 6, the bytes at rcx and r8) and the title goes on – which of the
/// title's code runs, with what, without changing it.
pub fn start_probes() {
    // `XCL_WATCH=rva` (hex, in the title's exe): every write to those 8 bytes is logged with
    // the writing code (`watch_writes`) – who fills a field the title later trips over
    if let Some(rva) = std::env::var("XCL_WATCH").ok().and_then(|v| usize::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) {
        let _ = std::thread::Builder::new().name("xcl-watch".into()).spawn(move || loop {
            if let Some(m) = modules::main_module() {
                watch_writes(m.base + rva);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        });
    }
    let rvas: Vec<usize> = std::env::var("XCL_PROBE").unwrap_or_default().split(',').filter_map(|v| usize::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).take(3).collect();
    if rvas.is_empty() {
        return;
    }
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, GetThreadContext, SetThreadContext, CONTEXT, CONTEXT_DEBUG_REGISTERS_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    unsafe {
        AddVectoredExceptionHandler(1, Some(probe_handler));
    }
    let _ = std::thread::Builder::new().name("xcl-probe".into()).spawn(move || {
        let base = loop {
            if let Some(m) = modules::main_module() {
                break m.base;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        for (k, rva) in rvas.iter().enumerate() {
            PROBES[k].store(base + rva, Ordering::Release);
            crate::xlog_info!("probe: {}", modules::describe_address(base + rva));
        }
        let me = unsafe { GetCurrentThreadId() };
        let mut armed = std::collections::HashSet::new();
        loop {
            unsafe {
                let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let mut e: THREADENTRY32 = std::mem::zeroed();
                e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
                let mut ok = Thread32First(snap, &mut e) != 0;
                while ok {
                    if e.th32OwnerProcessID == GetCurrentProcessId() && e.th32ThreadID != me && armed.insert(e.th32ThreadID) {
                        let h = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, 0, e.th32ThreadID);
                        if !h.is_null() {
                            #[repr(align(16))]
                            struct Aligned(CONTEXT);
                            let mut c: Aligned = std::mem::zeroed();
                            c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                            if SuspendThread(h) != u32::MAX {
                                if GetThreadContext(h, &mut c.0) != 0 {
                                    arm_context(&mut c.0);
                                    SetThreadContext(h, &c.0);
                                }
                                ResumeThread(h);
                            } else {
                                armed.remove(&e.th32ThreadID);
                            }
                            windows_sys::Win32::Foundation::CloseHandle(h);
                        }
                    }
                    ok = Thread32Next(snap, &mut e) != 0;
                }
                windows_sys::Win32::Foundation::CloseHandle(snap);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cstr_stops_at_nul_limit_and_unreadable_memory() {
        unsafe {
            let s = b"hello\0tail";
            assert_eq!(bounded_cstr(s.as_ptr() as usize, 64), "hello");
            assert_eq!(bounded_cstr(s.as_ptr() as usize, 3), "hel");
            assert_eq!(bounded_cstr(0, 64), "");
            // a string running to the end of a page followed by an unmapped one
            use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
            let p = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, 0x1000, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            std::ptr::write_bytes(p as *mut u8, b'x', 0x1000);
            assert_eq!(bounded_cstr(p + 0x1000 - 4, 64), "xxxx");
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn finds_both_cookie_checks() {
        let pc = [0x48, 0x3B, 0x0D, 1, 2, 3, 4, 0x75, 0x10, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0x75, 0x01, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9, 0, 0, 0, 0, 0xCC];
        assert_eq!(find_gs_check(&pc), Some(25));
        let console: Vec<u8> = [0xCC, 0x48, 0x3B, 0x0D, 0x99, 0x7E, 0x54, 0x00, 0xF2, 0x75, 0x12, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0xF2, 0x75, 0x02, 0xF2, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9, 0x4F, 0x06, 0, 0, 0xCC].to_vec();
        assert_eq!(find_gs_check(&console), Some(29));
        assert_eq!(find_gs_check(&[0x48; 64]), None);
    }
}
