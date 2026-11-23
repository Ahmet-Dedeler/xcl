//! Enhanced barriers of console code as the legacy barriers XCL records.
//!
//! Command list #146 is `Barrier(group count, groups)` with the PC's structures (read from a
//! retail game's writers): D3D12_BARRIER_GROUP { type, count, barriers } of 16 bytes, type 1 =
//! D3D12_TEXTURE_BARRIER (0x40 bytes: sync before / after, access before / after, layout
//! before / after, resource +0x18, subresource range +0x20, flags +0x38), type 2 =
//! D3D12_BUFFER_BARRIER (0x28 bytes: sync, access, resource +0x10, offset, size), type 0 =
//! D3D12_GLOBAL_BARRIER. XCL creates every resource in a legacy state (`layout_state`) and
//! makes its own transitions with ResourceBarrier, and the two kinds do not mix on one
//! resource – so a texture barrier becomes the transition between the states of its layouts,
//! and a barrier around unordered access a UAV barrier.

/// D3D12_BARRIER_LAYOUT_UNDEFINED
pub const LAYOUT_UNDEFINED: u32 = 0xFFFF_FFFF;
/// D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
const ACCESS_UNORDERED_ACCESS: u32 = 0x10;
const GROUP_BYTES: usize = 16;
const TEXTURE_BYTES: usize = 0x40;
const BUFFER_BYTES: usize = 0x28;
const GLOBAL_BYTES: usize = 16;
/// barriers taken from one group (a retail game flushes at 0x400)
const MAX_BARRIERS: usize = 0x1_0000;

/// The resource state (D3D12_RESOURCE_STATES) of a D3D12_BARRIER_LAYOUT (undefined: COMMON);
/// the queue-specific layouts are their plain ones.
pub fn layout_state(layout: u32) -> u32 {
    match layout {
        1 | 19 | 25 => 0xAC3, // generic read
        2 => 0x4,             // render target
        3 | 20 | 26 => 0x8,   // unordered access
        4 => 0x10,            // depth write
        5 => 0x20,            // depth read
        6 | 21 | 27 => 0xC0,  // shader resource (all stages)
        7 | 22 | 28 => 0x800, // copy source
        8 | 23 | 29 => 0x400, // copy dest
        9 => 0x2000,          // resolve source
        10 => 0x1000,         // resolve dest
        11 => 0x100_0000,     // shading rate source
        _ => 0,
    }
}

/// What a `Barrier` call held (for the log).
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Counts {
    pub textures: u32,
    pub buffers: u32,
    pub globals: u32,
    /// texture barriers from or to the undefined layout: no legacy transition says that
    pub undefined: u32,
    /// groups of a type the PC does not have, or whose memory is not readable
    pub unknown: u32,
}

/// The subresources of D3D12_BARRIER_SUBRESOURCE_RANGE { index or first mip, mip count, first
/// slice, slice count, first plane, plane count } in a texture of `mips` × `slices` × `planes`
/// (None: all of them).
pub fn subresources(range: [u32; 6], mips: u32, slices: u32, planes: u32) -> Option<Vec<u32>> {
    let [first_mip, n_mips, first_slice, n_slices, first_plane, n_planes] = range;
    if n_mips == 0 {
        // a subresource index
        return (first_mip != u32::MAX).then(|| vec![first_mip]);
    }
    let (mips, slices, planes) = (mips.max(1), slices.max(1), planes.max(1));
    let m = first_mip.min(mips)..first_mip.saturating_add(n_mips).min(mips);
    let s = first_slice.min(slices)..first_slice.saturating_add(n_slices.max(1)).min(slices);
    let p = first_plane.min(planes)..first_plane.saturating_add(n_planes.max(1)).min(planes);
    if m.len() as u32 == mips && s.len() as u32 == slices && p.len() as u32 == planes {
        return None;
    }
    let mut out = Vec::with_capacity(m.len() * s.len() * p.len());
    for plane in p {
        for slice in s.clone() {
            for mip in m.clone() {
                out.push(mip + slice * mips + plane * mips * slices);
            }
        }
    }
    Some(out)
}

unsafe fn u32_at(p: *const u8, o: usize) -> u32 {
    (p.add(o) as *const u32).read_unaligned()
}

unsafe fn u64_at(p: *const u8, o: usize) -> u64 {
    (p.add(o) as *const u64).read_unaligned()
}

/// `n` barrier groups as legacy barriers (D3D12_RESOURCE_BARRIER, 8 words each: transitions
/// { 0, 0, resource, subresource, before, after } and at most one UAV barrier for all
/// resources, last). `shape` gives a texture's (mips, array slices, planes), `readable`
/// whether console code's pointer can be read.
pub unsafe fn legacy(groups: *const u8, n: usize, shape: &mut dyn FnMut(u64) -> (u32, u32, u32), readable: &dyn Fn(usize, usize) -> bool) -> (Vec<[u32; 8]>, Counts) {
    let (mut out, mut c, mut uav) = (Vec::new(), Counts::default(), false);
    for g in 0..n {
        let group = groups.add(g * GROUP_BYTES);
        let (ty, count, barriers) = (u32_at(group, 0), (u32_at(group, 4) as usize).min(MAX_BARRIERS), u64_at(group, 8) as usize as *const u8);
        let size = match ty {
            0 => GLOBAL_BYTES,
            1 => TEXTURE_BYTES,
            2 => BUFFER_BYTES,
            _ => 0,
        };
        if size == 0 || count == 0 || barriers.is_null() || !readable(barriers as usize, count * size) {
            c.unknown += (size == 0 || count != 0) as u32;
            continue;
        }
        for k in 0..count {
            let b = barriers.add(k * size);
            let unordered = (u32_at(b, 8) | u32_at(b, 12)) & ACCESS_UNORDERED_ACCESS != 0;
            match ty {
                0 => {
                    // everything written so far is visible to what follows
                    c.globals += 1;
                    uav = true;
                }
                2 => {
                    // buffers have no layout; unordered access before or after is the UAV barrier
                    // (for all resources: `list_x::resource_barrier_x`)
                    c.buffers += 1;
                    uav |= unordered;
                }
                _ => {
                    c.textures += 1;
                    let (before, after, res) = (u32_at(b, 0x10), u32_at(b, 0x14), u64_at(b, 0x18));
                    if before == LAYOUT_UNDEFINED || after == LAYOUT_UNDEFINED {
                        c.undefined += 1;
                        continue;
                    }
                    let (before, after) = (layout_state(before), layout_state(after));
                    if before == after || res == 0 {
                        uav |= unordered;
                        continue;
                    }
                    let range = [u32_at(b, 0x20), u32_at(b, 0x24), u32_at(b, 0x28), u32_at(b, 0x2C), u32_at(b, 0x30), u32_at(b, 0x34)];
                    let all = range[1] == 0 && range[0] == u32::MAX;
                    let subs = if all {
                        None
                    } else {
                        let (mips, slices, planes) = shape(res);
                        subresources(range, mips, slices, planes)
                    };
                    let t = |sub: u32| [0, 0, res as u32, (res >> 32) as u32, sub, before, after, 0];
                    match subs {
                        None => out.push(t(u32::MAX)),
                        Some(v) => out.extend(v.into_iter().map(t)),
                    }
                }
            }
        }
    }
    if uav {
        out.push([2, 0, 0, 0, 0, 0, 0, 0]);
    }
    (out, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct Texture {
        sync: [u32; 2],
        access: [u32; 2],
        layout: [u32; 2],
        res: u64,
        range: [u32; 6],
        flags: u32,
        _pad: u32,
    }

    #[repr(C)]
    struct Buffer {
        sync: [u32; 2],
        access: [u32; 2],
        res: u64,
        offset: u64,
        size: u64,
    }

    #[repr(C)]
    struct Group {
        ty: u32,
        n: u32,
        p: u64,
    }

    fn texture(layout: [u32; 2], access: [u32; 2], res: u64, range: [u32; 6]) -> Texture {
        Texture { sync: [0; 2], access, layout, res, range, flags: 0, _pad: 0 }
    }

    #[test]
    fn the_structures_have_the_sizes_read_from_the_title() {
        assert_eq!(std::mem::size_of::<Texture>(), TEXTURE_BYTES);
        assert_eq!(std::mem::size_of::<Buffer>(), BUFFER_BYTES);
        assert_eq!(std::mem::size_of::<Group>(), GROUP_BYTES);
    }

    #[test]
    fn layouts_are_the_states_resources_are_created_in() {
        assert_eq!(layout_state(2), 0x4);
        assert_eq!(layout_state(6), 0xC0);
        assert_eq!(layout_state(21), layout_state(6), "DIRECT_QUEUE_SHADER_RESOURCE");
        assert_eq!(layout_state(26), layout_state(3), "COMPUTE_QUEUE_UNORDERED_ACCESS");
        assert_eq!(layout_state(0), 0);
        assert_eq!(layout_state(18), 0, "DIRECT_QUEUE_COMMON");
        assert_eq!(layout_state(LAYOUT_UNDEFINED), 0);
    }

    #[test]
    fn subresource_ranges_name_mips_slices_and_planes() {
        // an index, all subresources
        assert_eq!(subresources([3, 0, 0, 0, 0, 0], 4, 2, 1), Some(vec![3]));
        assert_eq!(subresources([u32::MAX, 0, 0, 0, 0, 0], 4, 2, 1), None);
        // the whole texture as a range
        assert_eq!(subresources([0, 4, 0, 2, 0, 1], 4, 2, 1), None);
        // mip 1 of both slices; slice 1 only; the stencil plane of a 2-mip depth texture
        assert_eq!(subresources([1, 1, 0, 2, 0, 1], 4, 2, 1), Some(vec![1, 5]));
        assert_eq!(subresources([0, 4, 1, 1, 0, 1], 4, 2, 1), Some(vec![4, 5, 6, 7]));
        assert_eq!(subresources([0, 2, 0, 1, 1, 1], 2, 1, 2), Some(vec![2, 3]));
        // counts of 0 for slices / planes are one; a range past the end is cut
        assert_eq!(subresources([0, 1, 0, 0, 0, 0], 4, 2, 1), Some(vec![0]));
        assert_eq!(subresources([3, 9, 0, 1, 0, 1], 4, 2, 1), Some(vec![3]));
    }

    #[test]
    fn groups_become_transitions_and_one_uav_barrier() {
        let textures = [
            // render target → shader resource, whole texture
            texture([2, 6], [8, 0x80], 0x1_2345_6780, [u32::MAX, 0, 0, 0, 0, 0]),
            // copy dest → shader resource, mip 1 of 3 (one slice)
            texture([8, 6], [0x400, 0x80], 0x2000, [1, 1, 0, 1, 0, 1]),
            // undefined → render target: not expressible
            texture([LAYOUT_UNDEFINED, 2], [0x8000_0000, 8], 0x3000, [u32::MAX, 0, 0, 0, 0, 0]),
            // unordered access → unordered access: a UAV barrier
            texture([3, 3], [0x10, 0x10], 0x4000, [u32::MAX, 0, 0, 0, 0, 0]),
            // common → DIRECT_QUEUE_COMMON: nothing
            texture([0, 18], [0, 0], 0x5000, [u32::MAX, 0, 0, 0, 0, 0]),
        ];
        let buffers = [
            Buffer { sync: [0; 2], access: [0x10, 0x80], res: 0x6000, offset: 0, size: u64::MAX },
            Buffer { sync: [0; 2], access: [0x400, 0x1], res: 0x7000, offset: 0, size: u64::MAX },
        ];
        let groups = [
            Group { ty: 1, n: textures.len() as u32, p: textures.as_ptr() as u64 },
            Group { ty: 2, n: buffers.len() as u32, p: buffers.as_ptr() as u64 },
            Group { ty: 7, n: 1, p: 0 },
        ];
        let mut asked = Vec::new();
        let (out, c) = unsafe {
            legacy(
                groups.as_ptr() as *const u8,
                groups.len(),
                &mut |res| {
                    asked.push(res);
                    (3, 1, 1)
                },
                &|_, _| true,
            )
        };
        assert_eq!(c, Counts { textures: 5, buffers: 2, globals: 0, undefined: 1, unknown: 1 });
        assert_eq!(asked, [0x2000], "the shape is asked for only where a range needs it");
        assert_eq!(
            out,
            [
                [0, 0, 0x2345_6780, 1, u32::MAX, 0x4, 0xC0, 0],
                [0, 0, 0x2000, 0, 1, 0x400, 0xC0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn buffers_without_unordered_access_and_unreadable_groups_record_nothing() {
        let buffers = [Buffer { sync: [0; 2], access: [0x400, 0x2], res: 0x6000, offset: 0, size: u64::MAX }];
        let groups = [Group { ty: 2, n: 1, p: buffers.as_ptr() as u64 }, Group { ty: 1, n: 4, p: 0x10 }];
        let (out, c) = unsafe { legacy(groups.as_ptr() as *const u8, 2, &mut |_| (1, 1, 1), &|p, _| p != 0x10) };
        assert!(out.is_empty());
        assert_eq!(c, Counts { buffers: 1, unknown: 1, ..Counts::default() });
        // a global barrier orders everything
        let global = [[0u32, 0, 0x10, 0x80]];
        let groups = [Group { ty: 0, n: 1, p: global.as_ptr() as u64 }];
        let (out, c) = unsafe { legacy(groups.as_ptr() as *const u8, 1, &mut |_| (1, 1, 1), &|_, _| true) };
        assert_eq!(out, [[2, 0, 0, 0, 0, 0, 0, 0]]);
        assert_eq!(c.globals, 1);
    }
}
