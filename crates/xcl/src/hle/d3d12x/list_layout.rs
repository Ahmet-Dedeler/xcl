//! Command list slots of the newer console layout (`real`: "command queue layouts").
//!
//! From #61 on every console method sits two slots later than in the older layout – read
//! from a retail game's call sites by argument shape: #61 ExecuteIndirect (older #59), #62
//! CopyMemoryToMemoryX, #70 OMSetDepthBounds, #71 / #72 Write{32,64}BitValueBottomOfPipeX,
//! #74 Wait64BitValueX, #85 CopyTextureRegionX, #86 CopyResourceX, #126 / #127
//! RSSetShadingRate / RSSetShadingRateImage. #50–#52 are where they were, so the two added
//! methods are in #53–#60 (no call seen there). #139 is Reset with one argument more.
//! #146 Barrier is at that slot on this layout (the same game); #148 AllocateUploadMemory was
//! found on XboxBvhBuilder.dll, whose layout is not known – both are left where they are.

/// first slot of the shift
pub const FIRST_SHIFTED: usize = 61;
/// last slot the shift is applied to
pub const LAST_SHIFTED: usize = 145;
/// Reset(allocator, initial pipeline, 0)
pub const RESET: usize = 139;

/// The older layout's slot of the method at `newer` (None: the same slot in both).
pub fn older_slot(newer: usize) -> Option<usize> {
    match newer {
        RESET => Some(10),
        FIRST_SHIFTED..=LAST_SHIFTED => Some(newer - 2),
        _ => None,
    }
}

/// The lists of a submission a PC queue of type `queue` takes: a list of another type is left
/// out (the PC removes the device for it); a type that is not known is taken as the queue's.
pub fn for_queue<T: Copy>(queue: Option<i32>, lists: &[(T, Option<i32>)]) -> Vec<T> {
    lists.iter().filter(|(_, ty)| queue.is_none() || ty.is_none() || *ty == queue).map(|e| e.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A retail game runs a state list per queue before its batches: direct on the direct
    /// queue, compute on the compute ones – a list of another type never reaches the PC queue.
    #[test]
    fn a_submission_keeps_the_lists_of_the_queues_type() {
        const DIRECT: Option<i32> = Some(0);
        const COMPUTE: Option<i32> = Some(2);
        let lists = [(1, DIRECT), (2, COMPUTE), (3, None), (4, DIRECT)];
        assert_eq!(for_queue(DIRECT, &lists), [1, 3, 4]);
        assert_eq!(for_queue(COMPUTE, &lists), [2, 3]);
        assert_eq!(for_queue(None, &lists), [1, 2, 3, 4]);
        assert!(for_queue::<usize>(DIRECT, &[]).is_empty());
    }

    /// The pairs read from call sites (SLOTS.md), and the slots that do not move.
    #[test]
    fn newer_list_slots_are_the_older_ones_two_later_from_61() {
        for (newer, older) in [(61, 59), (62, 60), (70, 68), (71, 69), (72, 70), (74, 72), (85, 83), (86, 84), (126, 124), (127, 125), (139, 10)] {
            assert_eq!(older_slot(newer), Some(older), "#{newer}");
        }
        // the PC's methods and their console moves below the shift, Barrier, AllocateUploadMemory
        for same in [9, 12, 25, 26, 32, 50, 51, 52, 60, 146, 148] {
            assert_eq!(older_slot(same), None, "#{same}");
        }
        // no two newer slots name one older method
        let mut seen = std::collections::HashSet::new();
        assert!((0..200).filter_map(older_slot).all(|o| seen.insert(o)));
    }
}
