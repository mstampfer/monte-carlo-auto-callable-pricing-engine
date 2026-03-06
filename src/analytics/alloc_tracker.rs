use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Wraps the system allocator with atomic counters.
/// Activate with `#[global_allocator]` in the profiler binary only.
pub struct TrackingAllocator;

// Cumulative counters (monotonically increasing).
static CUMULATIVE_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUMULATIVE_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
// Live gauge (alloc increments, dealloc decrements).
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            CUMULATIVE_ALLOC_BYTES.fetch_add(layout.size(), Relaxed);
            CUMULATIVE_ALLOC_COUNT.fetch_add(1, Relaxed);
            let current = CURRENT_BYTES.fetch_add(layout.size(), Relaxed) + layout.size();
            // Update peak via CAS loop
            let mut peak = PEAK_BYTES.load(Relaxed);
            while current > peak {
                match PEAK_BYTES.compare_exchange_weak(peak, current, Relaxed, Relaxed) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT_BYTES.fetch_sub(layout.size(), Relaxed);
    }
}

/// Snapshot of cumulative allocation counters at a point in time.
#[derive(Clone, Copy, Default)]
pub struct AllocSnapshot {
    pub bytes: usize,
    pub count: usize,
}

impl TrackingAllocator {
    pub fn snapshot() -> AllocSnapshot {
        AllocSnapshot {
            bytes: CUMULATIVE_ALLOC_BYTES.load(Relaxed),
            count: CUMULATIVE_ALLOC_COUNT.load(Relaxed),
        }
    }

    pub fn peak_bytes() -> usize {
        PEAK_BYTES.load(Relaxed)
    }

    pub fn current_bytes() -> usize {
        CURRENT_BYTES.load(Relaxed)
    }

    pub fn reset_peak() {
        PEAK_BYTES.store(CURRENT_BYTES.load(Relaxed), Relaxed);
    }
}
