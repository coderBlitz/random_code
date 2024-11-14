//! Custom allocator for the pico.
//!

// TODO: Investigate if this can be copied/initialized to memory rather than
//        reside purely on flash.
// NOTE: If this resides in the binary along with everything else, should get
//        copied into RAM without issue.
#[global_allocator]
static MY_ALLOX: Allox = Allox::new();

use core::{
	alloc::{GlobalAlloc, Layout},
	sync::atomic::{AtomicUsize, AtomicPtr, Ordering},
};

const ALLOX_POOL_SIZE: usize = 4;
const PTR_BITS: usize = core::mem::size_of::<usize>() * 8;

/** Modified custom allocator.

See allocator in rust folder for full details.
**/
// TODO: Const initialize all sectors with fixed base addresses.
// TODO: Determine if sector_pool is necessary if all sectors always exist
pub struct Allox {
	/// Allocated sector array.
	// Use of pointers necessary for atomic operations.
	sectors: [AtomicPtr<Sector>; ALLOX_POOL_SIZE],
	/// Allocation count for debugging/profiling
	// TODO: Keep or don't since debugging on-board is harder.
	total_allocs: AtomicUsize,
	/// Sectors available for use.
	sector_pool: [Sector; ALLOX_POOL_SIZE],
}
impl Allox {
	pub const fn new() -> Self {
		// Transmute necessary since atomics are not [Copy].
		// SAFETY: [AtomicPtr] guaranteed same size as `*mut Sector`.
		unsafe {
			Allox {
				sectors: core::mem::transmute([core::ptr::null_mut::<Sector>(); ALLOX_POOL_SIZE]),
				sector_pool: [Sector::NULL; ALLOX_POOL_SIZE + 1],
				total_allocs: AtomicUsize::new(0),
			}
		}
	}

	// Iterate sector pool and search for available/unallocated sector
	pub fn add_sector_for(&self, lay: Layout) -> Result<*mut u8, ()> {
		// Do initial check if there's room
		if self.num_sectors.load(Ordering::Relaxed) >= self.sectors.len() {
			return Err(())
		}

		// Find available sector
		let mut sec = None;
		for (_i,s) in self.sector_pool.iter().enumerate() {
			if s.base.load(Ordering::Acquire).is_null() {
				if s.viewers.compare_exchange(0, usize::MAX, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
					sec = Some(s);
					break;
				}
			}
		}

		// If sector found, allocate.
		match sec {
			// Allocate sector large enough so that `num_bits` <= PTR_BITS.
			Some(s) if s.alloc().is_ok() => {
				// Compute number of chunks/bits needed and claim first slots.
				let num_bits = lay.size().div_ceil(Sector::CHUNK_SIZE);
				let r = (!0) >> (PTR_BITS - num_bits);
				s.slots.store(r, Ordering::Release); // Claim slots

				// Reset viewer count. Safe now that base is non-null.
				s.viewers.store(0, Ordering::Release);

				// Try to push sector to array
				let n = self.num_sectors.load(Ordering::Acquire);
				if n < self.sectors.len() {
					if self.num_sectors.compare_exchange(n, n+1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
						// Store sector pointer
						self.sectors[n].store(s as *const _ as *mut _, Ordering::Release);

						return Ok(s.base.load(Ordering::Relaxed))
					}
				}

				Err(())
			},
			// If allocation fails, still reset viewer count.
			Some(s) => {
				s.viewers.store(0, Ordering::Release);
				Err(())
			},
			_ => Err(()),
		}
	}

	/// Iterate over valid [sectors] entries, calling `fun`, and returning the
	///  first [Some].
	///
	/// User function is called after a sector's `viewers` count has been
	///  incremented.
	///
	/// # Safety
	/// User function can modify the entire sector, or potentially invalidate
	///  the sector memory as a whole (since it can mutate the [Allox] object),
	///  so it is on the user to avoid invalidating the reference given to the
	///  function.
	///
	/// In short, the user must uphold the "Safety Contract" as specified in
	///  the [Allox] docs.
	unsafe fn iter_sectors<T, F: Fn(usize, &Sector) -> Option<T>>(&self, fun: F) -> Option<T> {
		// Iterate existing sectors
		for i in 0..self.num_sectors.load(Ordering::Relaxed) {
			let s = &self.sectors[i];
			// Load pointer and convert to [Sector] ref (if non-null).
			// SAFETY: See "Safety Contract" above.
			let p = s.load(Ordering::Relaxed);
			if let Some(sec) = unsafe { p.as_ref() } {
				// Check if not at max, then increment.
				let v = sec.viewers.load(Ordering::Acquire);
				if v < usize::MAX {
					// Increment viewer count before use.
					let r = sec.viewers.compare_exchange(v, v+1, Ordering::AcqRel, Ordering::Relaxed);

					// If viewers successfully incremented, call user fn.
					if let Ok(_) = r {
						if let Some(r) = fun(i, sec) {
							// Decrement viewer since usage is complete.
							sec.viewers.fetch_sub(1, Ordering::Release);
							return Some(r)
						}
					} else {
						// Decrement viewer since usage is complete.
						sec.viewers.fetch_sub(1, Ordering::Release);
					}
				}
			}
		}

		None
	}
}
unsafe impl GlobalAlloc for Allox {
	unsafe fn alloc(&self, lay: Layout) -> *mut u8 {
		// Try to allocate memory in existing sector
		let res = unsafe {
			self.iter_sectors(|i, sec| {
				// If request is successful, return ptr.
				match sec.request_mem(lay) {
					Ok(ptr) => Some(ptr),
					_ => None,
				}
			})
		};

		// If successfully allocated in existing sector, return ptr.
		if let Some(p) = res {
			self.total_allocs.fetch_add(1, Ordering::Relaxed);
			return p
		}

		// Else try to allocate new sector then assign memory there.
		match self.add_sector_for(lay) {
			Ok(p) => {
				self.total_allocs.fetch_add(1, Ordering::Relaxed);
				p
			},
			Err(_) => core::ptr::null_mut(),
		}
	}

	unsafe fn dealloc(&self, ptr: *mut u8, lay: Layout) {
		let _idx = self.iter_sectors(|i,sec| {
			match sec.release_mem(ptr, lay) {
				Ok(_) => Some(i),
				Err(_) => None,
			}
		});

		self.total_allocs.fetch_sub(1, Ordering::Relaxed);
	}
}

// TODO: Determine if viewers is necessary since sectors will never be freed or allocated.
pub struct Sector {
	/// How many threads are currently using this sector object. Yes usize is overkill.
	// usize::MAX or -1 indicates sector is being freed.
	viewers: AtomicUsize,
	/// Bitmap of slot status (1 - in use, 0 - Free).
	slots: AtomicUsize,
	/// First address in the allocated sector.
	base: AtomicPtr<u8>, // Use u8 ptr??
}
impl Sector {
	pub const NULL: Self = Self::null(); // Used for situations requiring [Copy].

	// Parameters of sectors (for fixed-size sectors).
	pub const ALLOC_SIZE: usize = 256;
	pub const CHUNK_SIZE: usize = Self::ALLOC_SIZE / (8 * core::mem::size_of::<usize>());
	pub const BASE_ADDR: *const u8 = 0x2000_0000; // TODO: Allow some amount of adjustment

	pub const fn from_idx(idx: usize) -> Self {
		Sector {
			viewers: AtomicUsize::new(0),
			slots: AtomicUsize::new(0),
			base: AtomicPtr::new((BASE_ADDR as *mut u8).add(idx * ALLOC_SIZE)),
		}
	}

	/// Attempt to claim a region of memory for `lay` and return start address.
	pub fn request_mem(&self, lay: Layout) -> Result<*mut u8, ()> {
		let num_bits = lay.size().div_ceil(Sector::CHUNK_SIZE);
		let mut r = (!0) >> (PTR_BITS - num_bits);
		let map = self.slots.load(Ordering::Acquire);

		for i in 0..(PTR_BITS - num_bits) {
			if (map & r) == 0 {
				let res = self.slots.compare_exchange(map, map | r, Ordering::AcqRel, Ordering::Relaxed);
				if res.is_ok() {
					// SAFETY: `i < PTR_BITS` so the chunk will always start inside base allocation range.
					unsafe {
						return Ok(self.base.load(Ordering::Relaxed).add(Sector::CHUNK_SIZE * i))
					}
				}
			}

			r <<= 1; // Shift r to align with map
		}

		Err(())
	}

	/// Release the memory for the given address and layout, if it is within this sector.
	pub fn release_mem(&self, addr: *mut u8, lay: Layout) -> Result<(), ()> {
		let p = self.base.load(Ordering::Relaxed);
		let off = (addr as isize - p as isize) / Self::CHUNK_SIZE as isize;

		// If pointer is within this sector, mark associated chunk(s) available.
		if (0..64).contains(&off) {
			let num_bits = lay.size().div_ceil(Sector::CHUNK_SIZE);
			let r = !(((!0) >> (PTR_BITS - num_bits)) << off);
			let map = self.slots.load(Ordering::Acquire);

			let res = self.slots.compare_exchange(map, map & r, Ordering::AcqRel, Ordering::Relaxed);
			match res {
				Ok(_) => Ok(()),
				Err(_) => Err(()),
			}
		} else {
			Err(())
		}
	}
}
