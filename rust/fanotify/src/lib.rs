//! fanotify - File Access (FA) notify
//!
//! A personal project to create Rust bindings for the fanotify API, since the
//!  nix crate lacks them, and an existing crate isn't complete.
//!
//! Information derived from the following man pages:
//! * fanotify
//! * fanotify_init
//! * fanotify_mark

pub mod sys;
pub mod flags;

use std::convert::TryFrom;
use std::mem;
use std::ffi;
use std::fs;
use std::io::{self, Read};
use std::os::fd::{self, AsFd, AsRawFd, FromRawFd, IntoRawFd};
use std::path::Path;
use flags::*;

type Result<T> = std::result::Result<T, io::Error>;

/// Represents a single event returned through fanotify.
///
/// TODO: Figure out how to fit all event types in this, or expand with an enum
///  or similar.
/// TODO: Needs to handle directory events (and those with files associated).
#[derive(Debug)]
pub struct Event {
	pub mask: EventFlags,
	pub file: fs::File,
	pub pid: u32
}

/// Should represent the various "file" references that fanotify returns.
pub enum EventFile {
	Fd(fs::File),
	Fh // TODO: Create FileHandle (or other) containing sys::file_handle plus f_handle bytes
}

/// Should represent the various extra info that can be returned.
pub enum InfoType {
	Fid,
	Dfid,
	DfidName,
	PidFd,
	Error
}
impl TryFrom<i32> for InfoType {
	type Error = ();

	fn try_from(n: i32) -> std::result::Result<Self, ()> {
		match n {
			sys::FAN_EVENT_INFO_TYPE_FID => Ok(Self::Fid),
			sys::FAN_EVENT_INFO_TYPE_DFID_NAME => Ok(Self::DfidName),
			sys::FAN_EVENT_INFO_TYPE_DFID => Ok(Self::Dfid),
			sys::FAN_EVENT_INFO_TYPE_PIDFD => Ok(Self::PidFd),
			sys::FAN_EVENT_INFO_TYPE_ERROR => Ok(Self::Error),
			_ =>Err(())
		}
	}
}

/// Fanotify instance
// `event_buffer_len` only exists because streaming iterator not possible.
#[derive(Debug)]
pub struct Fanotify {
	/// Hold the fd returned by fanotify. Converted to OwnedFd for Drop trait.
	fan_fd: fs::File,
}

/// Iterator returned by [Fanotify::iter()] which iterates the supplied event
/// buffer.
///
/// Since this iterate the supplied buffer, this iterator will not continuously
///  supply events. If continuous events are desired, [Fanotify::iter()] must
///  be called repeatedly.
#[derive(Debug)]
pub struct EventIter<'a> {
	/// Buffer used when reading from `fan_fd`
	evt_buffer: Box<[u8; 4096]>,
	/// Slice of valid buffer remaining
	next_buf: &'a [u8]
}

// TODO: Create has_pending_events() using poll() (needs lib/crate/implement).
// TODO: Add allow()/deny() or similar for responses to PERM events.
impl Fanotify {
	/// Creates an fanotify instance with the given flags.
	///
	/// Passes the given flag parameters directly to `fanotify_init()`, and
	///  if successful, returns an `Fanotify` instance for further
	///  interactions.
	pub fn init(flags: &InitFlags, event_fd_flags: &EventFdFlags) -> Result<Self> {
		let fid = unsafe {
			sys::fanotify_init(flags.to_bits(), event_fd_flags.to_bits())
		};
		let err = io::Error::last_os_error();

		if fid == -1 {
			return Err(err);
		}

		Ok(Self {
			fan_fd: unsafe { fs::File::from_raw_fd(fid) },
		})
	}

	/// Mark a path for which notification events are desired.
	///
	/// Passes the given flag parameters directly to `fanotify_mark()`.
	pub fn add_mark<P: AsRef<Path>>(&self, path: P, mtype: &MarkType, flags: &MarkFlags, mask: &EventFlags) -> Result<()> {
		fn inner(slf: &Fanotify, path: &Path, mtype: &MarkType, flags: &MarkFlags, mask: &EventFlags) -> Result<()> {
			if let Some(p) = path.to_str() {
				let c_path = ffi::CString::new(p).expect("Path to str will error if null byte.");

				// All bits except first three (FAN_MARK_{ADD,REMOVE,FLUSH})
				let add_flags = (flags.to_bits() & 0x7FFFFE6C) | sys::FAN_MARK_ADD | mtype.to_bits();

				// Call mark
				let res = unsafe {
					sys::fanotify_mark(slf.fan_fd.as_raw_fd(), add_flags, mask.to_bits() as u64, 0, c_path.as_ptr())
				};

				// If mark failed, return error
				let err = io::Error::last_os_error();
				if res == -1 {
					return Err(err);
				}

				return Ok(());
			}

			// Return this error to be consistent with error types.
			Err(io::Error::new(io::ErrorKind::InvalidInput, "Path contains invalid character(s)."))
		}

		inner(self, path.as_ref(), mtype, flags, mask)
	}

	/// Returns an [EventIter] with an iterable buffer of some notify events.
	///
	/// Since streaming iterators aren't (cleanly) possible, the returned
	///  iterator only contains a limited number of notify events. If more
	///  events are desired, this function must be called again.
	pub fn iter(&mut self) -> EventIter {
		let mut evti = EventIter {
			evt_buffer: Box::new([0; 4096]),
			next_buf: &[]
		};

		/* Read contents into buffer and update length.
		Unsafe required since lifetime of evti differs from &self, but
		 returned struct owns the boxed array. Since the boxed array will
		 live as long as the slice, slice will always be valid if it uses
		 said array.
		*/
		if let Ok(n) = self.fan_fd.read(&mut evti.evt_buffer[..]) {
			evti.next_buf = unsafe {
				std::slice::from_raw_parts(evti.evt_buffer.as_ptr(), n)
			};
		}

		evti
	}

	/// Clear all marks for mounts.
	pub fn clear_mnt_marks(&mut self) -> Result<()> {
		// Set flags and create valid pathname (flushing still requires pathname be valid).
		let flags = sys::FAN_MARK_FLUSH | sys::FAN_MARK_MOUNT;
		let root = ffi::CString::new("/").expect("String literal should not contain null bytes.");

		// Make call to flush
		let res = unsafe {
			sys::fanotify_mark(self.fan_fd.as_raw_fd(), flags, 0, 0, root.as_ptr())
		};
		let err = io::Error::last_os_error();

		if res == -1 {
			return Err(err);
		}

		Ok(())
	}
	/// Clear all marks for filesystems.
	pub fn clear_fs_marks(&mut self) -> Result<()> {
		// Set flags and create valid pathname (flushing still requires pathname be valid).
		let flags = sys::FAN_MARK_FLUSH | sys::FAN_MARK_FILESYSTEM;
		let root = ffi::CString::new("/").expect("String literal should not contain null bytes.");

		// Make call to flush
		let res = unsafe {
			sys::fanotify_mark(self.fan_fd.as_raw_fd(), flags, 0, 0, root.as_ptr())
		};
		let err = io::Error::last_os_error();

		if res == -1 {
			return Err(err);
		}

		Ok(())
	}
	/// Clear all marks on specific files and directories.
	pub fn clear_file_marks(&mut self) -> Result<()> {
		// Set flags and create valid pathname (flushing still requires pathname be valid).
		let flags = sys::FAN_MARK_FLUSH;
		let root = ffi::CString::new("/").expect("String literal should not contain null bytes.");

		// Make call to flush
		let res = unsafe {
			sys::fanotify_mark(self.fan_fd.as_raw_fd(), flags, 0, 0, root.as_ptr())
		};
		let err = io::Error::last_os_error();

		if res == -1 {
			return Err(err);
		}

		Ok(())
	}
	/// Clear all marks (mounts, filesystem, and specific files/dirs).
	///
	/// Equivalent to calling each of [clear_mnt_marks()], [clear_fs_marks()],
	///  and [clear_file_marks()].
	pub fn clear_all_marks(&mut self) -> Result<()> {
		if let Err(e) = self.clear_mnt_marks() {
			return Err(e);
		}
		if let Err(e) = self.clear_fs_marks() {
			return Err(e);
		}
		if let Err(e) = self.clear_file_marks() {
			return Err(e);
		}

		Ok(())
	}
}
impl AsFd for Fanotify {
	fn as_fd(&self) -> fd::BorrowedFd {
		self.fan_fd.as_fd()
	}
}
impl AsRawFd for Fanotify {
	fn as_raw_fd(&self) -> fd::RawFd {
		self.fan_fd.as_raw_fd()
	}
}
impl IntoRawFd for Fanotify {
	fn into_raw_fd(self) -> fd::RawFd {
		self.fan_fd.into_raw_fd()
	}
}

const EVT_META_SIZE: usize = mem::size_of::<sys::event_metadata>();
impl<'a> Iterator for EventIter<'a> {
	type Item = Event;

	/// Iterates through events in the current buffer.
	fn next(&mut self) -> Option<Self::Item> {
		// If slice too small (or empty), end of iterator reached.
		if self.next_buf.len() < EVT_META_SIZE {
				return None
		}

		/* Get event metadata from buffer
		Pointer guaranteed to be valid, since next_buf always points to a valid
		 region of evt_buffer.
		*/
		let evt = unsafe {
			&*(self.next_buf.as_ptr() as *const sys::event_metadata)
		};

		// If event (somehow) extends beyond buffer length, return.
		if (evt.event_len as usize) > self.next_buf.len() {
			return None
		}

		// If metadata version mismatch, panic
		assert_eq!(evt.vers, sys::FANOTIFY_METADATA_VERSION as u8);

		// Slice for ease of parsing supplementary info.
		let full_evt = &self.next_buf[..evt.event_len as usize];
		if evt.event_len as usize > EVT_META_SIZE {
			eprintln!("Long event len: {}", evt.event_len);

			// Loop over additional info
			let mut info_remain = &full_evt[EVT_META_SIZE as usize..]; // Start with full event to guarantee first loop
			eprintln!("Additional info length: {}", info_remain.len());

			// TODO: Check what maximum extra info structures is and possibly remove loop.
			// Since there should be info for distinct things (file, parent dir, etc.),
			//  a loop might not make as much sense as just parsing N many times.
			while !info_remain.is_empty() {
				// Get common header
				let info_hdr = unsafe {
					&*(info_remain.as_ptr() as *const sys::event_info_header)
				};
				eprintln!("Info type {}, with len {}", info_hdr.info_type, info_hdr.len);

				// Get full info struct based on header
				if let Ok(info_type) = InfoType::try_from(info_hdr.info_type as i32) {
					match info_type {
						InfoType::Fid | InfoType::Dfid => {
							eprintln!("\tINFO_(D)FID:");
							let info = unsafe {
								&*(info_remain.as_ptr() as *const sys::event_info_fid)
							};
							// Get handle bytes
							let handle: &[u8] = unsafe {
								std::slice::from_raw_parts(info.file_handle.handle.as_ptr(), info.file_handle.handle_bytes as usize)
							};

							// Dump info for dev/debug
							eprintln!("\t\tfsid({:?})\n\t\tfile_handle({:?})", info.fsid, info.file_handle);
						},
						InfoType::DfidName => {
							eprintln!("\tINFO_(D)FID_NAME");
							let info = unsafe {
								&*(info_remain.as_ptr() as *const sys::event_info_fid)
							};

							// Get handle bytes
							let handle: &[u8] = unsafe {
								std::slice::from_raw_parts(info.file_handle.handle.as_ptr(), info.file_handle.handle_bytes as usize)
							};

							// Filename guaranteed null-terminated by fanotify API
							let name_ptr = unsafe {
								(&info.file_handle.handle as *const _ as *const i8).offset(info.file_handle.handle_bytes as isize)
							};
							let fname = unsafe {
								ffi::CStr::from_ptr(name_ptr)
							};

							eprintln!("\t\tfsid({:X?})\n\t\tfile_handle({:?})\n\t\tname({:X?})", info.fsid, info.file_handle, fname);
						},
						_ => {}
					}
				} else {
					eprintln!("\tUnrecognized info type.");
				}

				// Move info slice forward by current length
				info_remain = &info_remain[info_hdr.len as usize..];
			}
		}

		// Event valid by this point. Move slice start to end of this event.
		self.next_buf = &self.next_buf[evt.event_len as usize..];

		// Final check of FD for proper event type
		if evt.fd == sys::FAN_NOFD || evt.fd == sys::FAN_NOPIDFD || evt.fd == sys::FAN_EPIDFD {
			eprintln!("File handle vs descriptor not properly handled.");
			return None;
		}

		/* Return the event
		File descriptor guaranteed valid by fanotify API.
		*/
		Some(Event {
			mask: EventFlags::from_bits(evt.mask as i32),
			file: unsafe {
				fs::File::from_raw_fd(evt.fd as i32)
			},
			pid: evt.pid
		})
	}

	/// Provide upper bound based on fanotify event minimum size.
	fn size_hint(&self) -> (usize, Option<usize>) {
		(0, Some(self.next_buf.len() / EVT_META_SIZE))
	}
}
