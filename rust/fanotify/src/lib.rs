//! fanotify - File Access (FA) notify
//!
//! A personal project to create Rust bindings for the fanotify API, since the
//!  nix crate lacks them, and an existing crate isn't complete.
//!
//! Information derived from the following man pages:
//! * fanotify
//! * fanotify_init
//! * fanotify_mark

// TODO: Grab the constants from linux/fanotify.h

pub mod sys;

use std::ffi;
use std::fs;
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::Path;

type Result<T> = std::result::Result<T, io::Error>;

/* TODO: Make these into an enum
	* `FAN_EVENT_INFO_TYPE_FID`
	* `FAN_EVENT_INFO_TYPE_DFID`
	* `FAN_EVENT_INFO_TYPE_DFID_NAME`
	* `FAN_EVENT_INFO_TYPE_PIDFD`
	* `FAN_EVENT_INFO_TYPE_ERROR`
*/
/* Functions for:
	* FAN_EVENT_OK - If event structure fits inside remaining buffer (and event_len is valid).
	* FAN_EVENT_NEXT - Simply moves event_len bytes forward.
	* FAN_EVENT_METADATA_LEN - sizeof event_metadata struct
*/

#[derive(Clone, Copy, Default)]
struct EventMask {
	access: bool,
	open: bool,
	open_exec: bool,
	attrib: bool,
	create: bool,
	delete: bool,
	delete_self: bool,
	fs_error: bool,
	rename: bool,
	moved_from: bool,
	moved_to: bool,
	move_self: bool,
	modify: bool,
	close_write: bool,
	close_nowrite: bool,
	q_overflow: bool,
	access_perm: bool,
	open_perm: bool,
	open_exec_perm: bool,
	close: bool, // (close_write | close_nowrite)
	moved: bool, // (moved_from | moved_to) renamed "moved" to avoid keyword conflict
	ondir: bool
}

/// Source: fanotify_init
#[derive(Clone, Copy, Default)]
pub struct InitFlags {
	/// Mutually exclusive with content and notif
	pre_content: bool,
	/// Mutually exclusive with pre-content and notif
	content: bool,
	/// Mutually exclusive with pre-content and content
	notif: bool,
	cloexec: bool,
	nonblock: bool,
	/// Requires CAP_SYS_ADMIN
	unlimited_queue: bool,
	/// Requires CAP_SYS_ADMIN
	unlimited_marks: bool,
	/// Requires CAP_SYS_ADMIN
	report_tid: bool,
	/// Requires CAP_AUDIT_WRITE
	enable_audit: bool,
	/// Mutually exclusive with pre-content and content
	report_fid: bool,
	report_dir_fid: bool,
	report_name: bool,
	/// Synonym for (report_dir_fid | report_name)
	report_dfid_name: bool,
	/// Must be provided in conjunction with report_fid and report_dfid_name
	report_target_fid: bool,
	/// Synonym for (report_dfid_name | report_fid | report_target_fid)
	report_dfid_name_target: bool,
	/// Mutually exclusive with report_tid
	report_pfid: bool
}

/// Source: fanotify_init
#[derive(Clone, Copy, Default)]
pub struct EventFdFlags {
	rdonly: bool,
	wronly: bool,
	rdwr: bool,
	largefile: bool,
	cloexec: bool,
	append: bool,
	dsync: bool,
	noatime: bool,
	nonblock: bool,
	sync: bool
}

/// Source: fanotify_mark
#[derive(Clone, Copy, Default)]
struct MarkFlags {
	/// Only one of the following
	add: bool,
	remove: bool,
	flush: bool,
	/// Zero or more of the following:
	dont_follow: bool,
	onlydir: bool,
	mount: bool,
	filesystem: bool,
	ignored_mask: bool,
	ignore: bool,
	ignored_surv_modify: bool,
	ignore_surv: bool,
	evictable: bool
}

/// Source: fanotify_mark
#[derive(Clone, Copy, Default)]
struct EventFlags {
	access: bool,
	access_perm: bool,
	attrib: bool, // (since linux 5.1)
	close: bool, // (close_write | close_nowrite)
	close_write: bool,
	close_nowrite: bool,
	create: bool,
	delete: bool, // (since linux 5.1)
	delete_self: bool, // (since linux 5.1)
	/// Exclusive to mark mask
	event_on_child: bool,
	fs_error: bool, // (since linux 5.16)
	modify: bool,
	moved: bool, // (moved_from | moved_to)
	moved_from: bool, // (since linux 5.1)
	moved_to: bool,
	move_self: bool, // (since linux 5.1)
	ondir: bool,
	open: bool,
	open_exec: bool, //(since linux 5.0)
	open_exec_perm: bool, // (since linux 5.0)
	open_perm: bool,
	/// Exclusive to event mask
	q_overflow: bool,
	rename: bool, // (since linux 5.17)
}

#[derive(Debug)]
pub struct Event {
	pub mask: u64,
	pub file: fs::File,
	pub pid: u32
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
// Uses index and length since self-referencial structs not possible.
#[derive(Debug)]
pub struct EventIter {
	/// Buffer used when reading from `fan_fd`
	evt_buffer: [u8; 4096],
	/// Length that buffer is valid for
	evt_buffer_len: usize,
	/// Index of `event_buffer` where next event should start
	next_evt: usize
}

// TODO: Create has_pending_events() using poll() (needs lib/crate/implement).
impl Fanotify {
	/// Creates an fanotify instance with the given flags.
	///
	/// Passes the given flag parameters directly to `fanotify_init()`, and
	///  if successful, returns an `Fanotify` instance for further
	///  interactions.
	pub fn init(flags: i32, event_fd_flags: i32) -> Result<Self> {
		let fid = unsafe {
			sys::fanotify_init(flags, event_fd_flags)
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
	pub fn add_mark<P: AsRef<Path>>(&self, path: P, flags: i32, mask: i32) -> Result<()> {
		if let Some(p) = path.as_ref().to_str() {
			let c_path = ffi::CString::new(p).expect("Path to str will error if null byte.");

			// All bits except first three (FAN_MARK_{ADD,REMOVE,FLUSH})
			let add_flags = (flags & 0x7FFFFFF8) | sys::FAN_MARK_ADD;

			// Call mark
			let res = unsafe {
				sys::fanotify_mark(self.fan_fd.as_raw_fd(), add_flags, mask as u64, 0, c_path.as_ptr())
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

	/// Returns an [EventIter] with an iterable buffer of some notify events.
	///
	/// Since streaming iterators aren't (cleanly) possible, the returned
	///  iterator only contains a limited number of notify events. If more
	///  events are desired, this function must be called again.
	pub fn iter(&mut self) -> EventIter {
		let mut evti = EventIter {
			evt_buffer: [0; 4096],
			evt_buffer_len: 0,
			next_evt: 0
		};

		// Read contents into buffer and update length.
		if let Ok(n) = self.fan_fd.read(&mut evti.evt_buffer) {
			evti.evt_buffer_len = n;
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

impl Iterator for EventIter {
	type Item = Event;

	/// Iterates through events in the current buffer.
	fn next(&mut self) -> Option<Self::Item> {
		// If slice too small, or at/beyond end of buffer, end of iterator reached.
		if self.next_evt >= self.evt_buffer_len
		|| self.evt_buffer_len.saturating_sub(self.next_evt) < std::mem::size_of::<sys::event_metadata>() {
				return None
		}

		// Get event metadata from buffer
		let evt = unsafe {
			*(self.evt_buffer.as_ptr().offset(self.next_evt as isize) as *const sys::event_metadata)
		};

		// If event (somehow) extends beyond buffer length, return.
		if (self.next_evt + evt.event_len as usize) > self.evt_buffer_len {
			return None
		}

		// Event valid by this point. Update next start value.
		self.next_evt += evt.event_len as usize;

		// Return the event
		Some(Event {
			mask: evt.mask,
			file: unsafe {
				fs::File::from_raw_fd(evt.fd as i32)
			},
			pid: evt.pid
		})
	}

	/// Provide upper bound based on fanotify event minimum size.
	fn size_hint(&self) -> (usize, Option<usize>) {
		(0, Some(self.evt_buffer_len / std::mem::size_of::<sys::event_metadata>()))
	}
}

/*#[cfg(test)]
mod tests {
	use super::*;
}*/