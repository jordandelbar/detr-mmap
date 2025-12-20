use nix::libc;
use std::ffi::CString;
use std::os::raw::c_int;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemaphoreError {
    #[error("Failed to create semaphore: {0}")]
    CreateFailed(#[from] nix::errno::Errno),
    #[error("Invalid semaphore name")]
    InvalidName,
    #[error("Semaphore operation timed out")]
    Timeout,
}

pub struct Semaphore {
    sem: *mut libc::sem_t,
}

impl Semaphore {
    pub fn new(name: &str, initial_value: u32) -> Result<Self, SemaphoreError> {
        let c_name = CString::new(name).map_err(|_| SemaphoreError::InvalidName)?;

        let sem = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o644, initial_value) };

        if sem == libc::SEM_FAILED {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }

        Ok(Self { sem })
    }

    pub fn open(name: &str) -> Result<Self, SemaphoreError> {
        let c_name = CString::new(name).map_err(|_| SemaphoreError::InvalidName)?;

        let sem = unsafe { libc::sem_open(c_name.as_ptr(), 0) };

        if sem == libc::SEM_FAILED {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }

        Ok(Self { sem })
    }

    pub fn wait(&self) -> Result<(), SemaphoreError> {
        let ret = unsafe { libc::sem_wait(self.sem) };
        if ret != 0 {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }
        Ok(())
    }

    pub fn try_wait(&self) -> Result<bool, SemaphoreError> {
        let ret = unsafe { libc::sem_trywait(self.sem) };
        if ret == 0 {
            Ok(true)
        } else {
            let errno = nix::errno::Errno::last();
            if errno == nix::errno::Errno::EAGAIN {
                Ok(false)
            } else {
                Err(SemaphoreError::CreateFailed(errno))
            }
        }
    }

    pub fn post(&self) -> Result<(), SemaphoreError> {
        let ret = unsafe { libc::sem_post(self.sem) };
        if ret != 0 {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }
        Ok(())
    }

    pub fn value(&self) -> Result<i32, SemaphoreError> {
        let mut val: c_int = 0;
        let ret = unsafe { libc::sem_getvalue(self.sem, &mut val) };
        if ret != 0 {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }
        Ok(val)
    }

    pub fn unlink(name: &str) -> Result<(), SemaphoreError> {
        let c_name = CString::new(name).map_err(|_| SemaphoreError::InvalidName)?;
        let ret = unsafe { libc::sem_unlink(c_name.as_ptr()) };
        if ret != 0 {
            return Err(SemaphoreError::CreateFailed(nix::errno::Errno::last()));
        }
        Ok(())
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            libc::sem_close(self.sem);
        }
    }
}

unsafe impl Send for Semaphore {}
unsafe impl Sync for Semaphore {}
