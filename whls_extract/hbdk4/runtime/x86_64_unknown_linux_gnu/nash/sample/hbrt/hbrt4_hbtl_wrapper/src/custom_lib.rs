use alloc::ffi::{CString, NulError};

const CUSTOM_LIBRARY_PATH_ENV: &str = "HBRT4_CUSTOM_LIBRARY_PATH";

pub fn get_custom_library() -> Result<Vec<CString>, NulError> {
    match std::env::var(CUSTOM_LIBRARY_PATH_ENV) {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(CString::new)
            .collect(),
        Err(_) => Ok(vec![]),
    }
}
