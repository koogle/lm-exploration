#[cfg(target_os = "macos")]
mod platform {
    use std::{
        ffi::{CStr, CString},
        os::raw::c_char,
        process,
    };

    unsafe extern "C" {
        fn coreml_run_smoke_model(
            package_path: *const c_char,
            input: f32,
            output: *mut f32,
            error_message: *mut *mut c_char,
        ) -> i32;
        fn coreml_free_string(value: *mut c_char);
    }

    pub fn run(args: &[String]) {
        let Some(path) = args.get(1) else {
            eprintln!("usage: local-mac-inference <model.mlpackage> [input]");
            process::exit(2);
        };
        let input = match args.get(2).map(|v| v.parse::<f32>()).transpose() {
            Ok(value) => value.unwrap_or(2.0),
            Err(_) => {
                eprintln!("input must be a 32-bit float");
                process::exit(2);
            }
        };
        let path = CString::new(path.as_bytes()).unwrap_or_else(|_| {
            eprintln!("model path contains an interior NUL byte");
            process::exit(2);
        });
        let mut output = 0.0_f32;
        let mut error = std::ptr::null_mut();
        let status =
            unsafe { coreml_run_smoke_model(path.as_ptr(), input, &mut output, &mut error) };
        if status != 0 {
            let message = if error.is_null() {
                "unknown Core ML error".to_owned()
            } else {
                let message = unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { coreml_free_string(error) };
                message
            };
            eprintln!("Core ML failed: {message}");
            process::exit(1);
        }
        let expected = 3.0 * input + 1.0;
        if (output - expected).abs() > 1e-4 {
            eprintln!("unexpected output: expected={expected} got={output}");
            process::exit(1);
        }
        println!("input={input} output={output} (verified)");
    }
}

#[cfg(target_os = "macos")]
fn main() {
    platform::run(&std::env::args().collect::<Vec<_>>());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This executable calls Apple's Core ML framework and must run on macOS.");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_case_expected_value_is_stable() {
        let input = 2.0_f32;
        let expected = 3.0 * input + 1.0;
        assert_eq!(expected, 7.0);
    }
}
