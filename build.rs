use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("could not {description}: {e}"));
    assert!(status.success(), "failed to {description}");
}

fn main() {
    println!("cargo:rerun-if-changed=native/coreml_bridge.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = out_dir.join("coreml_bridge.o");
    let archive = out_dir.join("libcoreml_bridge.a");

    run(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "clang", "-fobjc-arc", "-fmodules", "-c"])
            .arg("native/coreml_bridge.m")
            .arg("-o")
            .arg(&object),
        "compile the Objective-C Core ML bridge",
    );
    run(
        Command::new("ar").arg("crus").arg(&archive).arg(&object),
        "archive the Objective-C Core ML bridge",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=coreml_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
}
