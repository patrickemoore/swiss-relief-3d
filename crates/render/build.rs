//! Build support for the self-contained `windows-gnullvm` toolchain.
//!
//! That toolchain links with the LLD bundled with Rust and ships import
//! libraries for only a handful of system DLLs. A few dependencies link others
//! by name (wgpu's OpenGL backend needs `opengl32`, the windowing and
//! clipboard crates need the rest). LLD can link against a DLL directly,
//! so this script puts a copy of the system's DLL on the library search
//! path. Only its export table is read; at run time the program loads the
//! system DLL as usual.
//!
//! Every other toolchain finds these libraries on its own and skips this.
//! `relief-viewer` carries the same script, for the application.

use std::env;
use std::fs;
use std::path::PathBuf;

/// System DLLs that dependencies link by name.
const DIRECTLY_LINKED_DLLS: [&str; 4] = ["opengl32.dll", "shell32.dll", "gdi32.dll", "advapi32.dll"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = |key: &str| env::var(format!("CARGO_CFG_TARGET_{key}")).unwrap_or_default();
    if target("OS") != "windows" || target("ABI") != "llvm" {
        return;
    }

    let Some(system_root) = env::var_os("SystemRoot") else {
        println!("cargo:warning=SystemRoot is not set; linking system libraries may fail");
        return;
    };
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"));
    for dll in DIRECTLY_LINKED_DLLS {
        let source = PathBuf::from(&system_root).join("System32").join(dll);
        if let Err(error) = fs::copy(&source, out_dir.join(dll)) {
            println!("cargo:warning=could not copy {}: {error}", source.display());
        }
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
