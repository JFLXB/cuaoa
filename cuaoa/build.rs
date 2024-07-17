use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let aoa = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("internal");
    let make_status = Command::new("make")
        .arg("-C")
        .arg(aoa.to_str().unwrap())
        .status()
        .expect("Failed to execute make");

    assert!(make_status.success(), "Building the CUAOA package failed");

    let lib_aoa = aoa.join("lib");

    // Custatevec
    let conda_libs =
        Path::new(&env::var("CONDA_PREFIX").expect("Not in a conda environment.")).join("lib");
    println!("cargo:rustc-link-search=native={}", conda_libs.display());
    println!("cargo:rustc-link-lib=dylib=custatevec");
    println!("cargo:rustc-link-lib=dylib=lbfgs");

    // aoa library
    println!("cargo:rustc-link-search=native={}", lib_aoa.display());
    println!("cargo:rustc-link-lib=dylib=cuaoalg");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_aoa.display());

    // Enusre rebuild when changed.
    println!("cargo:rerun-if-changed={}", aoa.display());
}
