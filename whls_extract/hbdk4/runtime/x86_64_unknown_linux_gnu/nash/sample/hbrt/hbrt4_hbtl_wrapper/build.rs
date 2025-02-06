use std::env;
use std::path::{Path, PathBuf};

use hbrt4_build::bindgen::{
    dump_env_for_debug, dump_preprocessed, get_bindgen_builder_with_common_cc_args, setup_libclang,
};
use hbrt4_build::io::copy_if_different;

/// Generate bindings for register function
fn generate_cpp_to_rust_bindings(include_dir: &Path, legacy_include_dir: &Path, out_path: &Path) {
    let include_file = include_dir.join("hbtl_c_api.h");
    let include_file = include_file.as_os_str().to_str().unwrap();
    let include_dir = include_dir.as_os_str().to_str().unwrap();
    let legacy_include_dir = legacy_include_dir.as_os_str().to_str().unwrap();
    println!("cargo:rerun-if-changed={include_file}");

    let clang_args = vec![
        format!("--include-directory={legacy_include_dir}"),
        format!("-I{include_dir}"),
    ];

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = get_bindgen_builder_with_common_cc_args()
        .header(include_file)
        .newtype_enum(".*")
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks));

    dump_preprocessed(&bindings, out_path);
    // Finish the builder and generate the bindings.
    let bindings = bindings.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_path.join("bindings.rs.tmp"))
        .expect("Couldn't write bindings!");
    copy_if_different(
        out_path.join("bindings.rs.tmp"),
        out_path.join("bindings.rs"),
    );
    dump_env_for_debug(out_path);
}

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-lib=static=hbrt4-hbtl-interface");

    println!("cargo:rustc-link-lib=dylib=hbtl");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=lib/wrapper/wrapper.cpp");
    let include_dir = Path::new(&format!("{dir}/include/"))
        .canonicalize()
        .unwrap();
    let legacy_include_dir = Path::new(&format!("{dir}/../hbrt4_legacy/include/"))
        .canonicalize()
        .unwrap();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    setup_libclang();

    generate_cpp_to_rust_bindings(&include_dir, &legacy_include_dir, &out_path);
}
