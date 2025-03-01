#![warn(clippy::pedantic)]
#![warn(clippy::alloc_instead_of_core)]
#![warn(clippy::std_instead_of_core)]
#![warn(clippy::std_instead_of_alloc)]
#![warn(clippy::semicolon_inside_block)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::same_name_method)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::as_underscore)]
#![warn(clippy::empty_structs_with_brackets)]
#![warn(clippy::else_if_without_else)]
#![warn(clippy::let_underscore_untyped)]
#![warn(clippy::string_to_string)]
#![warn(clippy::non_ascii_literal)]
#![warn(clippy::impl_trait_in_params)]
#![warn(clippy::try_err)]
#![warn(clippy::empty_drop)]
#![warn(clippy::str_to_string)]
#![warn(clippy::print_stderr)]
#![warn(clippy::print_stdout)]
#![warn(clippy::tests_outside_test_module)]
#![warn(clippy::missing_assert_message)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::deref_by_slicing)]
#![warn(clippy::format_push_string)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::default_union_representation)]
#![warn(clippy::exit)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::mixed_read_write_in_expression)]
#![warn(clippy::ref_patterns)]
#![warn(clippy::pub_without_shorthand)]
#![warn(clippy::needless_raw_strings)]
#![warn(clippy::error_impl_error)]
#![warn(clippy::float_arithmetic)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(let_underscore_drop)]
#![cfg_attr(not(any(test, hbrust_build_std)), warn(unused_crate_dependencies))]
#![forbid(non_ascii_idents)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

/// This module is to ensure generated binding does not depending on our extra `use` below

mod bindgen {
    // For struct with bit fields. Bindgen triggers these warnings
    #![allow(clippy::transmute_int_to_bool)]
    #![allow(clippy::unnecessary_cast)]
    #![allow(clippy::useless_transmute)]
    #![allow(clippy::pedantic)]
    #![allow(clippy::too_many_arguments)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::multiple_inherent_impl)]
    #![allow(clippy::unseparated_literal_suffix)]
    #![allow(clippy::let_underscore_untyped)]
    #![allow(clippy::missing_assert_message)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(improper_ctypes)]
    include!(concat!(env!("OUT_DIR"), "/", "bindings.rs"));
}

extern crate alloc;

pub use bindgen::*;
pub mod custom_lib;
pub mod variable_slice;
