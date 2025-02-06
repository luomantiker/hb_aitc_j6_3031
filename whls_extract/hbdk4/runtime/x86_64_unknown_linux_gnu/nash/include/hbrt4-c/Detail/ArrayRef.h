/// \file
/// \ref ArrayRef Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup ArrayRef ArrayRef
/// Read-only reference to continuous chunk of memory includes CString, CStringArray, Array, Int64Array, Int32 Array,
/// Float32Array, SizeTArray and PtrdiffTArray
///
/// # Task Sequence
///
/// These array references can be retrievd by various APIs,
/// such as \ref hbrt4TypeGetTensorStrides.
/// Note that user should never modify the memory region where it points.
///
/// @{

/// A const reference to a null terminated char array
///
/// \warning
/// This reference may contain binary data,
/// so `NULL` character may appear in the middle of the reference
/// \n
/// Using `printf("%s")` on the `data` pointer is not recommended,
/// unless you know no `NULL` character in the middle of the reference
///
/// \invariant
/// The pointer `data` is never `NULL`
/// \n
/// `data[length]` always contain `NULL` character, even if `length` is 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a `char` array of size `len+1`,
/// where the last one is always the `NULL` character
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4CStringRef {
  const char *data; ///< Pointer to the first character. This pointer is never `NULL`
  size_t len;       ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4CStringRef);
/// \endcond

/// A const reference to an array of null terminated c-string
///
/// \invariant
/// `data` are guaranteed to be `NULL` if `len` == 0
/// \n
/// All strings inside are not `NULL`
///
/// \remark
/// `data[0]` is the first string, `data[1]` is the second string, etc
/// \n
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a `const char*` array of size `len`,
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4CStringArrayRef {
  const char *const *data; ///< Pointer to the first string
  size_t len;              ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4CStringArrayRef);
/// \endcond

/// A const reference to a sized fragment of `char`
///
/// \warning
/// The `data` may *NOT* be `NULL` terminated
///
/// \invariant
/// `data` is be `NULL` if `len` == 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `char` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4ArrayRef {
  const void *data; ///< Pointer to the first character.
  size_t len;       ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4ArrayRef);
/// \endcond

/// A const reference to a sized fragment of `int64_t`
///
/// \invariant
/// `data` is `NULL` if `len` == 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `int64_t` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4Int64ArrayRef {
  const int64_t *data; ///< Pointer to the first `int64_t` value
  size_t len;          ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4Int64ArrayRef);
/// \endcond

/// A const reference to a sized fragment of `int32_t`
///
/// \invariant
/// `data` is `NULL` if `len` == 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `int32_t` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4Int32ArrayRef {
  const int32_t *data; ///< Pointer to the first `int32_t` value
  size_t len;          ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4Int32ArrayRef);
/// \endcond

/// A const reference to a sized fragment of `float`
///
/// \invariant
/// `data` is `NULL` if `len` == 0.
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `float` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4Float32ArrayRef {
  const float *data; ///< Pointer to the first float value
  size_t len;        ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4Float32ArrayRef);
/// \endcond

/// A const reference to a sized fragment of `size_t`
///
/// \invariant
/// `data` is `NULL` if `len` == 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `size_t` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4SizeTArrayRef {
  const size_t *data; ///< Pointer to the first `size_t` value
  size_t len;         ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4SizeTArrayRef);
/// \endcond

/// A const reference to a sized fragment of `ptrdiff_t`
///
/// \invariant
/// `data` is `NULL` if `len` == 0
///
/// \remark
/// User does not have the ownership of the underlying data
/// \n
/// This is equivalent to a const `ptrdiff_t` array of size `len`
/// \n
/// Reading data out of bound is an undefined behavior
struct Hbrt4PtrdiffTArrayRef {
  const ptrdiff_t *data; ///< Pointer to the first `ptrdiff_t` value
  size_t len;            ///< Length of the fragment
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4PtrdiffTArrayRef);
/// \endcond

/// @}

HBRT4_PRIV_C_EXTERN_C_END
