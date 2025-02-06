/// \file
/// \ref Status Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Status
/// @{

/// Simple status enum to indicate whether success or failed
/// \internal
/// Error must be negative to make it possible to fuse within `int32_t`
/// \n
/// The enum key is inspired by the IREE project
enum Hbrt4Status {

  /// API done normally as expected
  HBRT4_STATUS_OK = 0,

  /// The output value is not statically known
  /// #### Examples
  /// - The return of \ref hbrt4MemspaceGetSize when size is dynamic
  /// - The return of \ref hbrt4TypeGetTensorDims when number of dims is dynamic
  HBRT4_STATUS_DYNAMIC_VALUE = -1,

  /// The input argument to an API is a \ref null_object. \n
  /// Note that destructor like API, such as \ref hbrt4CommandDestroy,
  /// will **NOT** return this
  HBRT4_STATUS_NULL_OBJECT = -2,

  /// Function argument is invalid
  ///
  /// For null input object, API prefers to return \ref HBRT4_STATUS_NULL_OBJECT instead
  ///
  /// #### Examples
  /// - The return of \ref hbrt4VersionGetSemantic when input `version` is \ref null_object
  /// - The output pointer is `NULL`
  HBRT4_STATUS_INVALID_ARGUMENT = -3,

  /// The operation was rejected because the object does not satisfy the
  /// precondition required to call the API
  HBRT4_STATUS_FAILED_PRECONDITION = -4,

  /// A referenced resource could not be found
  /// #### Examples
  /// - The return of \ref hbrt4HbmGetGraphByName when model with the required
  ///   name not found
  HBRT4_STATUS_NOT_FOUND = -5,

  /// The resource the caller attempted to create already exists.
  /// #### Examples
  /// - The return of \ref hbrt4CommandBuilderBindBuffer when a memspace is bind twice
  HBRT4_STATUS_ALREADY_EXISTS = -6,

  /// An undefined behavior has been detected by HBRT
  ///
  /// \warning
  /// User should always avoid undefined behavior.
  /// \n
  /// The undefined behavior detection provided by HBRT is best-in-effort basis,
  /// and the undefined behavior is not guaranteed to be detected
  ///
  /// #### Examples
  /// - The return of \ref hbrt4VersionGetCString when the input object is not a
  ///   `Hbrt4Version` object returned by HBRT API
  /// - The return of \ref  hbrt4CommandBuilderInto when the input object has
  ///   been destroyed
  HBRT4_STATUS_UNDEFINED_BEHAVIOR = -7,

  /// Some resource type has been exhausted.
  /// Usually occur when we are out of memory
  ///
  /// #### Examples
  /// - The return of \ref hbrt4HbmCreateByFilename when not enough bpu memory
  ///   to load the hbm file
  HBRT4_STATUS_RESOURCE_EXHAUSTED = -8,

  /// IO related error ocurred, such as file read/write error
  ///
  /// #### Examples
  /// - The return of \ref hbrt4HbmCreateByFilename when the input file does not
  ///   exist on disk
  HBRT4_STATUS_IO_ERROR = -9,

  /// Any POSIX API returns an error, when error could not be mapped to the enum above
  HBRT4_STATUS_OS_ERROR = -10,

  /// Unexpected input data detected
  HBRT4_STATUS_BAD_DATA = -11,

  /// Some internal error happens inside HBRT
  /// Return this usually indicates that HBRT4 has a bug
  ///
  /// This enum variant will not be documented in the return value HBRT4 API doc
  HBRT4_STATUS_INTERNAL_ERROR = -12,

  /// Critical error has occured that HBRT4 has calls `panic`
  /// This is similar to C++ throwing an exception
  ///
  /// Return this usually indicates that HBRT4 has a bug,
  /// or some undefined behavior and been triggered.
  ///
  /// This enum variant will not be documented in the return value HBRT4 API doc
  ///
  /// \warning
  /// Do not continue your program after this status has been detected.
  /// HBRT4 may be inside an invalid state
  ///
  /// \see What is panic: <https://doc.rust-lang.org/std/macro.panic.html>
  HBRT4_STATUS_PANIC = -13,

  /// Not supported feature of HBRT is used
  HBRT4_STATUS_NOT_SUPPORTED = -14,

  /// Not implemented feature of HBRT is used
  HBRT4_STATUS_NOT_IMPLEMENTED = -15,

  /// Unknown error, or error that could not be mapped to this enum.
  HBRT4_STATUS_UNKNOWN = -16,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4Status);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
