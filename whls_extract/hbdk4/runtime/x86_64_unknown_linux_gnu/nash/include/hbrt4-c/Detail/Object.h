/// \file
/// \ref Object Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Object Object
/// The declaration of all HBRT object types,
/// used for representation various component units from top and down in HBRT
///
/// # List of Features
/// - Declare the struct of all hbrt4 object types
/// - Provide utility to compare the object types
///
/// # Task Sequence
/// - User can check if two objects are the same by \ref hbrt4ObjectIsSame
/// - \ref hbrt4ObjectPtrLess is a helper macro to help to use objects in C++ `std::map`
///
/// See all defined object types in \ref ObjectTypes
/// @{

/// Object to represent base struct for all HBRT types
///
/// \note
/// All types listed in \ref Object can be converted to
/// \ref Hbrt4Object by the macro \ref hbrt4AsObject
struct Hbrt4Object {
  /// Opaque pointer for the object.  If this equals to 0, this object is a \ref null_object
  /// \private_api
  uintptr_t priv_ptr;

  /// The object may be wrapped in some container type.
  /// This is the ptr to container and some other type info
  /// \private_api
  uintptr_t priv_opaque;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4Object);
/// \endcond

/// Define an opaque C struct
///
/// \ingroup Hbrt4Private
///
/// \private_api
#define HBRT4_PRIV_DEFINE_C_API_STRUCT(name)                                                                           \
  struct name {                                                                                                        \
    /** \private_api */                                                                                                \
    /** \remark */                                                                                                     \
    /** Use the macro \ref hbrt4AsObject instead of using `priv_object` directly */                                    \
    Hbrt4Object priv_object;                                                                                           \
  };                                                                                                                   \
  typedef struct name name

/// Convert any HBRT objects whose type listed in \ref Object to \ref Hbrt4Object type,
/// except the \ref Hbrt4Object itself
#define hbrt4AsObject(obj) obj.priv_object

/// Check if two object are exactly the same (They use the same address)
///
/// \internal
/// The pointer of container may be different
#define hbrt4ObjectIsSame(lhs, rhs) ((lhs.priv_object.priv_ptr) == (rhs.priv_object.priv_ptr))

/// Compare the opaque pointer of object by "<"
/// To make it possible to store the objects in C++ `std::map`
#define hbrt4ObjectPtrLess(lhs, rhs) ((lhs.priv_object.priv_ptr) < (rhs.priv_object.priv_ptr))

/// \defgroup ObjectTypes Defined Object types
/// @{

/// \ingroup BpuTask
///
/// Wrapper for new bpu driver API based on task
///
/// \see_module_doc{BpuTask}
///
/// \warning
/// This object cannot be used directly by bpu driver API
/// \see \ref hbrt4BpuTaskGetDriverHandle
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4BpuTask);

/// \ingroup Buffer
/// Object to represent actual memory buffer
///
/// \see_module_doc{Buffer}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Buffer);

/// \ingroup Command
/// Use to bind memory buffer value
///
/// \see_module_doc{Command}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Command);

/// \ingroup Command
/// Builder to create \ref Hbrt4Command
///
/// \see_module_doc{Command}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4CommandBuilder);

/// \ingroup Description
/// Object to represent user defined description
///
/// \see_module_doc{Description}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Description);

/// \ingroup Error
/// Object to represent detailed error in object whose type listed in \ref Object
///
/// \see_module_doc{Error}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Error);

/// \ingroup Graph
/// Object to represent computational graph
///
/// \see_module_doc{Graph}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Graph);

/// \ingroup Graph
/// Group of graphs which share similarities
///
/// \see_module_doc{Graph}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4GraphGroup);

/// \ingroup Hbm
/// Object to represent loaded model file
///
/// \see_module_doc{Hbm}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Hbm);

/// \ingroup HbmHeader
/// Object to represent hbm information. This object is created without loading the entire hbm
///
/// \see_module_doc{HbmHeader}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4HbmHeader);

/// \ingroup Instance
/// Store the global variable and state of HBRT
///
/// \see_module_doc{Instance}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Instance);

/// \ingroup Instance
/// Builder to create \ref Hbrt4Instance
///
/// \see_module_doc{Instance}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4InstanceBuilder);

/// \ingroup Preinit
/// Represent the UnKnown march Instance
///
///
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4PreInit);

/// \ingroup Preinit
/// Builder to create \ref Hbrt4PreInit
///
///
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4PreInitBuilder);

/// \ingroup Logger
/// Logger to redirect hbrt log
///
/// \see_module_doc{Logger}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Logger);

/// \ingroup Logger
/// Extra logger data. Reserved for future use
///
/// \see_module_doc{Logger}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4LoggerData);

/// \ingroup Memspace
/// Object to represent memory allocation information
///
/// \see_module_doc{Memspace}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Memspace);

/// \ingroup Node
/// Object to represent computation node in graph
///
/// \see_module_doc{Node}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Node);

/// \ingroup Pipeline
/// Fuse \ref Command
///
/// \see_module_doc{Pipeline}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Pipeline);

/// \ingroup Pipeline
/// Builder to create \ref Hbrt4Pipeline
///
/// \see_module_doc{Pipeline}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4PipelineBuilder);

/// \ingroup TensorType
/// Object to represent tensor type
///
/// \see_module_doc{TensorType}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4TensorType);

/// \ingroup Variable
/// Variable in the DAG
/// May or may not have statically known value
///
/// \see_module_doc{Version}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Variable);

/// \ingroup Value
/// Value of variable
///
/// \see_module_doc{Value}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Value);

/// \ingroup ValueBuilder
/// Builder of Value
///
/// \see_module_doc{Value}
///
/// \obj_decl_common
/// \obj_decl_own
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4ValueBuilder);

/// \ingroup Version
/// Object to represent version information
///
/// \see_module_doc{Version}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Version);

/// \ingroup Type
/// Object to represent type information
///
/// \see_module_doc{Type}
///
/// \obj_decl_common
/// \obj_decl_no_ownership
HBRT4_PRIV_DEFINE_C_API_STRUCT(Hbrt4Type);

/// @}

// NOTE(hehaoqian):
// Add name of new hbrt4 object types to `build.rs`

/// @}

HBRT4_PRIV_C_EXTERN_C_END

#undef HBRT4_PRIV_DEFINE_C_API_STRUCT
