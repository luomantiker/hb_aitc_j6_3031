/// \file
/// The only HBRT C header that should be directly included by user
///
/// Include this file will be enough to include all HBRT C API and structs
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \since v4.0.1
///
/// #### Examples
/// \code
/// #include "hbrt4-c/hbrt4-c.h"
/// \endcode

#pragma once

/// \cond hidden
///
/// Guard to define user does not include header in `Detail` folder directly
#define HBRT4_DETAIL_GUARD
/// \endcond

/// \example cpu_bpu_mixed_model.cpp
/// Example code to run cpu and bpu mixed model

// IWYU pragma: begin_exports

#include "hbrt4-c/Detail/Enums/AllEnums.h"

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Batch.h"
#include "hbrt4-c/Detail/BpuTask.h"
#include "hbrt4-c/Detail/Buffer.h"
#include "hbrt4-c/Detail/Command.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Description.h"
#include "hbrt4-c/Detail/Error.h"
#include "hbrt4-c/Detail/External/NewDriverApi.h"
#include "hbrt4-c/Detail/Graph.h"
#include "hbrt4-c/Detail/GraphGroup.h"
#include "hbrt4-c/Detail/Hbm.h"
#include "hbrt4-c/Detail/HbmHeader.h"
#include "hbrt4-c/Detail/Instance.h"
#include "hbrt4-c/Detail/Logger.h"
#include "hbrt4-c/Detail/Memspace.h"
#include "hbrt4-c/Detail/Node.h"
#include "hbrt4-c/Detail/NodeContextSwitch.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Pipeline.h"
#include "hbrt4-c/Detail/PreInit.h"
#include "hbrt4-c/Detail/Profiling.h"
#include "hbrt4-c/Detail/Status.h"
#include "hbrt4-c/Detail/TensorType.h"
#include "hbrt4-c/Detail/Type.h"
#include "hbrt4-c/Detail/Value.h"
#include "hbrt4-c/Detail/Variable.h"
#include "hbrt4-c/Detail/Version.h"

// Must be included at last to cleanup macros
#include "hbrt4-c/Detail/UndefineMacros/CompilerUndefineMacros.h"

// IWYU pragma: end_exports

#undef HBRT4_DETAIL_GUARD
