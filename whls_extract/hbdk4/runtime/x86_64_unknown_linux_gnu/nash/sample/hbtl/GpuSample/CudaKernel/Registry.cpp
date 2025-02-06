#include "Kernel.h"
#include "hbtl/HbtlAdaptor.h" // IWYU pragma: keep
#include "hbtl/StlAdaptor.h"  // IWYU pragma: keep
#include "ude/public/Common.h"
#include "ude/public/Library.h"

namespace ude {

// NOLINTNEXTLINE
UDE_LIBRARY(Gpu, UCP) { m.def<1>("native::Sigmoid", hbtl::Sigmoid); }

} // namespace ude
