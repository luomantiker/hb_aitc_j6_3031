#include "hbtl/Core/Tensor.h"
#include "hbtl/Core/TensorType.h"
#include "hbtl/HbtlAdaptor.h"
#include "hbtl/Native.h"
#include "ude/public/Common.h"
#include "ude/public/Library.h"

namespace ude {

Status config(hbtl::TensorType &outs, hbtl::TensorType ins) {
  outs = ins;
  return Status::success();
}

// NOLINTNEXTLINE
UDE_LIBRARY(testAlpha, REFERENCE) {
  m.def<1>("native::sigmoid", hbtl::native::Sigmoid, ude::doc("Sigmoid function called at cpu"), ude::arg("out"),
           ude::arg("input"), ude::backend("cpu"));
  m.def<1>("native::add", config, hbtl::native::Add, ude::doc("Add function called at cpu"), ude::backend("cpu"));
}

} // namespace ude
