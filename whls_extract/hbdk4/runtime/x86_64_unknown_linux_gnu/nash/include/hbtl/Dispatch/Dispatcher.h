#pragma once
#include "hbtl/Support/Compiler.h"
#include <ude/internal/Dispatcher.h>

namespace hbtl {

class HBTL_EXPORTED Dispatcher {
public:
  static ude::Dispatcher *singleton();

  Dispatcher(const Dispatcher &) = delete;
  Dispatcher &operator=(const Dispatcher &) = delete;
  Dispatcher() = delete;

private:
  ude::Dispatcher disp;
};

} // namespace hbtl
