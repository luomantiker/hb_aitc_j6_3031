#include "ude/public/Library.h"
#include "ude/public/Protocols.h"
#include "ude/public/Status.h"
#include "ude/public/Types.h"

namespace user {
ude::Status MatMulInfer(
    ude::TensorRef& out,
    const ude::TensorRef& lhs,
    const ude::TensorRef& rhs) {
  // only support rank 2 TensorType
  if (lhs.rank != 2 || rhs.rank != 2) {
    return ude::Status::failure();
  }
  // lhs must have same element type with rhs
  if (lhs.dtype != rhs.dtype) {
    return ude::Status::failure();
  }
  // lhs dim 1 must equals rhs dim 0
  if (lhs.shape[1] != rhs.shape[0]) {
    return ude::Status::failure();
  }
  out.shape = {lhs.shape[0], rhs.shape[1]};
  out.dtype = lhs.dtype;
  return ude::Status::success();
}

template <typename T>
void MatMulImplTemplate(
    ude::TensorRef& out,
    const ude::TensorRef& lhs,
    const ude::TensorRef& rhs) {
  auto outptr = static_cast<T*>(out.ptr);
  auto lhsptr = static_cast<T*>(lhs.ptr);
  auto rhsptr = static_cast<T*>(rhs.ptr);

  auto M = lhs.shape[0];
  auto K = lhs.shape[1];
  auto N = rhs.shape[1];

  auto elt_size = ude::getByteSize(lhs.dtype);
  auto lhs_stride_0 = lhs.stride[0] / elt_size;
  auto lhs_stride_1 = lhs.stride[1] / elt_size;
  auto rhs_stride_0 = rhs.stride[0] / elt_size;
  auto rhs_stride_1 = rhs.stride[1] / elt_size;
  auto out_stride_0 = out.stride[0] / elt_size;
  auto out_stride_1 = out.stride[1] / elt_size;

  for (auto m = 0U; m < M; ++m) {
    for (auto n = 0U; n < N; ++n) {
      *(outptr + (m * out_stride_0) + (n * out_stride_1)) = 0;
    }
  }

  for (auto m = 0U; m < M; ++m) {
    for (auto n = 0U; n < N; ++n) {
      for (auto k = 0U; k < K; ++k) {
        *(outptr + (m * out_stride_0) + (n * out_stride_1)) +=
            *(lhsptr + (m * lhs_stride_0) + (k * lhs_stride_1)) *
            *(rhsptr + (k * rhs_stride_0) + (n * rhs_stride_1));
      }
    }
  }
}

ude::Status MatMulImpl(
    ude::TensorRef& out,
    const ude::TensorRef& lhs,
    const ude::TensorRef& rhs) {
  if (lhs.dtype == ude::Type::f32) {
    MatMulImplTemplate<float>(out, lhs, rhs);
  } else if (lhs.dtype == ude::Type::f64) {
    MatMulImplTemplate<double>(out, lhs, rhs);
  } else if (lhs.dtype == ude::Type::f64) {
    MatMulImplTemplate<double>(out, lhs, rhs);
  } else {
    return ude::Status::failure(true, "Unsupported dtype");
  }
  return ude::Status::success();
}
} // namespace user

UDE_LIBRARY(user, CUSTOM) {
  m.def<1>("user::MatMul", user::MatMulInfer, user::MatMulImpl);
}
