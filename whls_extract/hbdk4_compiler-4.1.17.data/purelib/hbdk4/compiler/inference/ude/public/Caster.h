#pragma once

#include "ArrayRef.h"
#include "Common.h"
#include "Status.h"
#include "Types.h"
#include "Variable.h"
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace ude {

/// This requires type caster to provide operator T&() and operator T*().
/// If T required not pointer type, this first one will be called.
/// If T required pointer type, this last one will be called.
template <typename T>
using cast_op_type = std::conditional_t<is_pointer_v<std::remove_reference_t<T>>, intrinsic_t<T>,
                                        add_lvalue_reference_t<intrinsic_t<T>>>;

/// This requires type caster to provide operator T&() 、operator T&&() && and operator T*().
/// If T required not pointer and not move type, this first one will be called.
/// If T required move type, this second one will be called.
/// If T required pointer type, this last one will be called.
/// Use the marco 'UDE_TYPE_CASTER' will default use this mode.
template <typename T>
using movable_cast_op_type =
    std::conditional_t<is_pointer_v<std::remove_reference_t<T>>, intrinsic_t<T>,
                       std::conditional_t<std::is_rvalue_reference<T>::value, add_rvalue_reference_t<intrinsic_t<T>>,
                                          add_lvalue_reference_t<intrinsic_t<T>>>>;

/// We will only need load and sync function, the convert logical is very simple.
/// This class is base of type caster for Opaque type.
/// It treats all type as void*.
/// For custom type caster, you need to define a suitable type caster.
template <typename T> class TypeCasterBase {
public:
  using ThisT = std::remove_pointer_t<intrinsic_t<T>>;

  /// For Opaque type, we use ptr as it's typeId
  static constexpr auto typeId = Type::ptr;

  template <typename CT = T, std::enable_if_t<is_pointer_v<intrinsic_t<CT>>, bool> = true>
  TypeCasterBase() : type(getType<Pointer>()) {}

  template <typename CT = T, std::enable_if_t<!is_pointer_v<intrinsic_t<CT>>, bool> = true>
  TypeCasterBase() : type(getType<intrinsic_t<T>>()) {}

  Status load(Variable *var, size_t idx = 0) {
    if (var->getType() != type) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(getType<ThisT>()) + ", but got a type " +
                           toString(var->getType());
      return Status::failure(true, errMsg.c_str());
    }

    value = static_cast<void *>(&var->getRef<T>());

    return Status::success();
  }

  /// Base keeps pointer of variable, so sync always return true.
  Status sync(Variable *var) { return Status::success(); }

  // Determine which convert function should call.
  template <typename type> using castOpType = cast_op_type<type>;

  // NOLINTNEXTLINE
  operator ThisT *() { return (ThisT *)value; }
  // NOLINTNEXTLINE
  operator ThisT &() { return *((ThisT *)value); }

protected:
  Type type = Type::invalid;
  void *value = nullptr;
};

// Why use two template parameters?
// You can only define a custom TypeCaster for a specify type's converting,
// not relying inherit the TypeCasterBase.
template <typename T, typename SFINAE = void> class TypeCaster : public TypeCasterBase<T> {};
template <typename T> using MakeCaster = TypeCaster<intrinsic_t<T>>;

template <typename T> typename MakeCaster<T>::template castOpType<T> castOp(MakeCaster<T> &caster) {
  return caster.operator typename MakeCaster<T>::template castOpType<T>();
}
template <typename T>
typename MakeCaster<T>::template castOpType<add_rvalue_reference_t<T>> castOp(MakeCaster<T> &&caster) {
  return std::move(caster).operator typename MakeCaster<T>::template castOpType<add_rvalue_reference_t<T>>();
}

#define UDE_TYPE_CASTER(type, id)                                                                                      \
protected:                                                                                                             \
  type value;                                                                                                          \
                                                                                                                       \
public:                                                                                                                \
  static constexpr auto typeId = id;                                                                                   \
  operator type *() { return &value; }               /* NOLINT(bugprone-macro-parentheses) */                          \
  operator type &() { return value; }                /* NOLINT(bugprone-macro-parentheses) */                          \
  operator type &&() && { return std::move(value); } /*NOLINT(bugprone-macro-parentheses) */                           \
  template <typename T_> using castOpType = movable_cast_op_type<T_>;

template <typename T> struct VariableTypeId {
  static constexpr auto typeId = Type::invalid;
};

#define VariableTypeIdSpec(type, id)                                                                                   \
  template <> struct VariableTypeId<type> {                                                                            \
    static constexpr auto typeId = Type::id;                                                                           \
  }

VariableTypeIdSpec(bool, bool8);
VariableTypeIdSpec(int8_t, si8);
VariableTypeIdSpec(int16_t, si16);
VariableTypeIdSpec(int32_t, si32);
VariableTypeIdSpec(int64_t, si64);
VariableTypeIdSpec(uint8_t, ui8);
VariableTypeIdSpec(uint16_t, ui16);
VariableTypeIdSpec(uint32_t, ui32);
VariableTypeIdSpec(uint64_t, ui64);
VariableTypeIdSpec(float, f32);
VariableTypeIdSpec(double, f64);

/// For inner type, we use the default caster will work.
template <typename T> class TypeCaster<T, std::enable_if_t<is_arithmetic_v<intrinsic_t<T>>>> {
public:
  using iType = intrinsic_t<T>;

  Status load(Variable *src, size_t idx = 0) {
    if (src == nullptr || (src->getType() != ude::getType<iType>() && src->getType() != getType<Pointer>())) {
      // todo(weiguozhao02): make error reporter display better.
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(getType<iType>()) + ", but got a type " +
                           toString(src->getType());
      return Status::failure(true, errMsg.c_str());
    }
    value = src->getRef<iType>();
    return Status::success();
  }

  Status sync(Variable *src) {
    src->getRef<iType>() = value;
    return Status::success();
  }

  UDE_TYPE_CASTER(iType, VariableTypeId<iType>::typeId)
};

template <> class TypeCaster<TensorRef> {
public:
  Status load(Variable *src, size_t idx = 0) {
    if (src == nullptr || (src->getType() != ude::Type::tensor)) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(Type::tensor) + ", but got a type " +
                           toString(src->getType());
      return Status::failure(true, errMsg.c_str());
    }
    value = src->getRef<TensorRef>();
    return Status::success();
  }

  Status sync(Variable *src) {
    src->getRef<TensorRef>() = value;
    return Status::success();
  }

  UDE_TYPE_CASTER(TensorRef, ude::Type::tensor)
};

/// ArgLoader class
/// The load method is the place where convert logical happened.
/// The call method is the place where function call logical happened.
template <size_t outNum, typename... Args> class ArgLoader {
  using indices = std::make_index_sequence<sizeof...(Args)>;
  using CastType = std::tuple<MakeCaster<Args>...>;
  using outInd = std::make_index_sequence<outNum>;
  using inInd = std::make_index_sequence<sizeof...(Args) - outNum>;

public:
  template <size_t... Is> constexpr static std::array<Type, 5> genOutIds(std::index_sequence<Is...> /*unused*/) {
    return std::array<Type, 5>{std::tuple_element_t<Is, CastType>::typeId...};
  }

  template <size_t... Is> constexpr static std::array<Type, 40> genInIds(std::index_sequence<Is...> /*unused*/) {
    return std::array<Type, 40>{std::tuple_element_t<Is, CastType>::typeId...};
  }

  Status loadInArgs(ArrayRef<Variable *> args) {
    assert(args.size() == sizeof...(Args) - outNum && "Number of Input Args should be equal to function's definition");
    return loadImplSequence<outNum>(args, inInd{});
  }

  Status loadOutArgs(ArrayRef<Variable *> args) {
    assert(args.size() == outNum && "Number of Output Args should be equal to function's definition");
    return loadImplSequence<0>(args, outInd{});
  }

  Status syncOutArgs(ArrayRef<Variable *> args) {
    assert(args.size() == outNum && "Number of Output Args should be equal to function's definition");
    return syncImplSequence(args, outInd{});
  }

  template <typename Ret, typename Func> Ret call(Func &&f) && {
    return std::move(*this).template callImpl<std::remove_cv_t<Ret>>(std::forward<Func>(f), indices{});
  }

private:
  template <size_t offset> static Status loadImplSequence(ArrayRef<Variable *> args, std::index_sequence<> /*unused*/) {
    return Status::success();
  }

  template <size_t offset, size_t... Is>
  Status loadImplSequence(ArrayRef<Variable *> args, std::index_sequence<Is...> /*unused*/) {
    for (Status r : {std::get<Is + offset>(argCasters).load(args[Is], Is)...}) { // NOLINT
      if (r.failed()) {
        return r;
      }
    }
    return Status::success();
  }

  static Status syncImplSequence(ArrayRef<Variable *> args, std::index_sequence<> /*unused*/) {
    return Status::success();
  }

  template <size_t... Is> Status syncImplSequence(ArrayRef<Variable *> args, std::index_sequence<Is...> /*unused*/) {
    for (Status r : {std::get<Is>(argCasters).sync(args[Is])...}) {
      if (r.failed()) {
        return r;
      }
    }
    return Status::success();
  }

  template <typename Ret, typename Func, size_t... Is> Ret callImpl(Func &&f, std::index_sequence<Is...> /*unused*/) {
    return std::forward<Func>(f)(castOp<Args>(std::move(std::get<Is>(argCasters)))...);
  }

  CastType argCasters;
};

} // namespace ude
