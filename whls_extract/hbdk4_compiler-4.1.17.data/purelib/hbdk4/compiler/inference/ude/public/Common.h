#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#define UDE_MAJOR_VERSION 1U
#define UDE_MINOR_VERSION 0U
#define MIN_UDE_MAJOR_VERSION 1U
#define MIN_UDE_MINOR_VERSION 0U

/// Version as a single 2-bytes hex number
/// e.g. 0x0101 -> 1.1
#define UDE_VERSION_HEX(MAJOR, MINOR) ((MAJOR << 8U) | (MINOR << 0U))
#define UDE_MIN_REQUIRED UDE_VERSION_HEX(MIN_UDE_MAJOR_VERSION, MIN_UDE_MINOR_VERSION)

namespace ude {

template <typename T> struct negation : std::integral_constant<bool, !T::value> {};

template <class... Ts> using all_type_of = std::conjunction<Ts...>;
template <class... Ts> using any_type_of = std::disjunction<Ts...>;
template <typename T, typename U> inline constexpr bool is_same_v = std::is_same_v<T, U>;

template <typename T> inline constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

template <typename T> inline constexpr bool is_integral_v = std::is_integral_v<T>;

template <typename T> inline constexpr bool is_signed_v = std::is_signed_v<T>;

template <class... Ts> using none_of = negation<any_type_of<Ts...>>;

template <typename T, typename U> using is_different = negation<std::is_same<T, U>>;

template <typename T> using is_not_pointer = negation<std::is_pointer<T>>;

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type {
  using type = T;
};
template <typename T> struct intrinsic_type<const T> {
  using type = typename intrinsic_type<T>::type;
};
template <typename T> struct intrinsic_type<T *> {
  using type = typename intrinsic_type<T>::type;
};
template <typename T> struct intrinsic_type<T &> {
  using type = typename intrinsic_type<T>::type;
};
template <typename T> struct intrinsic_type<T &&> {
  using type = typename intrinsic_type<T>::type;
};
template <typename T, size_t N> struct intrinsic_type<const T[N]> { // NOLINT
  using type = typename intrinsic_type<T>::type;
};
template <typename T, size_t N> struct intrinsic_type<T[N]> { // NOLINT
  using type = typename intrinsic_type<T>::type;
};

// template type
// use this make intrinsic_type more clear.
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

/// Support some c++17 template using
template <typename T> using add_pointer_t = typename std::add_pointer<T>::type;

template <typename T> using add_lvalue_reference_t = typename std::add_lvalue_reference<T>::type;

template <typename T> using add_rvalue_reference_t = typename std::add_rvalue_reference<T>::type;

template <typename T> inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

template <typename T>
constexpr bool is_function_v = std::is_function<typename std::remove_pointer<std::remove_reference_t<T>>::type>::value;

template <size_t N, typename T> struct offset_sequence;

template <size_t N, size_t... Ints> struct offset_sequence<N, std::index_sequence<Ints...>> {
  using type = std::index_sequence<(N + Ints)...>;
};

template <class T, class Enable = void> struct has_msg_trait : std::false_type {};

template <class T> struct has_msg_trait<T, std::void_t<decltype(std::declval<T &>().getMsg())>> : std::true_type {};

template <class T> constexpr bool has_msg_trait_v = has_msg_trait<T>::value;

/// Marco for converting params to string and concat A and B to AB
#define UDE_STRING_(name) #name
#define UDE_STRING(name) UDE_STRING_(name)
#define UDE_CONCAT_(A, B) A##B
#define UDE_CONCAT(A, B) UDE_CONCAT_(A, B)

/// Dispatcher key
enum class DispatchKey : size_t { REFERENCE = 0, EXTENSION = 1, CUSTOM = 2, HBDNN = 3, UCP = 4, UNKNOWN = 5 };

/// Helper class for define extra attributes of Kernel
/// For example, you can use it as below:
/// m.def<2>("add", addImpl, ude::doc("add kernel"), ude::arg("parama"), ude::arg("paramb"))
struct arg {
  const char *value;
  explicit arg(const char *value) : value(value) {}
};

struct doc {
  const char *value;
  explicit doc(const char *value) : value(value) {}
};

struct name {
  const char *value;
  explicit name(const char *value) : value(value) {}
};

struct backend {
  const char *value;
  explicit backend(const char *value) : value(value) {}
};

struct file {
  const char *value;
  explicit file(const char *value) : value(value) {}
};

struct line {
  const size_t value;
  explicit line(const size_t &value) : value(value) {}
};

} // namespace ude
