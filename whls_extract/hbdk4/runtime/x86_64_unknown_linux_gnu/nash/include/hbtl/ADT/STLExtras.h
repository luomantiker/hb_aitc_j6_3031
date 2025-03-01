//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains some templates that are useful if you are working with
/// the STL at all.
///
/// No library is required when using these functions.
///
//===----------------------------------------------------------------------===//

/// forked from llvm/ADT/STLExtras.h
#pragma once

// #include "llvm/ADT/Hashing.h"
#include "hbtl/ADT/Optional.h"
#include "hbtl/ADT/STLForwardCompat.h"
#include "hbtl/ADT/STLFunctionalExtras.h"
#include "hbtl/ADT/identity.h"
#include "hbtl/ADT/iterator.h"
#include "hbtl/ADT/iterator_range.h"
#include "hbtl/ADT/string_view.h"
#include "hbtl/Support/Compiler.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

HBTL_NAMESPACE_BEGIN {

  template <class...> struct disjunction : std::false_type {};
  template <class B1> struct disjunction<B1> : B1 {};
  template <class B1, class... Bn>
  struct disjunction<B1, Bn...> : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>> {};

  // Only used by compiler if both template types are the same.  Useful when
  // using SFINAE to test for the existence of member functions.
  template <typename T, T> struct SameType;

  namespace adl_detail {

  using std::begin;

  template <typename RangeT> constexpr auto begin_impl(RangeT &&range) -> decltype(begin(std::forward<RangeT>(range))) {
    return begin(std::forward<RangeT>(range));
  }

  using std::end;

  template <typename RangeT> constexpr auto end_impl(RangeT &&range) -> decltype(end(std::forward<RangeT>(range))) {
    return end(std::forward<RangeT>(range));
  }

  using std::swap;

  template <typename T>
  constexpr void swap_impl(T &&lhs, T &&rhs) noexcept(noexcept(swap(std::declval<T>(), std::declval<T>()))) {
    swap(std::forward<T>(lhs), std::forward<T>(rhs));
  }

  } // end namespace adl_detail

  /// Returns the begin iterator to \p range using `std::begin` and
  /// function found through Argument-Dependent Lookup (ADL).
  template <typename RangeT>
  constexpr auto adl_begin(RangeT && range)->decltype(adl_detail::begin_impl(std::forward<RangeT>(range))) {
    return adl_detail::begin_impl(std::forward<RangeT>(range));
  }

  /// Returns the end iterator to \p range using `std::end` and
  /// functions found through Argument-Dependent Lookup (ADL).
  template <typename RangeT>
  constexpr auto adl_end(RangeT && range)->decltype(adl_detail::end_impl(std::forward<RangeT>(range))) {
    return adl_detail::end_impl(std::forward<RangeT>(range));
  }

  /// Swaps \p lhs with \p rhs using `std::swap` and functions found through
  /// Argument-Dependent Lookup (ADL).
  template <typename T>
  constexpr void adl_swap(T && lhs,
                          T && rhs) noexcept(noexcept(adl_detail::swap_impl(std::declval<T>(), std::declval<T>()))) {
    adl_detail::swap_impl(std::forward<T>(lhs), std::forward<T>(rhs));
  }

  namespace detail {

  template <typename RangeT> using IterOfRange = decltype(hbtl::adl_begin(std::declval<RangeT &>()));

  template <typename RangeT>
  using ValueOfRange = std::remove_reference_t<decltype(*hbtl::adl_begin(std::declval<RangeT &>()))>;

  } // end namespace detail

  //===----------------------------------------------------------------------===//
  //     Extra additions to <type_traits>
  //===----------------------------------------------------------------------===//

  template <typename T> struct make_const_ptr {
    using type = std::add_pointer_t<std::add_const_t<T>>;
  };

  template <typename T> struct make_const_ref {
    using type = std::add_lvalue_reference_t<std::add_const_t<T>>;
  };

  namespace detail {
  template <class, template <class...> class Op, class... Args> struct detector {
    using value_t = std::false_type;
  };
  template <class...> using void_t = void;
  template <template <class...> class Op, class... Args> struct detector<void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
  };
  } // end namespace detail

  /// Detects if a given trait holds for some set of arguments 'Args'.
  /// For example, the given trait could be used to detect if a given type
  /// has a copy assignment operator:
  ///   template<class T>
  ///   using has_copy_assign_t = decltype(std::declval<T&>()
  ///                                                 = std::declval<const T&>());
  ///   bool fooHasCopyAssign = is_detected<has_copy_assign_t, FooClass>::value;
  template <template <class...> class Op, class... Args>
  using is_detected = typename detail::detector<void, Op, Args...>::value_t;

  /// This class provides various trait information about a callable object.
  ///   * To access the number of arguments: Traits::num_args
  ///   * To access the type of an argument: Traits::arg_t<Index>
  ///   * To access the type of the result:  Traits::result_t
  template <typename T, bool isClass = std::is_class<T>::value>
  struct function_traits : public function_traits<decltype(&T::operator())> {};

  /// Overload for class function types.
  template <typename ClassType, typename ReturnType, typename... Args>
  struct function_traits<ReturnType (ClassType::*)(Args...) const, false> {
    /// The number of arguments to this function.
    enum { num_args = sizeof...(Args) };

    /// The result type of this function.
    using result_t = ReturnType;

    /// The type of an argument to this function.
    template <size_t Index> using arg_t = std::tuple_element_t<Index, std::tuple<Args...>>;
  };
  /// Overload for class function types.
  template <typename ClassType, typename ReturnType, typename... Args>
  struct function_traits<ReturnType (ClassType::*)(Args...), false>
      : public function_traits<ReturnType (ClassType::*)(Args...) const> {};
  /// Overload for non-class function types.
  template <typename ReturnType, typename... Args> struct function_traits<ReturnType (*)(Args...), false> {
    /// The number of arguments to this function.
    enum { num_args = sizeof...(Args) };

    /// The result type of this function.
    using result_t = ReturnType;

    /// The type of an argument to this function.
    template <size_t i> using arg_t = std::tuple_element_t<i, std::tuple<Args...>>;
  };
  template <typename ReturnType, typename... Args>
  struct function_traits<ReturnType (*const)(Args...), false> : public function_traits<ReturnType (*)(Args...)> {};
  /// Overload for non-class function type references.
  template <typename ReturnType, typename... Args>
  struct function_traits<ReturnType (&)(Args...), false> : public function_traits<ReturnType (*)(Args...)> {};

  /// traits class for checking whether type T is one of any of the given
  /// types in the variadic list.
  template <typename T, typename... Ts> using is_one_of = disjunction<std::is_same<T, Ts>...>;

  /// traits class for checking whether type T is a base class for all
  ///  the given types in the variadic list.
  template <typename T, typename... Ts> using are_base_of = detail::conjunction<std::is_base_of<T, Ts>...>;

  namespace detail {
  template <typename T, typename... Us> struct TypesAreDistinct;
  template <typename T, typename... Us>
  struct TypesAreDistinct
      : std::integral_constant<bool, !is_one_of<T, Us...>::value && TypesAreDistinct<Us...>::value> {};
  template <typename T> struct TypesAreDistinct<T> : std::true_type {};
  } // namespace detail

  /// Determine if all types in Ts are distinct.
  ///
  /// Useful to statically assert when Ts is intended to describe a non-multi set
  /// of types.
  ///
  /// Expensive (currently quadratic in sizeof(Ts...)), and so should only be
  /// asserted once per instantiation of a type which requires it.
  template <typename... Ts> struct TypesAreDistinct;
  template <> struct TypesAreDistinct<> : std::true_type {};
  template <typename... Ts>
  struct TypesAreDistinct : std::integral_constant<bool, detail::TypesAreDistinct<Ts...>::value> {};

  /// Find the first index where a type appears in a list of types.
  ///
  /// FirstIndexOfType<T, Us...>::value is the first index of T in Us.
  ///
  /// Typically only meaningful when it is otherwise statically known that the
  /// type pack has no duplicate types. This should be guaranteed explicitly with
  /// static_assert(TypesAreDistinct<Us...>::value).
  ///
  /// It is a compile-time error to instantiate when T is not present in Us, i.e.
  /// if is_one_of<T, Us...>::value is false.
  template <typename T, typename... Us> struct FirstIndexOfType;
  template <typename T, typename U, typename... Us>
  struct FirstIndexOfType<T, U, Us...> : std::integral_constant<size_t, 1 + FirstIndexOfType<T, Us...>::value> {};
  template <typename T, typename... Us> struct FirstIndexOfType<T, T, Us...> : std::integral_constant<size_t, 0> {};

  /// Find the type at a given index in a list of types.
  ///
  /// TypeAtIndex<I, Ts...> is the type at index I in Ts.
  template <size_t I, typename... Ts> using TypeAtIndex = std::tuple_element_t<I, std::tuple<Ts...>>;

  /// Helper which adds two underlying types of enumeration type.
  /// Implicit conversion to a common type is accepted.
  template <typename EnumTy1, typename EnumTy2,
            typename UT1 = std::enable_if_t<std::is_enum<EnumTy1>::value, std::underlying_type_t<EnumTy1>>,
            typename UT2 = std::enable_if_t<std::is_enum<EnumTy2>::value, std::underlying_type_t<EnumTy2>>>
  constexpr auto addEnumValues(EnumTy1 LHS, EnumTy2 RHS) {
    return static_cast<UT1>(LHS) + static_cast<UT2>(RHS);
  }

  //===----------------------------------------------------------------------===//
  //     Extra additions to <iterator>
  //===----------------------------------------------------------------------===//

  namespace callable_detail {

  /// Templated storage wrapper for a callable.
  ///
  /// This class is consistently default constructible, copy / move
  /// constructible / assignable.
  ///
  /// Supported callable types:
  ///  - Function pointer
  ///  - Function reference
  ///  - Lambda
  ///  - Function object
  template <typename T, bool = std::is_function<std::remove_pointer_t<remove_cvref_t<T>>>::value> class Callable {
    using value_type = std::remove_reference_t<T>;
    using reference = value_type &;
    using const_reference = value_type const &;

    hbtl::optional<value_type> Obj;

    static_assert(!std::is_pointer<value_type>::value, "Pointers to non-functions are not callable.");

  public:
    Callable() = default;
    Callable(T const &O) : Obj(in_place, O) {}

    Callable(Callable const &Other) = default;
    Callable(Callable &&Other) = default;

    Callable &operator=(Callable const &Other) {
      Obj = hbtl::nullopt;
      if (Other.Obj)
        Obj.emplace(*Other.Obj);
      return *this;
    }

    Callable &operator=(Callable &&Other) {
      Obj = hbtl::nullopt;
      if (Other.Obj)
        Obj.emplace(std::move(*Other.Obj));
      return *this;
    }

    template <typename... Pn, std::enable_if_t<hbtl::__is_invocable<T, Pn...>::value, int> = 0>
    decltype(auto) operator()(Pn &&...Params) {
      return (*Obj)(std::forward<Pn>(Params)...);
    }

    template <typename... Pn, std::enable_if_t<hbtl::__is_invocable<T const, Pn...>::value, int> = 0>
    decltype(auto) operator()(Pn &&...Params) const {
      return (*Obj)(std::forward<Pn>(Params)...);
    }

    bool valid() const { return Obj != hbtl::nullopt; }
    bool reset() { return Obj = hbtl::nullopt; }

    operator reference() { return *Obj; }
    operator const_reference() const { return *Obj; }
  };

  // Function specialization.  No need to waste extra space wrapping with a
  // hbtl::optional.
  template <typename T> class Callable<T, true> {
    static constexpr bool IsPtr = std::is_pointer<remove_cvref_t<T>>::value;

    using StorageT = std::conditional_t<IsPtr, T, std::remove_reference_t<T> *>;
    using CastT = std::conditional_t<IsPtr, T, T &>;

  private:
    StorageT Func = nullptr;

  private:
    template <typename In> static constexpr auto convertIn(In &&I) {
      if HBTL_CONSTEXPR_IF (IsPtr) {
        // Pointer... just echo it back.
        return I;
      } else {
        // Must be a function reference.  Return its address.
        return &I;
      }
    }

  public:
    Callable() = default;

    // Construct from a function pointer or reference.
    //
    // Disable this constructor for references to 'Callable' so we don't violate
    // the rule of 0.
    template < // clang-format off
    typename FnPtrOrRef,
    std::enable_if_t<
      !std::is_same<remove_cvref_t<FnPtrOrRef>, Callable>::value, int
    > = 0
  > // clang-format on
    Callable(FnPtrOrRef &&F) : Func(convertIn(F)) {}

    template <typename... Pn, std::enable_if_t<hbtl::__is_invocable<T, Pn...>::value, int> = 0>
    decltype(auto) operator()(Pn &&...Params) const {
      return Func(std::forward<Pn>(Params)...);
    }

    bool valid() const { return Func != nullptr; }
    void reset() { Func = nullptr; }

    operator T const &() const {
      if HBTL_CONSTEXPR_IF (IsPtr) {
        // T is a pointer... just echo it back.
        return Func;
      } else {
        static_assert(std::is_reference<T>::value, "Expected a reference to a function.");
        // T is a function reference... dereference the stored pointer.
        return *Func;
      }
    }
  };

  } // namespace callable_detail

  /// Returns true if the given container only contains a single element.
  template <typename ContainerTy> bool hasSingleElement(ContainerTy && C) {
    auto B = std::begin(C), E = std::end(C);
    return B != E && std::next(B) == E;
  }

  /// Return a range covering \p RangeOrContainer with the first N elements
  /// excluded.
  template <typename T> auto drop_begin(T && RangeOrContainer, size_t N = 1) {
    return make_range(std::next(hbtl::adl_begin(RangeOrContainer), N), hbtl::adl_end(RangeOrContainer));
  }

  /// Return a range covering \p RangeOrContainer with the last N elements
  /// excluded.
  template <typename T> auto drop_end(T && RangeOrContainer, size_t N = 1) {
    return make_range(hbtl::adl_begin(RangeOrContainer), std::prev(hbtl::adl_end(RangeOrContainer), N));
  }

  // mapped_iterator - This is a simple iterator adapter that causes a function to
  // be applied whenever operator* is invoked on the iterator.

  template <typename ItTy, typename FuncTy,
            typename ReferenceTy = decltype(std::declval<FuncTy>()(*std::declval<ItTy>()))>
  class mapped_iterator
      : public iterator_adaptor_base<
            mapped_iterator<ItTy, FuncTy>, ItTy, typename std::iterator_traits<ItTy>::iterator_category,
            std::remove_reference_t<ReferenceTy>, typename std::iterator_traits<ItTy>::difference_type,
            std::remove_reference_t<ReferenceTy> *, ReferenceTy> {
  public:
    mapped_iterator() = default;
    mapped_iterator(ItTy U, FuncTy F) : mapped_iterator::iterator_adaptor_base(std::move(U)), F(std::move(F)) {}

    ItTy getCurrent() { return this->I; }

    const FuncTy &getFunction() const { return F; }

    ReferenceTy operator*() const { return F(*this->I); }

  private:
    callable_detail::Callable<FuncTy> F{};
  };

  // map_iterator - Provide a convenient way to create mapped_iterators, just like
  // make_pair is useful for creating pairs...
  template <class ItTy, class FuncTy> inline mapped_iterator<ItTy, FuncTy> map_iterator(ItTy I, FuncTy F) {
    return mapped_iterator<ItTy, FuncTy>(std::move(I), std::move(F));
  }

  template <class ContainerTy, class FuncTy> auto map_range(ContainerTy && C, FuncTy F) {
    return make_range(map_iterator(C.begin(), F), map_iterator(C.end(), F));
  }

  /// A base type of mapped iterator, that is useful for building derived
  /// iterators that do not need/want to store the map function (as in
  /// mapped_iterator). These iterators must simply provide a `mapElement` method
  /// that defines how to map a value of the iterator to the provided reference
  /// type.
  template <typename DerivedT, typename ItTy, typename ReferenceTy>
  class mapped_iterator_base
      : public iterator_adaptor_base<DerivedT, ItTy, typename std::iterator_traits<ItTy>::iterator_category,
                                     std::remove_reference_t<ReferenceTy>,
                                     typename std::iterator_traits<ItTy>::difference_type,
                                     std::remove_reference_t<ReferenceTy> *, ReferenceTy> {
  public:
    using BaseT = mapped_iterator_base;

    mapped_iterator_base(ItTy U) : mapped_iterator_base::iterator_adaptor_base(std::move(U)) {}

    ItTy getCurrent() { return this->I; }

    ReferenceTy operator*() const { return static_cast<const DerivedT &>(*this).mapElement(*this->I); }
  };

  /// Helper to determine if type T has a member called rbegin().
  template <typename Ty> class has_rbegin_impl {
    using yes = char[1];
    using no = char[2];

    template <typename Inner> static yes &test(Inner *I, decltype(I->rbegin()) * = nullptr);

    template <typename> static no &test(...);

  public:
    static const bool value = sizeof(test<Ty>(nullptr)) == sizeof(yes);
  };

  /// Metafunction to determine if T& or T has a member called rbegin().
  template <typename Ty> struct has_rbegin : has_rbegin_impl<std::remove_reference_t<Ty>> {};

  // Returns an iterator_range over the given container which iterates in reverse.
  template <typename ContainerTy> auto reverse(ContainerTy && C) {
    if HBTL_CONSTEXPR_IF (has_rbegin<ContainerTy>::value)
      return make_range(C.rbegin(), C.rend());
    else
      return make_range(std::make_reverse_iterator(std::end(C)), std::make_reverse_iterator(std::begin(C)));
  }

  /// An iterator adaptor that filters the elements of given inner iterators.
  ///
  /// The predicate parameter should be a callable object that accepts the wrapped
  /// iterator's reference type and returns a bool. When incrementing or
  /// decrementing the iterator, it will call the predicate on each element and
  /// skip any where it returns false.
  ///
  /// \code
  ///   int A[] = { 1, 2, 3, 4 };
  ///   auto R = make_filter_range(A, [](int N) { return N % 2 == 1; });
  ///   // R contains { 1, 3 }.
  /// \endcode
  ///
  /// Note: filter_iterator_base implements support for forward iteration.
  /// filter_iterator_impl exists to provide support for bidirectional iteration,
  /// conditional on whether the wrapped iterator supports it.
  template <typename WrappedIteratorT, typename PredicateT, typename IterTag>
  class filter_iterator_base
      : public iterator_adaptor_base<
            filter_iterator_base<WrappedIteratorT, PredicateT, IterTag>, WrappedIteratorT,
            std::common_type_t<IterTag, typename std::iterator_traits<WrappedIteratorT>::iterator_category>> {
    using BaseT = typename filter_iterator_base::iterator_adaptor_base;

  protected:
    WrappedIteratorT End;
    PredicateT Pred;

    void findNextValid() {
      while (this->I != End && !Pred(*this->I))
        BaseT::operator++();
    }

    filter_iterator_base() = default;

    // Construct the iterator. The begin iterator needs to know where the end
    // is, so that it can properly stop when it gets there. The end iterator only
    // needs the predicate to support bidirectional iteration.
    filter_iterator_base(WrappedIteratorT Begin, WrappedIteratorT End, PredicateT Pred)
        : BaseT(Begin), End(End), Pred(Pred) {
      findNextValid();
    }

  public:
    using BaseT::operator++;

    filter_iterator_base &operator++() {
      BaseT::operator++();
      findNextValid();
      return *this;
    }

    decltype(auto) operator*() const {
      assert(BaseT::wrapped() != End && "Cannot dereference end iterator!");
      return BaseT::operator*();
    }

    decltype(auto) operator->() const {
      assert(BaseT::wrapped() != End && "Cannot dereference end iterator!");
      return BaseT::operator->();
    }
  };

  /// Specialization of filter_iterator_base for forward iteration only.
  template <typename WrappedIteratorT, typename PredicateT, typename IterTag = std::forward_iterator_tag>
  class filter_iterator_impl : public filter_iterator_base<WrappedIteratorT, PredicateT, IterTag> {
  public:
    filter_iterator_impl() = default;

    filter_iterator_impl(WrappedIteratorT Begin, WrappedIteratorT End, PredicateT Pred)
        : filter_iterator_impl::filter_iterator_base(Begin, End, Pred) {}
  };

  /// Specialization of filter_iterator_base for bidirectional iteration.
  template <typename WrappedIteratorT, typename PredicateT>
  class filter_iterator_impl<WrappedIteratorT, PredicateT, std::bidirectional_iterator_tag>
      : public filter_iterator_base<WrappedIteratorT, PredicateT, std::bidirectional_iterator_tag> {
    using BaseT = typename filter_iterator_impl::filter_iterator_base;

    void findPrevValid() {
      while (!this->Pred(*this->I))
        BaseT::operator--();
    }

  public:
    using BaseT::operator--;

    filter_iterator_impl() = default;

    filter_iterator_impl(WrappedIteratorT Begin, WrappedIteratorT End, PredicateT Pred) : BaseT(Begin, End, Pred) {}

    filter_iterator_impl &operator--() {
      BaseT::operator--();
      findPrevValid();
      return *this;
    }
  };

  namespace detail {

  template <bool is_bidirectional> struct fwd_or_bidi_tag_impl {
    using type = std::forward_iterator_tag;
  };

  template <> struct fwd_or_bidi_tag_impl<true> {
    using type = std::bidirectional_iterator_tag;
  };

  /// Helper which sets its type member to forward_iterator_tag if the category
  /// of \p IterT does not derive from bidirectional_iterator_tag, and to
  /// bidirectional_iterator_tag otherwise.
  template <typename IterT> struct fwd_or_bidi_tag {
    using type = typename fwd_or_bidi_tag_impl<std::is_base_of<
        std::bidirectional_iterator_tag, typename std::iterator_traits<IterT>::iterator_category>::value>::type;
  };

  } // namespace detail

  /// Defines filter_iterator to a suitable specialization of
  /// filter_iterator_impl, based on the underlying iterator's category.
  template <typename WrappedIteratorT, typename PredicateT>
  using filter_iterator =
      filter_iterator_impl<WrappedIteratorT, PredicateT, typename detail::fwd_or_bidi_tag<WrappedIteratorT>::type>;

  /// Convenience function that takes a range of elements and a predicate,
  /// and return a new filter_iterator range.
  ///
  /// FIXME: Currently if RangeT && is a rvalue reference to a temporary, the
  /// lifetime of that temporary is not kept by the returned range object, and the
  /// temporary is going to be dropped on the floor after the make_iterator_range
  /// full expression that contains this function call.
  template <typename RangeT, typename PredicateT>
  iterator_range<filter_iterator<detail::IterOfRange<RangeT>, PredicateT>> make_filter_range(RangeT && Range,
                                                                                             PredicateT Pred) {
    using FilterIteratorT = filter_iterator<detail::IterOfRange<RangeT>, PredicateT>;
    return make_range(
        FilterIteratorT(std::begin(std::forward<RangeT>(Range)), std::end(std::forward<RangeT>(Range)), Pred),
        FilterIteratorT(std::end(std::forward<RangeT>(Range)), std::end(std::forward<RangeT>(Range)), Pred));
  }

  /// A pseudo-iterator adaptor that is designed to implement "early increment"
  /// style loops.
  ///
  /// This is *not a normal iterator* and should almost never be used directly. It
  /// is intended primarily to be used with range based for loops and some range
  /// algorithms.
  ///
  /// The iterator isn't quite an `OutputIterator` or an `InputIterator` but
  /// somewhere between them. The constraints of these iterators are:
  ///
  /// - On construction or after being incremented, it is comparable and
  ///   dereferencable. It is *not* incrementable.
  /// - After being dereferenced, it is neither comparable nor dereferencable, it
  ///   is only incrementable.
  ///
  /// This means you can only dereference the iterator once, and you can only
  /// increment it once between dereferences.
  template <typename WrappedIteratorT>
  class early_inc_iterator_impl : public iterator_adaptor_base<early_inc_iterator_impl<WrappedIteratorT>,
                                                               WrappedIteratorT, std::input_iterator_tag> {
    using BaseT = typename early_inc_iterator_impl::iterator_adaptor_base;

    using PointerT = typename std::iterator_traits<WrappedIteratorT>::pointer;

  protected:
  public:
    early_inc_iterator_impl(WrappedIteratorT I) : BaseT(I) {}

    using BaseT::operator*;
    decltype(*std::declval<WrappedIteratorT>()) operator*() { return *(this->I)++; }

    using BaseT::operator++;
    early_inc_iterator_impl &operator++() { return *this; }

    friend bool operator==(const early_inc_iterator_impl &LHS, const early_inc_iterator_impl &RHS) {
      return (const BaseT &)LHS == (const BaseT &)RHS;
    }
  };

  /// Make a range that does early increment to allow mutation of the underlying
  /// range without disrupting iteration.
  ///
  /// The underlying iterator will be incremented immediately after it is
  /// dereferenced, allowing deletion of the current node or insertion of nodes to
  /// not disrupt iteration provided they do not invalidate the *next* iterator --
  /// the current iterator can be invalidated.
  ///
  /// This requires a very exact pattern of use that is only really suitable to
  /// range based for loops and other range algorithms that explicitly guarantee
  /// to dereference exactly once each element, and to increment exactly once each
  /// element.
  template <typename RangeT>
  iterator_range<early_inc_iterator_impl<detail::IterOfRange<RangeT>>> make_early_inc_range(RangeT && Range) {
    using EarlyIncIteratorT = early_inc_iterator_impl<detail::IterOfRange<RangeT>>;
    return make_range(EarlyIncIteratorT(std::begin(std::forward<RangeT>(Range))),
                      EarlyIncIteratorT(std::end(std::forward<RangeT>(Range))));
  }

  // Forward declarations required by zip_shortest/zip_equal/zip_first/zip_longest
  template <typename R, typename UnaryPredicate> bool all_of(R && range, UnaryPredicate P);

  template <typename R, typename UnaryPredicate> bool any_of(R && range, UnaryPredicate P);

  template <typename T> bool all_equal(std::initializer_list<T> Values);

  namespace detail {

  using std::declval;

  // We have to alias this since inlining the actual type at the usage site
  // in the parameter list of iterator_facade_base<> below ICEs MSVC 2017.
  template <typename... Iters> struct ZipTupleType {
    using type = std::tuple<decltype(*declval<Iters>())...>;
  };

  template <typename ZipType, typename... Iters>
  using zip_traits = iterator_facade_base<
      ZipType,
      std::common_type_t<std::bidirectional_iterator_tag, typename std::iterator_traits<Iters>::iterator_category...>,
      // ^ TODO: Implement random access methods.
      typename ZipTupleType<Iters...>::type,
      typename std::iterator_traits<std::tuple_element_t<0, std::tuple<Iters...>>>::difference_type,
      // ^ FIXME: This follows boost::make_zip_iterator's assumption that all
      // inner iterators have the same difference_type. It would fail if, for
      // instance, the second field's difference_type were non-numeric while the
      // first is.
      typename ZipTupleType<Iters...>::type *, typename ZipTupleType<Iters...>::type>;

  template <typename ZipType, typename... Iters> struct zip_common : public zip_traits<ZipType, Iters...> {
    using Base = zip_traits<ZipType, Iters...>;
    using IndexSequence = std::index_sequence_for<Iters...>;
    using value_type = typename Base::value_type;

    std::tuple<Iters...> iterators;

  protected:
    template <size_t... Ns> value_type deref(std::index_sequence<Ns...>) const {
      return value_type(*std::get<Ns>(iterators)...);
    }

    template <size_t... Ns> decltype(iterators) tup_inc(std::index_sequence<Ns...>) const {
      return std::tuple<Iters...>(std::next(std::get<Ns>(iterators))...);
    }

    template <size_t... Ns> decltype(iterators) tup_dec(std::index_sequence<Ns...>) const {
      return std::tuple<Iters...>(std::prev(std::get<Ns>(iterators))...);
    }

    template <size_t... Ns> bool test_all_equals(const zip_common &other, std::index_sequence<Ns...>) const {
      return all_of(std::initializer_list<bool>{std::get<Ns>(this->iterators) == std::get<Ns>(other.iterators)...},
                    identity<bool>{});
    }

  public:
    zip_common(Iters &&...ts) : iterators(std::forward<Iters>(ts)...) {}

    value_type operator*() const { return deref(IndexSequence{}); }

    ZipType &operator++() {
      iterators = tup_inc(std::index_sequence_for<Iters...>{});
      return *reinterpret_cast<ZipType *>(this);
    }

    ZipType &operator--() {
      static_assert(Base::IsBidirectional, "All inner iterators must be at least bidirectional.");
      iterators = tup_dec(std::index_sequence_for<Iters...>{});
      return *reinterpret_cast<ZipType *>(this);
    }

    /// Return true if all the iterator are matching `other`'s iterators.
    bool all_equals(zip_common &other) { return test_all_equals(other, IndexSequence{}); }
  };

  template <typename... Iters> struct zip_first : zip_common<zip_first<Iters...>, Iters...> {
    using zip_common<zip_first, Iters...>::zip_common;

    bool operator==(const zip_first &other) const {
      return std::get<0>(this->iterators) == std::get<0>(other.iterators);
    }
  };

  template <typename... Iters> struct zip_shortest : zip_common<zip_shortest<Iters...>, Iters...> {
    using zip_common<zip_shortest, Iters...>::zip_common;

    bool operator==(const zip_shortest &other) const {
      return any_iterator_equals(other, std::index_sequence_for<Iters...>{});
    }

  private:
    template <size_t... Ns> bool any_iterator_equals(const zip_shortest &other, std::index_sequence<Ns...>) const {
      return any_of(std::initializer_list<bool>{std::get<Ns>(this->iterators) == std::get<Ns>(other.iterators)...},
                    identity<bool>{});
    }
  };

  /// Helper to obtain the iterator types for the tuple storage within `zippy`.
  template <template <typename...> class ItType, typename TupleStorageType, typename IndexSequence>
  struct ZippyIteratorTuple;

  /// Partial specialization for non-const tuple storage.
  template <template <typename...> class ItType, typename... Args, std::size_t... Ns>
  struct ZippyIteratorTuple<ItType, std::tuple<Args...>, std::index_sequence<Ns...>> {
    using type = ItType<decltype(hbtl::adl_begin(std::get<Ns>(declval<std::tuple<Args...> &>())))...>;
  };

  /// Partial specialization for const tuple storage.
  template <template <typename...> class ItType, typename... Args, std::size_t... Ns>
  struct ZippyIteratorTuple<ItType, const std::tuple<Args...>, std::index_sequence<Ns...>> {
    using type = ItType<decltype(hbtl::adl_begin(std::get<Ns>(declval<const std::tuple<Args...> &>())))...>;
  };

  template <template <typename...> class ItType, typename... Args> class zippy {
  private:
    std::tuple<Args...> storage;
    using IndexSequence = std::index_sequence_for<Args...>;

  public:
    using iterator = typename ZippyIteratorTuple<ItType, decltype(storage), IndexSequence>::type;
    using const_iterator = typename ZippyIteratorTuple<ItType, const decltype(storage), IndexSequence>::type;
    using iterator_category = typename iterator::iterator_category;
    using value_type = typename iterator::value_type;
    using difference_type = typename iterator::difference_type;
    using pointer = typename iterator::pointer;
    using reference = typename iterator::reference;
    using const_reference = typename const_iterator::reference;

    zippy(Args &&...args) : storage(std::forward<Args>(args)...) {}

    const_iterator begin() const { return begin_impl(IndexSequence{}); }
    iterator begin() { return begin_impl(IndexSequence{}); }
    const_iterator end() const { return end_impl(IndexSequence{}); }
    iterator end() { return end_impl(IndexSequence{}); }

  private:
    template <size_t... Ns> const_iterator begin_impl(std::index_sequence<Ns...>) const {
      return const_iterator(hbtl::adl_begin(std::get<Ns>(storage))...);
    }
    template <size_t... Ns> iterator begin_impl(std::index_sequence<Ns...>) {
      return iterator(hbtl::adl_begin(std::get<Ns>(storage))...);
    }

    template <size_t... Ns> const_iterator end_impl(std::index_sequence<Ns...>) const {
      return const_iterator(hbtl::adl_end(std::get<Ns>(storage))...);
    }
    template <size_t... Ns> iterator end_impl(std::index_sequence<Ns...>) {
      return iterator(hbtl::adl_end(std::get<Ns>(storage))...);
    }
  };

  } // end namespace detail

  /// zip iterator for two or more iteratable types. Iteration continues until the
  /// end of the *shortest* iteratee is reached.
  template <typename T, typename U, typename... Args>
  detail::zippy<detail::zip_shortest, T, U, Args...> zip(T && t, U && u, Args && ...args) {
    return detail::zippy<detail::zip_shortest, T, U, Args...>(std::forward<T>(t), std::forward<U>(u),
                                                              std::forward<Args>(args)...);
  }

  /// zip iterator that assumes that all iteratees have the same length.
  /// In builds with assertions on, this assumption is checked before the
  /// iteration starts.
  template <typename T, typename U, typename... Args>
  detail::zippy<detail::zip_first, T, U, Args...> zip_equal(T && t, U && u, Args && ...args) {
    assert(all_equal({std::distance(hbtl::adl_begin(t), hbtl::adl_end(t)),
                      std::distance(hbtl::adl_begin(u), hbtl::adl_end(u)),
                      std::distance(hbtl::adl_begin(args), hbtl::adl_end(args))...}) &&
           "Iteratees do not have equal length");
    return detail::zippy<detail::zip_first, T, U, Args...>(std::forward<T>(t), std::forward<U>(u),
                                                           std::forward<Args>(args)...);
  }

  /// zip iterator that, for the sake of efficiency, assumes the first iteratee to
  /// be the shortest. Iteration continues until the end of the first iteratee is
  /// reached. In builds with assertions on, we check that the assumption about
  /// the first iteratee being the shortest holds.
  template <typename T, typename U, typename... Args>
  detail::zippy<detail::zip_first, T, U, Args...> zip_first(T && t, U && u, Args && ...args) {
    assert(std::distance(hbtl::adl_begin(t), hbtl::adl_end(t)) <=
               std::min({std::distance(hbtl::adl_begin(u), hbtl::adl_end(u)),
                         std::distance(hbtl::adl_begin(args), hbtl::adl_end(args))...}) &&
           "First iteratee is not the shortest");

    return detail::zippy<detail::zip_first, T, U, Args...>(std::forward<T>(t), std::forward<U>(u),
                                                           std::forward<Args>(args)...);
  }

  namespace detail {
  template <typename Iter> Iter next_or_end(const Iter &I, const Iter &End) {
    if (I == End)
      return End;
    return std::next(I);
  }

  template <typename Iter>
  auto deref_or_none(const Iter &I, const Iter &End)
      -> hbtl::optional<std::remove_const_t<std::remove_reference_t<decltype(*I)>>> {
    if (I == End)
      return hbtl::nullopt;
    return *I;
  }

  template <typename Iter> struct ZipLongestItemType {
    using type = hbtl::optional<std::remove_const_t<std::remove_reference_t<decltype(*std::declval<Iter>())>>>;
  };

  template <typename... Iters> struct ZipLongestTupleType {
    using type = std::tuple<typename ZipLongestItemType<Iters>::type...>;
  };

  template <typename... Iters>
  class zip_longest_iterator
      : public iterator_facade_base<
            zip_longest_iterator<Iters...>,
            std::common_type_t<std::forward_iterator_tag, typename std::iterator_traits<Iters>::iterator_category...>,
            typename ZipLongestTupleType<Iters...>::type,
            typename std::iterator_traits<std::tuple_element_t<0, std::tuple<Iters...>>>::difference_type,
            typename ZipLongestTupleType<Iters...>::type *, typename ZipLongestTupleType<Iters...>::type> {
  public:
    using value_type = typename ZipLongestTupleType<Iters...>::type;

  private:
    std::tuple<Iters...> iterators;
    std::tuple<Iters...> end_iterators;

    template <size_t... Ns> bool test(const zip_longest_iterator<Iters...> &other, std::index_sequence<Ns...>) const {
      return any_of(std::initializer_list<bool>{std::get<Ns>(this->iterators) != std::get<Ns>(other.iterators)...},
                    identity<bool>{});
    }

    template <size_t... Ns> value_type deref(std::index_sequence<Ns...>) const {
      return value_type(deref_or_none(std::get<Ns>(iterators), std::get<Ns>(end_iterators))...);
    }

    template <size_t... Ns> decltype(iterators) tup_inc(std::index_sequence<Ns...>) const {
      return std::tuple<Iters...>(next_or_end(std::get<Ns>(iterators), std::get<Ns>(end_iterators))...);
    }

  public:
    zip_longest_iterator(std::pair<Iters &&, Iters &&>... ts)
        : iterators(std::forward<Iters>(ts.first)...), end_iterators(std::forward<Iters>(ts.second)...) {}

    value_type operator*() const { return deref(std::index_sequence_for<Iters...>{}); }

    zip_longest_iterator<Iters...> &operator++() {
      iterators = tup_inc(std::index_sequence_for<Iters...>{});
      return *this;
    }

    bool operator==(const zip_longest_iterator<Iters...> &other) const {
      return !test(other, std::index_sequence_for<Iters...>{});
    }
  };

  template <typename... Args> class zip_longest_range {
  public:
    using iterator = zip_longest_iterator<decltype(hbtl::adl_begin(std::declval<Args>()))...>;
    using iterator_category = typename iterator::iterator_category;
    using value_type = typename iterator::value_type;
    using difference_type = typename iterator::difference_type;
    using pointer = typename iterator::pointer;
    using reference = typename iterator::reference;

  private:
    std::tuple<Args...> ts;

    template <size_t... Ns> iterator begin_impl(std::index_sequence<Ns...>) const {
      return iterator(std::make_pair(hbtl::adl_begin(std::get<Ns>(ts)), hbtl::adl_end(std::get<Ns>(ts)))...);
    }

    template <size_t... Ns> iterator end_impl(std::index_sequence<Ns...>) const {
      return iterator(std::make_pair(hbtl::adl_end(std::get<Ns>(ts)), hbtl::adl_end(std::get<Ns>(ts)))...);
    }

  public:
    zip_longest_range(Args &&...ts_) : ts(std::forward<Args>(ts_)...) {}

    iterator begin() const { return begin_impl(std::index_sequence_for<Args...>{}); }
    iterator end() const { return end_impl(std::index_sequence_for<Args...>{}); }
  };
  } // namespace detail

  /// Iterate over two or more iterators at the same time. Iteration continues
  /// until all iterators reach the end. The hbtl::optional only contains a value
  /// if the iterator has not reached the end.
  template <typename T, typename U, typename... Args>
  detail::zip_longest_range<T, U, Args...> zip_longest(T && t, U && u, Args && ...args) {
    return detail::zip_longest_range<T, U, Args...>(std::forward<T>(t), std::forward<U>(u),
                                                    std::forward<Args>(args)...);
  }

  /// Iterator wrapper that concatenates sequences together.
  ///
  /// This can concatenate different iterators, even with different types, into
  /// a single iterator provided the value types of all the concatenated
  /// iterators expose `reference` and `pointer` types that can be converted to
  /// `ValueT &` and `ValueT *` respectively. It doesn't support more
  /// interesting/customized pointer or reference types.
  ///
  /// Currently this only supports forward or higher iterator categories as
  /// inputs and always exposes a forward iterator interface.
  template <typename ValueT, typename... IterTs>
  class concat_iterator
      : public iterator_facade_base<concat_iterator<ValueT, IterTs...>, std::forward_iterator_tag, ValueT> {
    using BaseT = typename concat_iterator::iterator_facade_base;

    /// We store both the current and end iterators for each concatenated
    /// sequence in a tuple of pairs.
    ///
    /// Note that something like iterator_range seems nice at first here, but the
    /// range properties are of little benefit and end up getting in the way
    /// because we need to do mutation on the current iterators.
    std::tuple<IterTs...> Begins;
    std::tuple<IterTs...> Ends;

    /// Attempts to increment a specific iterator.
    ///
    /// Returns true if it was able to increment the iterator. Returns false if
    /// the iterator is already at the end iterator.
    template <size_t Index> bool incrementHelper() {
      auto &Begin = std::get<Index>(Begins);
      auto &End = std::get<Index>(Ends);
      if (Begin == End)
        return false;

      ++Begin;
      return true;
    }

    /// Increments the first non-end iterator.
    ///
    /// It is an error to call this with all iterators at the end.
    template <size_t... Ns> void increment(std::index_sequence<Ns...>) {
      // Build a sequence of functions to increment each iterator if possible.
      bool (concat_iterator::*IncrementHelperFns[])() = {&concat_iterator::incrementHelper<Ns>...};

      // Loop over them, and stop as soon as we succeed at incrementing one.
      for (auto &IncrementHelperFn : IncrementHelperFns)
        if ((this->*IncrementHelperFn)())
          return;

      assert(false && "Attempted to increment an end concat iterator!");
    }

    /// Returns null if the specified iterator is at the end. Otherwise,
    /// dereferences the iterator and returns the address of the resulting
    /// reference.
    template <size_t Index> ValueT *getHelper() const {
      auto &Begin = std::get<Index>(Begins);
      auto &End = std::get<Index>(Ends);
      if (Begin == End)
        return nullptr;

      return &*Begin;
    }

    /// Finds the first non-end iterator, dereferences, and returns the resulting
    /// reference.
    ///
    /// It is an error to call this with all iterators at the end.
    template <size_t... Ns> ValueT &get(std::index_sequence<Ns...>) const {
      // Build a sequence of functions to get from iterator if possible.
      ValueT *(concat_iterator::*GetHelperFns[])() const = {&concat_iterator::getHelper<Ns>...};

      // Loop over them, and return the first result we find.
      for (auto &GetHelperFn : GetHelperFns)
        if (ValueT *P = (this->*GetHelperFn)())
          return *P;

      assert(false && "Attempted to get a pointer from an end concat iterator!");
    }

  public:
    /// Constructs an iterator from a sequence of ranges.
    ///
    /// We need the full range to know how to switch between each of the
    /// iterators.
    template <typename... RangeTs>
    explicit concat_iterator(RangeTs &&...Ranges) : Begins(std::begin(Ranges)...), Ends(std::end(Ranges)...) {}

    using BaseT::operator++;

    concat_iterator &operator++() {
      increment(std::index_sequence_for<IterTs...>());
      return *this;
    }

    ValueT &operator*() const { return get(std::index_sequence_for<IterTs...>()); }

    bool operator==(const concat_iterator &RHS) const { return Begins == RHS.Begins && Ends == RHS.Ends; }
  };

  namespace detail {

  /// Helper to store a sequence of ranges being concatenated and access them.
  ///
  /// This is designed to facilitate providing actual storage when temporaries
  /// are passed into the constructor such that we can use it as part of range
  /// based for loops.
  template <typename ValueT, typename... RangeTs> class concat_range {
  public:
    using iterator = concat_iterator<ValueT, decltype(std::begin(std::declval<RangeTs &>()))...>;

  private:
    std::tuple<RangeTs...> Ranges;

    template <size_t... Ns> iterator begin_impl(std::index_sequence<Ns...>) {
      return iterator(std::get<Ns>(Ranges)...);
    }
    template <size_t... Ns> iterator begin_impl(std::index_sequence<Ns...>) const {
      return iterator(std::get<Ns>(Ranges)...);
    }
    template <size_t... Ns> iterator end_impl(std::index_sequence<Ns...>) {
      return iterator(make_range(std::end(std::get<Ns>(Ranges)), std::end(std::get<Ns>(Ranges)))...);
    }
    template <size_t... Ns> iterator end_impl(std::index_sequence<Ns...>) const {
      return iterator(make_range(std::end(std::get<Ns>(Ranges)), std::end(std::get<Ns>(Ranges)))...);
    }

  public:
    concat_range(RangeTs &&...Ranges) : Ranges(std::forward<RangeTs>(Ranges)...) {}

    iterator begin() { return begin_impl(std::index_sequence_for<RangeTs...>{}); }
    iterator begin() const { return begin_impl(std::index_sequence_for<RangeTs...>{}); }
    iterator end() { return end_impl(std::index_sequence_for<RangeTs...>{}); }
    iterator end() const { return end_impl(std::index_sequence_for<RangeTs...>{}); }
  };

  } // end namespace detail

  /// Concatenated range across two or more ranges.
  ///
  /// The desired value type must be explicitly specified.
  template <typename ValueT, typename... RangeTs>
  detail::concat_range<ValueT, RangeTs...> concat(RangeTs && ...Ranges) {
    static_assert(sizeof...(RangeTs) > 1, "Need more than one range to concatenate!");
    return detail::concat_range<ValueT, RangeTs...>(std::forward<RangeTs>(Ranges)...);
  }

  /// A utility class used to implement an iterator that contains some base object
  /// and an index. The iterator moves the index but keeps the base constant.
  template <typename DerivedT, typename BaseT, typename T, typename PointerT = T *, typename ReferenceT = T &>
  class indexed_accessor_iterator : public hbtl::iterator_facade_base<DerivedT, std::random_access_iterator_tag, T,
                                                                      std::ptrdiff_t, PointerT, ReferenceT> {
  public:
    ptrdiff_t operator-(const indexed_accessor_iterator &rhs) const {
      assert(base == rhs.base && "incompatible iterators");
      return index - rhs.index;
    }
    bool operator==(const indexed_accessor_iterator &rhs) const { return base == rhs.base && index == rhs.index; }
    bool operator<(const indexed_accessor_iterator &rhs) const {
      assert(base == rhs.base && "incompatible iterators");
      return index < rhs.index;
    }

    DerivedT &operator+=(ptrdiff_t offset) {
      this->index += offset;
      return static_cast<DerivedT &>(*this);
    }
    DerivedT &operator-=(ptrdiff_t offset) {
      this->index -= offset;
      return static_cast<DerivedT &>(*this);
    }

    /// Returns the current index of the iterator.
    ptrdiff_t getIndex() const { return index; }

    /// Returns the current base of the iterator.
    const BaseT &getBase() const { return base; }

  protected:
    indexed_accessor_iterator(BaseT base, ptrdiff_t index) : base(base), index(index) {}
    BaseT base;
    ptrdiff_t index;
  };

  namespace detail {
  /// The class represents the base of a range of indexed_accessor_iterators. It
  /// provides support for many different range functionalities, e.g.
  /// drop_front/slice/etc.. Derived range classes must implement the following
  /// static methods:
  ///   * ReferenceT dereference_iterator(const BaseT &base, ptrdiff_t index)
  ///     - Dereference an iterator pointing to the base object at the given
  ///       index.
  ///   * BaseT offset_base(const BaseT &base, ptrdiff_t index)
  ///     - Return a new base that is offset from the provide base by 'index'
  ///       elements.
  template <typename DerivedT, typename BaseT, typename T, typename PointerT = T *, typename ReferenceT = T &>
  class indexed_accessor_range_base {
  public:
    using RangeBaseT = indexed_accessor_range_base;

    /// An iterator element of this range.
    class iterator : public indexed_accessor_iterator<iterator, BaseT, T, PointerT, ReferenceT> {
    public:
      // Index into this iterator, invoking a static method on the derived type.
      ReferenceT operator*() const { return DerivedT::dereference_iterator(this->getBase(), this->getIndex()); }

    private:
      iterator(BaseT owner, ptrdiff_t curIndex) : iterator::indexed_accessor_iterator(owner, curIndex) {}

      /// Allow access to the constructor.
      friend indexed_accessor_range_base<DerivedT, BaseT, T, PointerT, ReferenceT>;
    };

    indexed_accessor_range_base(iterator begin, iterator end)
        : base(offset_base(begin.getBase(), begin.getIndex())), count(end.getIndex() - begin.getIndex()) {}
    indexed_accessor_range_base(const iterator_range<iterator> &range)
        : indexed_accessor_range_base(range.begin(), range.end()) {}
    indexed_accessor_range_base(BaseT base, ptrdiff_t count) : base(base), count(count) {}

    iterator begin() const { return iterator(base, 0); }
    iterator end() const { return iterator(base, count); }
    ReferenceT operator[](size_t Index) const {
      assert(Index < size() && "invalid index for value range");
      return DerivedT::dereference_iterator(base, static_cast<ptrdiff_t>(Index));
    }
    ReferenceT front() const {
      assert(!empty() && "expected non-empty range");
      return (*this)[0];
    }
    ReferenceT back() const {
      assert(!empty() && "expected non-empty range");
      return (*this)[size() - 1];
    }

    /// Compare this range with another.
    template <typename OtherT> friend bool operator==(const indexed_accessor_range_base &lhs, const OtherT &rhs) {
      return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
    template <typename OtherT> friend bool operator!=(const indexed_accessor_range_base &lhs, const OtherT &rhs) {
      return !(lhs == rhs);
    }

    /// Return the size of this range.
    size_t size() const { return count; }

    /// Return if the range is empty.
    bool empty() const { return size() == 0; }

    /// Drop the first N elements, and keep M elements.
    DerivedT slice(size_t n, size_t m) const {
      assert(n + m <= size() && "invalid size specifiers");
      return DerivedT(offset_base(base, n), m);
    }

    /// Drop the first n elements.
    DerivedT drop_front(size_t n = 1) const {
      assert(size() >= n && "Dropping more elements than exist");
      return slice(n, size() - n);
    }
    /// Drop the last n elements.
    DerivedT drop_back(size_t n = 1) const {
      assert(size() >= n && "Dropping more elements than exist");
      return DerivedT(base, size() - n);
    }

    /// Take the first n elements.
    DerivedT take_front(size_t n = 1) const {
      return n < size() ? drop_back(size() - n) : static_cast<const DerivedT &>(*this);
    }

    /// Take the last n elements.
    DerivedT take_back(size_t n = 1) const {
      return n < size() ? drop_front(size() - n) : static_cast<const DerivedT &>(*this);
    }

    /// Allow conversion to any type accepting an iterator_range.
    template <typename RangeT,
              typename = std::enable_if_t<std::is_constructible<RangeT, iterator_range<iterator>>::value>>
    operator RangeT() const {
      return RangeT(iterator_range<iterator>(*this));
    }

    /// Returns the base of this range.
    const BaseT &getBase() const { return base; }

  private:
    /// Offset the given base by the given amount.
    static BaseT offset_base(const BaseT &base, size_t n) { return n == 0 ? base : DerivedT::offset_base(base, n); }

  protected:
    indexed_accessor_range_base(const indexed_accessor_range_base &) = default;
    indexed_accessor_range_base(indexed_accessor_range_base &&) = default;
    indexed_accessor_range_base &operator=(const indexed_accessor_range_base &) = default;

    /// The base that owns the provided range of values.
    BaseT base;
    /// The size from the owning range.
    ptrdiff_t count;
  };
  } // end namespace detail

  /// This class provides an implementation of a range of
  /// indexed_accessor_iterators where the base is not indexable. Ranges with
  /// bases that are offsetable should derive from indexed_accessor_range_base
  /// instead. Derived range classes are expected to implement the following
  /// static method:
  ///   * ReferenceT dereference(const BaseT &base, ptrdiff_t index)
  ///     - Dereference an iterator pointing to a parent base at the given index.
  template <typename DerivedT, typename BaseT, typename T, typename PointerT = T *, typename ReferenceT = T &>
  class indexed_accessor_range
      : public detail::indexed_accessor_range_base<DerivedT, std::pair<BaseT, ptrdiff_t>, T, PointerT, ReferenceT> {
  public:
    indexed_accessor_range(BaseT base, ptrdiff_t startIndex, ptrdiff_t count)
        : detail::indexed_accessor_range_base<DerivedT, std::pair<BaseT, ptrdiff_t>, T, PointerT, ReferenceT>(
              std::make_pair(base, startIndex), count) {}
    using detail::indexed_accessor_range_base<DerivedT, std::pair<BaseT, ptrdiff_t>, T, PointerT,
                                              ReferenceT>::indexed_accessor_range_base;

    /// Returns the current base of the range.
    const BaseT &getBase() const { return this->base.first; }

    /// Returns the current start index of the range.
    ptrdiff_t getStartIndex() const { return this->base.second; }

    /// See `detail::indexed_accessor_range_base` for details.
    static std::pair<BaseT, ptrdiff_t> offset_base(const std::pair<BaseT, ptrdiff_t> &base, ptrdiff_t index) {
      // We encode the internal base as a pair of the derived base and a start
      // index into the derived base.
      return std::make_pair(base.first, base.second + index);
    }
    /// See `detail::indexed_accessor_range_base` for details.
    static ReferenceT dereference_iterator(const std::pair<BaseT, ptrdiff_t> &base, ptrdiff_t index) {
      return DerivedT::dereference(base.first, base.second + index);
    }
  };

  namespace detail {
  /// Return a reference to the first or second member of a reference. Otherwise,
  /// return a copy of the member of a temporary.
  ///
  /// When passing a range whose iterators return values instead of references,
  /// the reference must be dropped from `decltype((elt.first))`, which will
  /// always be a reference, to avoid returning a reference to a temporary.
  template <typename EltTy, typename FirstTy> class first_or_second_type {
  public:
    using type = std::conditional_t<std::is_reference<EltTy>::value, FirstTy, std::remove_reference_t<FirstTy>>;
  };
  } // end namespace detail

  /// Given a container of pairs, return a range over the first elements.
  template <typename ContainerTy> auto make_first_range(ContainerTy && c) {
    using EltTy = decltype((*std::begin(c)));
    return hbtl::map_range(std::forward<ContainerTy>(c),
                           [](EltTy elt) -> typename detail::first_or_second_type<EltTy, decltype((elt.first))>::type {
                             return elt.first;
                           });
  }

  /// Given a container of pairs, return a range over the second elements.
  template <typename ContainerTy> auto make_second_range(ContainerTy && c) {
    using EltTy = decltype((*std::begin(c)));
    return hbtl::map_range(std::forward<ContainerTy>(c),
                           [](EltTy elt) -> typename detail::first_or_second_type<EltTy, decltype((elt.second))>::type {
                             return elt.second;
                           });
  }

  //===----------------------------------------------------------------------===//
  //     Extra additions to <utility>
  //===----------------------------------------------------------------------===//

  /// Function object to check whether the first component of a container
  /// supported by std::get (like std::pair and std::tuple) compares less than the
  /// first component of another container.
  struct less_first {
    template <typename T> bool operator()(const T &lhs, const T &rhs) const {
      return std::less<>()(std::get<0>(lhs), std::get<0>(rhs));
    }
  };

  /// Function object to check whether the second component of a container
  /// supported by std::get (like std::pair and std::tuple) compares less than the
  /// second component of another container.
  struct less_second {
    template <typename T> bool operator()(const T &lhs, const T &rhs) const {
      return std::less<>()(std::get<1>(lhs), std::get<1>(rhs));
    }
  };

  /// \brief Function object to apply a binary function to the first component of
  /// a std::pair.
  template <typename FuncTy> struct on_first {
    FuncTy func;

    template <typename T> decltype(auto) operator()(const T &lhs, const T &rhs) const {
      return func(lhs.first, rhs.first);
    }
  };

  /// Utility type to build an inheritance chain that makes it easy to rank
  /// overload candidates.
  template <int N> struct rank : rank<N - 1> {};
  template <> struct rank<0> {};

  /// traits class for checking whether type T is one of any of the given
  /// types in the variadic list.
  template <typename T, typename... Ts> using is_one_of = disjunction<std::is_same<T, Ts>...>;

  /// traits class for checking whether type T is a base class for all
  ///  the given types in the variadic list.
  template <typename T, typename... Ts> using are_base_of = detail::conjunction<std::is_base_of<T, Ts>...>;

  namespace detail {
  template <typename... Ts> struct Visitor;

  template <typename HeadT, typename... TailTs>
  struct Visitor<HeadT, TailTs...> : remove_cvref_t<HeadT>, Visitor<TailTs...> {
    explicit constexpr Visitor(HeadT &&Head, TailTs &&...Tail)
        : remove_cvref_t<HeadT>(std::forward<HeadT>(Head)), Visitor<TailTs...>(std::forward<TailTs>(Tail)...) {}
    using remove_cvref_t<HeadT>::operator();
    using Visitor<TailTs...>::operator();
  };

  template <typename HeadT> struct Visitor<HeadT> : remove_cvref_t<HeadT> {
    explicit constexpr Visitor(HeadT &&Head) : remove_cvref_t<HeadT>(std::forward<HeadT>(Head)) {}
    using remove_cvref_t<HeadT>::operator();
  };
  } // namespace detail

  /// Returns an opaquely-typed Callable object whose operator() overload set is
  /// the sum of the operator() overload sets of each CallableT in CallableTs.
  ///
  /// The type of the returned object derives from each CallableT in CallableTs.
  /// The returned object is constructed by invoking the appropriate copy or move
  /// constructor of each CallableT, as selected by overload resolution on the
  /// corresponding argument to makeVisitor.
  ///
  /// Example:
  ///
  /// \code
  /// auto visitor = makeVisitor([](auto) { return "unhandled type"; },
  ///                            [](int i) { return "int"; },
  ///                            [](std::string s) { return "str"; });
  /// auto a = visitor(42);    // `a` is now "int".
  /// auto b = visitor("foo"); // `b` is now "str".
  /// auto c = visitor(3.14f); // `c` is now "unhandled type".
  /// \endcode
  ///
  /// Example of making a visitor with a lambda which captures a move-only type:
  ///
  /// \code
  /// std::unique_ptr<FooHandler> FH = /* ... */;
  /// auto visitor = makeVisitor(
  ///     [FH{std::move(FH)}](Foo F) { return FH->handle(F); },
  ///     [](int i) { return i; },
  ///     [](std::string s) { return atoi(s); });
  /// \endcode
  template <typename... CallableTs> constexpr decltype(auto) makeVisitor(CallableTs && ...Callables) {
    return detail::Visitor<CallableTs...>(std::forward<CallableTs>(Callables)...);
  }

  //===----------------------------------------------------------------------===//
  //     Extra additions to <algorithm>
  //===----------------------------------------------------------------------===//

  /// Adapt std::less<T> for array_pod_sort.
  template <typename T> inline int array_pod_sort_comparator(const void *P1, const void *P2) {
    if (std::less<T>()(*reinterpret_cast<const T *>(P1), *reinterpret_cast<const T *>(P2)))
      return -1;
    if (std::less<T>()(*reinterpret_cast<const T *>(P2), *reinterpret_cast<const T *>(P1)))
      return 1;
    return 0;
  }

  /// get_array_pod_sort_comparator - This is an internal helper function used to
  /// get type deduction of T right.
  template <typename T> inline int (*get_array_pod_sort_comparator(const T &))(const void *, const void *) {
    return array_pod_sort_comparator<T>;
  }

#ifdef EXPENSIVE_CHECKS
  namespace detail {

  template <class IteratorTy> inline void presortShuffle(IteratorTy Start, IteratorTy End) {
    ::hbtl::shuffle({Start, End});
  }

  } // end namespace detail
#endif

  /// array_pod_sort - This sorts an array with the specified start and end
  /// extent.  This is just like std::sort, except that it calls qsort instead of
  /// using an inlined template.  qsort is slightly slower than std::sort, but
  /// most sorts are not performance critical in LLVM and std::sort has to be
  /// template instantiated for each type, leading to significant measured code
  /// bloat.  This function should generally be used instead of std::sort where
  /// possible.
  ///
  /// This function assumes that you have simple POD-like types that can be
  /// compared with std::less and can be moved with memcpy.  If this isn't true,
  /// you should use std::sort.
  ///
  /// NOTE: If qsort_r were portable, we could allow a custom comparator and
  /// default to std::less.
  template <class IteratorTy> inline void array_pod_sort(IteratorTy Start, IteratorTy End) {
    // Don't inefficiently call qsort with one element or trigger undefined
    // behavior with an empty sequence.
    auto NElts = End - Start;
    if (NElts <= 1)
      return;
#ifdef EXPENSIVE_CHECKS
    detail::presortShuffle<IteratorTy>(Start, End);
#endif
    qsort(&*Start, NElts, sizeof(*Start), get_array_pod_sort_comparator(*Start));
  }

  template <class IteratorTy>
  inline void array_pod_sort(IteratorTy Start, IteratorTy End,
                             int (*Compare)(const typename std::iterator_traits<IteratorTy>::value_type *,
                                            const typename std::iterator_traits<IteratorTy>::value_type *)) {
    // Don't inefficiently call qsort with one element or trigger undefined
    // behavior with an empty sequence.
    auto NElts = End - Start;
    if (NElts <= 1)
      return;
#ifdef EXPENSIVE_CHECKS
    detail::presortShuffle<IteratorTy>(Start, End);
#endif
    qsort(&*Start, NElts, sizeof(*Start), reinterpret_cast<int (*)(const void *, const void *)>(Compare));
  }

  namespace detail {
  template <typename T>
  // We can use qsort if the iterator type is a pointer and the underlying value
  // is trivially copyable.
  using sort_trivially_copyable =
      detail::conjunction<std::is_pointer<T>, std::is_trivially_copyable<typename std::iterator_traits<T>::value_type>>;
  } // namespace detail

  // Provide wrappers to std::sort which shuffle the elements before sorting
  // to help uncover non-deterministic behavior (PR35135).
  template <typename IteratorTy> inline void sort(IteratorTy Start, IteratorTy End) {
    if HBTL_CONSTEXPR_IF (detail::sort_trivially_copyable<IteratorTy>::value) {
      // Forward trivially copyable types to array_pod_sort. This avoids a large
      // amount of code bloat for a minor performance hit.
      array_pod_sort(Start, End);
    } else {
#ifdef EXPENSIVE_CHECKS
      detail::presortShuffle<IteratorTy>(Start, End);
#endif
      std::sort(Start, End);
    }
  }

  template <typename Container> inline void sort(Container && C) { hbtl::sort(hbtl::adl_begin(C), hbtl::adl_end(C)); }

  template <typename IteratorTy, typename Compare> inline void sort(IteratorTy Start, IteratorTy End, Compare Comp) {
#ifdef EXPENSIVE_CHECKS
    detail::presortShuffle<IteratorTy>(Start, End);
#endif
    std::sort(Start, End, Comp);
  }

  template <typename Container, typename Compare> inline void sort(Container && C, Compare Comp) {
    hbtl::sort(hbtl::adl_begin(C), hbtl::adl_end(C), Comp);
  }

  /// Get the size of a range. This is a wrapper function around std::distance
  /// which is only enabled when the operation is O(1).
  template <typename R>
  auto size(R && Range,
            std::enable_if_t<
                std::is_base_of<std::random_access_iterator_tag,
                                typename std::iterator_traits<decltype(Range.begin())>::iterator_category>::value,
                void> * = nullptr) {
    return std::distance(Range.begin(), Range.end());
  }

  /// Provide wrappers to std::for_each which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename UnaryFunction> UnaryFunction for_each(R && Range, UnaryFunction F) {
    return std::for_each(hbtl::adl_begin(Range), hbtl::adl_end(Range), F);
  }

  /// Provide wrappers to std::all_of which take ranges instead of having to pass
  /// begin/end explicitly.
  template <typename R, typename UnaryPredicate> bool all_of(R && Range, UnaryPredicate P) {
    return std::all_of(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::any_of which take ranges instead of having to pass
  /// begin/end explicitly.
  template <typename R, typename UnaryPredicate> bool any_of(R && Range, UnaryPredicate P) {
    return std::any_of(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::none_of which take ranges instead of having to pass
  /// begin/end explicitly.
  template <typename R, typename UnaryPredicate> bool none_of(R && Range, UnaryPredicate P) {
    return std::none_of(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::find which take ranges instead of having to pass
  /// begin/end explicitly.
  template <typename R, typename T> auto find(R && Range, const T &Val) {
    return std::find(hbtl::adl_begin(Range), hbtl::adl_end(Range), Val);
  }

  /// Provide wrappers to std::find_if which take ranges instead of having to pass
  /// begin/end explicitly.
  template <typename R, typename UnaryPredicate> auto find_if(R && Range, UnaryPredicate P) {
    return std::find_if(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  template <typename R, typename UnaryPredicate> auto find_if_not(R && Range, UnaryPredicate P) {
    return std::find_if_not(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::remove_if which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename UnaryPredicate> auto remove_if(R && Range, UnaryPredicate P) {
    return std::remove_if(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::copy_if which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename OutputIt, typename UnaryPredicate>
  OutputIt copy_if(R && Range, OutputIt Out, UnaryPredicate P) {
    return std::copy_if(hbtl::adl_begin(Range), hbtl::adl_end(Range), Out, P);
  }

  /// Return the single value in \p Range that satisfies
  /// \p P(<member of \p Range> *, AllowRepeats)->T * returning nullptr
  /// when no values or multiple values were found.
  /// When \p AllowRepeats is true, multiple values that compare equal
  /// are allowed.
  template <typename T, typename R, typename Predicate>
  T *find_singleton(R && Range, Predicate P, bool AllowRepeats = false) {
    T *RC = nullptr;
    for (auto *A : Range) {
      if (T *PRC = P(A, AllowRepeats)) {
        if (RC) {
          if (!AllowRepeats || PRC != RC)
            return nullptr;
        } else
          RC = PRC;
      }
    }
    return RC;
  }

  /// Return a pair consisting of the single value in \p Range that satisfies
  /// \p P(<member of \p Range> *, AllowRepeats)->std::pair<T*, bool> returning
  /// nullptr when no values or multiple values were found, and a bool indicating
  /// whether multiple values were found to cause the nullptr.
  /// When \p AllowRepeats is true, multiple values that compare equal are
  /// allowed.  The predicate \p P returns a pair<T *, bool> where T is the
  /// singleton while the bool indicates whether multiples have already been
  /// found.  It is expected that first will be nullptr when second is true.
  /// This allows using find_singleton_nested within the predicate \P.
  template <typename T, typename R, typename Predicate>
  std::pair<T *, bool> find_singleton_nested(R && Range, Predicate P, bool AllowRepeats = false) {
    T *RC = nullptr;
    for (auto *A : Range) {
      std::pair<T *, bool> PRC = P(A, AllowRepeats);
      if (PRC.second) {
        assert(PRC.first == nullptr && "Inconsistent return values in find_singleton_nested.");
        return PRC;
      }
      if (PRC.first) {
        if (RC) {
          if (!AllowRepeats || PRC.first != RC)
            return {nullptr, true};
        } else
          RC = PRC.first;
      }
    }
    return {RC, false};
  }

  template <typename R, typename OutputIt> OutputIt copy(R && Range, OutputIt Out) {
    return std::copy(hbtl::adl_begin(Range), hbtl::adl_end(Range), Out);
  }

  /// Provide wrappers to std::replace_copy_if which take ranges instead of having
  /// to pass begin/end explicitly.
  template <typename R, typename OutputIt, typename UnaryPredicate, typename T>
  OutputIt replace_copy_if(R && Range, OutputIt Out, UnaryPredicate P, const T &NewValue) {
    return std::replace_copy_if(hbtl::adl_begin(Range), hbtl::adl_end(Range), Out, P, NewValue);
  }

  /// Provide wrappers to std::replace_copy which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename OutputIt, typename T>
  OutputIt replace_copy(R && Range, OutputIt Out, const T &OldValue, const T &NewValue) {
    return std::replace_copy(hbtl::adl_begin(Range), hbtl::adl_end(Range), Out, OldValue, NewValue);
  }

  /// Provide wrappers to std::move which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename OutputIt> OutputIt move(R && Range, OutputIt Out) {
    return std::move(hbtl::adl_begin(Range), hbtl::adl_end(Range), Out);
  }

  /// Wrapper function around std::find to detect if an element exists
  /// in a container.
  template <typename R, typename E> bool is_contained(R && Range, const E &Element) {
    return std::find(hbtl::adl_begin(Range), hbtl::adl_end(Range), Element) != hbtl::adl_end(Range);
  }

  /// Returns true iff \p Element exists in \p Set. This overload takes \p Set as
  /// an initializer list and is `constexpr`-friendly.
  template <typename T, typename E> constexpr bool is_contained(std::initializer_list<T> Set, const E &Element) {
    // TODO: Use std::find when we switch to C++20.
    for (const T &V : Set)
      if (V == Element)
        return true;
    return false;
  }

  /// Wrapper function around std::is_sorted to check if elements in a range \p R
  /// are sorted with respect to a comparator \p C.
  template <typename R, typename Compare> bool is_sorted(R && Range, Compare C) {
    return std::is_sorted(hbtl::adl_begin(Range), hbtl::adl_end(Range), C);
  }

  /// Wrapper function around std::is_sorted to check if elements in a range \p R
  /// are sorted in non-descending order.
  template <typename R> bool is_sorted(R && Range) {
    return std::is_sorted(hbtl::adl_begin(Range), hbtl::adl_end(Range));
  }

  /// Wrapper function around std::count to count the number of times an element
  /// \p Element occurs in the given range \p Range.
  template <typename R, typename E> auto count(R && Range, const E &Element) {
    return std::count(hbtl::adl_begin(Range), hbtl::adl_end(Range), Element);
  }

  /// Wrapper function around std::count_if to count the number of times an
  /// element satisfying a given predicate occurs in a range.
  template <typename R, typename UnaryPredicate> auto count_if(R && Range, UnaryPredicate P) {
    return std::count_if(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Wrapper function around std::transform to apply a function to a range and
  /// store the result elsewhere.
  template <typename R, typename OutputIt, typename UnaryFunction>
  OutputIt transform(R && Range, OutputIt d_first, UnaryFunction F) {
    return std::transform(hbtl::adl_begin(Range), hbtl::adl_end(Range), d_first, F);
  }

  /// Provide wrappers to std::partition which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename UnaryPredicate> auto partition(R && Range, UnaryPredicate P) {
    return std::partition(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  /// Provide wrappers to std::lower_bound which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename T> auto lower_bound(R && Range, T && Value) {
    return std::lower_bound(hbtl::adl_begin(Range), hbtl::adl_end(Range), std::forward<T>(Value));
  }

  template <typename R, typename T, typename Compare> auto lower_bound(R && Range, T && Value, Compare C) {
    return std::lower_bound(hbtl::adl_begin(Range), hbtl::adl_end(Range), std::forward<T>(Value), C);
  }

  /// Provide wrappers to std::upper_bound which take ranges instead of having to
  /// pass begin/end explicitly.
  template <typename R, typename T> auto upper_bound(R && Range, T && Value) {
    return std::upper_bound(hbtl::adl_begin(Range), hbtl::adl_end(Range), std::forward<T>(Value));
  }

  template <typename R, typename T, typename Compare> auto upper_bound(R && Range, T && Value, Compare C) {
    return std::upper_bound(hbtl::adl_begin(Range), hbtl::adl_end(Range), std::forward<T>(Value), C);
  }

  template <typename R> void stable_sort(R && Range) { std::stable_sort(hbtl::adl_begin(Range), hbtl::adl_end(Range)); }

  template <typename R, typename Compare> void stable_sort(R && Range, Compare C) {
    std::stable_sort(hbtl::adl_begin(Range), hbtl::adl_end(Range), C);
  }

  /// Binary search for the first iterator in a range where a predicate is false.
  /// Requires that C is always true below some limit, and always false above it.
  template <typename R, typename Predicate, typename Val = decltype(*adl_begin(std::declval<R>()))>
  auto partition_point(R && Range, Predicate P) {
    return std::partition_point(hbtl::adl_begin(Range), hbtl::adl_end(Range), P);
  }

  template <typename Range, typename Predicate> auto unique(Range && R, Predicate P) {
    return std::unique(hbtl::adl_begin(R), hbtl::adl_end(R), P);
  }

  /// Wrapper function around std::equal to detect if pair-wise elements between
  /// two ranges are the same.
  template <typename L, typename R> bool equal(L && LRange, R && RRange) {
    return std::equal(hbtl::adl_begin(LRange), hbtl::adl_end(LRange), hbtl::adl_begin(RRange), hbtl::adl_end(RRange));
  }

  /// Returns true if all elements in Range are equal or when the Range is empty.
  template <typename R> bool all_equal(R && Range) {
    auto Begin = hbtl::adl_begin(Range);
    auto End = hbtl::adl_end(Range);
    return Begin == End || std::equal(Begin + 1, End, Begin);
  }

  /// Returns true if all Values in the initializer lists are equal or the list
  // is empty.
  template <typename T> bool all_equal(std::initializer_list<T> Values) {
    return all_equal<std::initializer_list<T>>(std::move(Values));
  }

  /// Provide a container algorithm similar to C++ Library Fundamentals v2's
  /// `erase_if` which is equivalent to:
  ///
  ///   C.erase(remove_if(C, pred), C.end());
  ///
  /// This version works for any container with an erase method call accepting
  /// two iterators.
  template <typename Container, typename UnaryPredicate> void erase_if(Container & C, UnaryPredicate P) {
    C.erase(remove_if(C, P), C.end());
  }

  /// Wrapper function to remove a value from a container:
  ///
  /// C.erase(remove(C.begin(), C.end(), V), C.end());
  template <typename Container, typename ValueType> void erase_value(Container & C, ValueType V) {
    C.erase(std::remove(C.begin(), C.end(), V), C.end());
  }

  /// Wrapper function to append a range to a container.
  ///
  /// C.insert(C.end(), R.begin(), R.end());
  template <typename Container, typename Range> inline void append_range(Container & C, Range && R) {
    C.insert(C.end(), hbtl::adl_begin(R), hbtl::adl_end(R));
  }

  /// Given a sequence container Cont, replace the range [ContIt, ContEnd) with
  /// the range [ValIt, ValEnd) (which is not from the same container).
  template <typename Container, typename RandomAccessIterator>
  void replace(Container & Cont, typename Container::iterator ContIt, typename Container::iterator ContEnd,
               RandomAccessIterator ValIt, RandomAccessIterator ValEnd) {
    while (true) {
      if (ValIt == ValEnd) {
        Cont.erase(ContIt, ContEnd);
        return;
      } else if (ContIt == ContEnd) {
        Cont.insert(ContIt, ValIt, ValEnd);
        return;
      }
      *ContIt++ = *ValIt++;
    }
  }

  /// Given a sequence container Cont, replace the range [ContIt, ContEnd) with
  /// the range R.
  template <typename Container, typename Range = std::initializer_list<typename Container::value_type>>
  void replace(Container & Cont, typename Container::iterator ContIt, typename Container::iterator ContEnd, Range R) {
    replace(Cont, ContIt, ContEnd, R.begin(), R.end());
  }

  /// An STL-style algorithm similar to std::for_each that applies a second
  /// functor between every pair of elements.
  ///
  /// This provides the control flow logic to, for example, print a
  /// comma-separated list:
  /// \code
  ///   interleave(names.begin(), names.end(),
  ///              [&](hbtl::string_view name) { os << name; },
  ///              [&] { os << ", "; });
  /// \endcode
  template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor,
            typename = std::enable_if_t<!std::is_constructible<hbtl::string_view, UnaryFunctor>::value &&
                                        !std::is_constructible<hbtl::string_view, NullaryFunctor>::value>>
  inline void interleave(ForwardIterator begin, ForwardIterator end, UnaryFunctor each_fn, NullaryFunctor between_fn) {
    if (begin == end)
      return;
    each_fn(*begin);
    ++begin;
    for (; begin != end; ++begin) {
      between_fn();
      each_fn(*begin);
    }
  }

  template <typename Container, typename UnaryFunctor, typename NullaryFunctor,
            typename = std::enable_if_t<!std::is_constructible<hbtl::string_view, UnaryFunctor>::value &&
                                        !std::is_constructible<hbtl::string_view, NullaryFunctor>::value>>
  inline void interleave(const Container &c, UnaryFunctor each_fn, NullaryFunctor between_fn) {
    interleave(c.begin(), c.end(), each_fn, between_fn);
  }

  /// Overload of interleave for the common case of string separator.
  template <typename Container, typename UnaryFunctor, typename StreamT, typename T = detail::ValueOfRange<Container>>
  inline void interleave(const Container &c, StreamT &os, UnaryFunctor each_fn, const hbtl::string_view &separator) {
    interleave(c.begin(), c.end(), each_fn, [&] { os << separator; });
  }
  template <typename Container, typename StreamT, typename T = detail::ValueOfRange<Container>>
  inline void interleave(const Container &c, StreamT &os, const hbtl::string_view &separator) {
    interleave(
        c, os, [&](const T &a) { os << a; }, separator);
  }

  template <typename Container, typename UnaryFunctor, typename StreamT, typename T = detail::ValueOfRange<Container>>
  inline void interleaveComma(const Container &c, StreamT &os, UnaryFunctor each_fn) {
    interleave(c, os, each_fn, ", ");
  }
  template <typename Container, typename StreamT, typename T = detail::ValueOfRange<Container>>
  inline void interleaveComma(const Container &c, StreamT &os) {
    interleaveComma(c, os, [&](const T &a) { os << a; });
  }

  //===----------------------------------------------------------------------===//
  //     Extra additions to <memory>
  //===----------------------------------------------------------------------===//

  struct FreeDeleter {
    void operator()(void *v) { ::free(v); }
  };

  // template<typename First, typename Second>
  // struct pair_hash {
  //   size_t operator()(const std::pair<First, Second> &P) const {
  //     return std::hash<First>()(P.first) * 31 + std::hash<Second>()(P.second);
  //   }
  // };

  /// Binary functor that adapts to any other binary functor after dereferencing
  /// operands.
  template <typename T> struct deref {
    T func;

    // Could be further improved to cope with non-derivable functors and
    // non-binary functors (should be a variadic template member function
    // operator()).
    template <typename A, typename B> auto operator()(A &lhs, B &rhs) const {
      assert(lhs);
      assert(rhs);
      return func(*lhs, *rhs);
    }
  };

  namespace detail {

  template <typename R> class enumerator_iter;

  template <typename R> struct result_pair {
    using value_reference = typename std::iterator_traits<IterOfRange<R>>::reference;

    friend class enumerator_iter<R>;

    result_pair(std::size_t Index, IterOfRange<R> Iter) : Index(Index), Iter(Iter) {}

    std::size_t index() const { return Index; }
    value_reference value() const { return *Iter; }

  private:
    std::size_t Index = std::numeric_limits<std::size_t>::max();
    IterOfRange<R> Iter;
  };

  template <std::size_t i, typename R> decltype(auto) get(const result_pair<R> &Pair) {
    static_assert(i < 2, "");
    if HBTL_CONSTEXPR_IF (i == 0) {
      return Pair.index();
    } else {
      return Pair.value();
    }
  }

  template <typename R>
  class enumerator_iter
      : public iterator_facade_base<enumerator_iter<R>, std::forward_iterator_tag, const result_pair<R>> {
    using result_type = result_pair<R>;

  public:
    explicit enumerator_iter(IterOfRange<R> EndIter) : Result(std::numeric_limits<size_t>::max(), EndIter) {}

    enumerator_iter(std::size_t Index, IterOfRange<R> Iter) : Result(Index, Iter) {}

    const result_type &operator*() const { return Result; }

    enumerator_iter &operator++() {
      assert(Result.Index != std::numeric_limits<size_t>::max());
      ++Result.Iter;
      ++Result.Index;
      return *this;
    }

    bool operator==(const enumerator_iter &RHS) const {
      // Don't compare indices here, only iterators.  It's possible for an end
      // iterator to have different indices depending on whether it was created
      // by calling std::end() versus incrementing a valid iterator.
      return Result.Iter == RHS.Result.Iter;
    }

  private:
    result_type Result;
  };

  template <typename R> class enumerator {
  public:
    explicit enumerator(R &&Range) : TheRange(std::forward<R>(Range)) {}

    enumerator_iter<R> begin() { return enumerator_iter<R>(0, hbtl::adl_begin(TheRange)); }
    enumerator_iter<R> begin() const { return enumerator_iter<R>(0, hbtl::adl_begin(TheRange)); }

    enumerator_iter<R> end() { return enumerator_iter<R>(hbtl::adl_end(TheRange)); }
    enumerator_iter<R> end() const { return enumerator_iter<R>(hbtl::adl_end(TheRange)); }

  private:
    R TheRange;
  };

  } // end namespace detail

  /// Given an input range, returns a new range whose values are are pair (A,B)
  /// such that A is the 0-based index of the item in the sequence, and B is
  /// the value from the original sequence.  Example:
  ///
  /// std::vector<char> Items = {'A', 'B', 'C', 'D'};
  /// for (auto X : enumerate(Items)) {
  ///   printf("Item %zu - %c\n", X.index(), X.value());
  /// }
  ///
  /// or using structured bindings:
  ///
  /// for (auto [Index, Value] : enumerate(Items)) {
  ///   printf("Item %zu - %c\n", Index, Value);
  /// }
  ///
  /// Output:
  ///   Item 0 - A
  ///   Item 1 - B
  ///   Item 2 - C
  ///   Item 3 - D
  ///
  template <typename R> detail::enumerator<R> enumerate(R && TheRange) {
    return detail::enumerator<R>(std::forward<R>(TheRange));
  }

  namespace detail {
  template <typename F, typename Tuple, std::size_t... I>
  decltype(auto) apply_tuple_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
    return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
  }

  } // end namespace detail

  /// Given an input tuple (a1, a2, ..., an), pass the arguments of the
  /// tuple variadically to f as if by calling f(a1, a2, ..., an) and
  /// return the result.
  template <typename F, typename Tuple> decltype(auto) apply_tuple(F && f, Tuple && t) {
    using Indices = std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>;

    return detail::apply_tuple_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
  }

  namespace detail {

  template <typename Predicate, typename... Args> bool all_of_zip_predicate_first(Predicate &&P, Args &&...args) {
    auto z = zip(args...);
    auto it = z.begin();
    auto end = z.end();
    while (it != end) {
      if (!apply_tuple([&](auto &&...args) { return P(args...); }, *it))
        return false;
      ++it;
    }
    return it.all_equals(end);
  }

  // Just an adaptor to switch the order of argument and have the predicate before
  // the zipped inputs.
  template <typename... ArgsThenPredicate, size_t... InputIndexes>
  bool all_of_zip_predicate_last(std::tuple<ArgsThenPredicate...> argsThenPredicate,
                                 std::index_sequence<InputIndexes...>) {
    auto constexpr OutputIndex = std::tuple_size<decltype(argsThenPredicate)>::value - 1;
    return all_of_zip_predicate_first(std::get<OutputIndex>(argsThenPredicate),
                                      std::get<InputIndexes>(argsThenPredicate)...);
  }

  } // end namespace detail

  /// Compare two zipped ranges using the provided predicate (as last argument).
  /// Return true if all elements satisfy the predicate and false otherwise.
  //  Return false if the zipped iterator aren't all at end (size mismatch).
  template <typename... ArgsAndPredicate> bool all_of_zip(ArgsAndPredicate && ...argsAndPredicate) {
    return detail::all_of_zip_predicate_last(std::forward_as_tuple(argsAndPredicate...),
                                             std::make_index_sequence<sizeof...(argsAndPredicate) - 1>{});
  }

  /// Return true if the sequence [Begin, End) has exactly N items. Runs in O(N)
  /// time. Not meant for use with random-access iterators.
  /// Can optionally take a predicate to filter lazily some items.
  template <typename IterTy, typename Pred = bool (*)(const decltype(*std::declval<IterTy>()) &)>
  bool hasNItems(
      IterTy && Begin, IterTy && End, unsigned N,
      Pred &&ShouldBeCounted = [](const decltype(*std::declval<IterTy>()) &) { return true; },
      std::enable_if_t<!std::is_base_of<std::random_access_iterator_tag,
                                        typename std::iterator_traits<
                                            std::remove_reference_t<decltype(Begin)>>::iterator_category>::value,
                       void> * = nullptr) {
    for (; N; ++Begin) {
      if (Begin == End)
        return false; // Too few.
      N -= ShouldBeCounted(*Begin);
    }
    for (; Begin != End; ++Begin)
      if (ShouldBeCounted(*Begin))
        return false; // Too many.
    return true;
  }

  /// Return true if the sequence [Begin, End) has N or more items. Runs in O(N)
  /// time. Not meant for use with random-access iterators.
  /// Can optionally take a predicate to lazily filter some items.
  template <typename IterTy, typename Pred = bool (*)(const decltype(*std::declval<IterTy>()) &)>
  bool hasNItemsOrMore(
      IterTy && Begin, IterTy && End, unsigned N,
      Pred &&ShouldBeCounted = [](const decltype(*std::declval<IterTy>()) &) { return true; },
      std::enable_if_t<!std::is_base_of<std::random_access_iterator_tag,
                                        typename std::iterator_traits<
                                            std::remove_reference_t<decltype(Begin)>>::iterator_category>::value,
                       void> * = nullptr) {
    for (; N; ++Begin) {
      if (Begin == End)
        return false; // Too few.
      N -= ShouldBeCounted(*Begin);
    }
    return true;
  }

  /// Returns true if the sequence [Begin, End) has N or less items. Can
  /// optionally take a predicate to lazily filter some items.
  template <typename IterTy, typename Pred = bool (*)(const decltype(*std::declval<IterTy>()) &)>
  bool hasNItemsOrLess(
      IterTy && Begin, IterTy && End, unsigned N,
      Pred &&ShouldBeCounted = [](const decltype(*std::declval<IterTy>()) &) { return true; }) {
    assert(N != std::numeric_limits<unsigned>::max());
    return !hasNItemsOrMore(Begin, End, N + 1, ShouldBeCounted);
  }

  /// Returns true if the given container has exactly N items
  template <typename ContainerTy> bool hasNItems(ContainerTy && C, unsigned N) {
    return hasNItems(std::begin(C), std::end(C), N);
  }

  /// Returns true if the given container has N or more items
  template <typename ContainerTy> bool hasNItemsOrMore(ContainerTy && C, unsigned N) {
    return hasNItemsOrMore(std::begin(C), std::end(C), N);
  }

  /// Returns true if the given container has N or less items
  template <typename ContainerTy> bool hasNItemsOrLess(ContainerTy && C, unsigned N) {
    return hasNItemsOrLess(std::begin(C), std::end(C), N);
  }

  /// Returns a raw pointer that represents the same address as the argument.
  ///
  /// This implementation can be removed once we move to C++20 where it's defined
  /// as std::to_address().
  ///
  /// The std::pointer_traits<>::to_address(p) variations of these overloads has
  /// not been implemented.
  template <class Ptr> auto to_address(const Ptr &P) { return P.operator->(); }
  template <class T> constexpr T *to_address(T * P) { return P; }
}
HBTL_NAMESPACE_END
