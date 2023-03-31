//! TLists: Type-level linked lists.
//!
//! These are useful if you need to keep track of a _list_ of types inside your type,
//! and manipulate them in generic ways (like taking the first type, reversing the list, etc.)
//!
//! In naperian, they are used to talk about the dimensions of the tensors (each dimension is a type-level number).
//!
//! The easiest way to use a TList is to use the [`TList!`] macro:
//!
//! ```rust
//! use naperian::functional::tlist::*;
//! use typenum::consts::*;
//! type MyList = TList![U10, U20, U100];
//! ```
use core::marker::PhantomData;
use core::ops::Add;


pub use crate::TList;

pub trait TList {}

/// The empty TList.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TNil;

/// A non-empty TList whose first element is `H` and whose tail is the TList `T`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TCons<H, T: TList>(PhantomData<(H, T)>);

impl TList for TNil {}
impl<H, T: TList> TList for TCons<H, T> {}


#[macro_export]
// Implementation based on the frunk crate's HList! macro.
macro_rules! TList {
    () => { $crate::functional::tlist::TNil };
    (...$Rest:ty) => { $Rest };
    ($A:ty) => { $crate::TList![$A,] };
    ($A:ty, $($tok:tt)*) => {
        $crate::functional::tlist::TCons<$A, $crate::TList![$($tok)*]>
    };
}

/// Type-level 'function' to return the first element of a TList
///
/// Only implemented for non-empty TLists.
///
/// ```rust
/// use naperian::functional::tlist::*;
/// use typenum::consts::{U1, U2};
///
/// static_assertions::assert_type_eq_all!(U1, First<TList![U1, U2]>);
/// ```
pub type First<List> = <List as TFirst>::Output;
/// Implementation of [`First`].
pub trait TFirst: TList {
    type Output;
}

impl<H, Ts> TFirst for TCons<H, Ts>
where
    Ts: TList
{
    type Output = H;
}

/// Type-level 'function' to return the first element of a TList
///
/// Only implemented for non-empty TLists.
///
/// ```rust
/// use naperian::functional::tlist::*;
/// use typenum::consts::{U1, U2};
///
/// static_assertions::assert_type_eq_all!(TList![U2], Rest<TList![U1, U2]>);
/// ```
pub type Rest<List> = <List as TRest>::Output;
/// Implementation of [`First`].
pub trait TRest: TList {
    type Output;
}

impl<H, Ts> TRest for TCons<H, Ts>
where
    Ts: TList
{
    type Output = Ts;
}

/// Type-level 'function' to return the all elements but the last element of a TList
///
/// Only implemented for non-empty TLists.
pub type Last<List> = <List as TLast>::Output;
/// Implementation of [`Last`].
pub trait TLast: TList {
    type Output;
}

impl<H> TLast for TCons<H, TNil> {
    type Output = H;
}

impl<H, Ts> TLast for TCons<H, Ts>
    where
    Ts: TLast,
{
    type Output = Ts::Output;
}


/// Type-level 'function' to return the all elements but the last element of a TList
///
/// Only implemented for non-empty TLists.
pub type Inits<List> = <List as TInits>::Output;
/// Implementation of [`Inits`].
pub trait TInits: TList {
    type Output: TList;
}

impl<H> TInits for TCons<H, TNil> {
    type Output = TNil;
}

impl<H, Ts> TInits for TCons<H, Ts>
where
    Ts: TInits,
{
    type Output = TCons<H, Ts::Output>;
}


/// Type-level 'function' to concatenate two TLists.
pub type Concat<Lhs, Rhs> = <Lhs as TConcat<Rhs>>::Output;

pub trait TConcat<Rhs: TList>: TList
{
    type Output: TList;
}

impl<Rhs: TList> TConcat<Rhs> for TNil {
    type Output = Rhs;
}

impl<H, T, Rhs: TList> TConcat<Rhs> for TCons<H, T>
where
    T: TConcat<Rhs>,
{
    type Output = TCons<H, Concat<T, Rhs>>;
}

/// Type-level 'function' to reverse a TList.
pub type Reverse<List> = <List as TReverse>::Output;

/// Implementation of [`Reverse`].
pub trait TReverse {
    type Output: TList;
}

impl TReverse for TNil {
    type Output = TNil;
}

impl<H, T> TReverse for TCons<H, T>
    where
    T: TConcat<TCons<H, TNil>> + TReverse,
    Reverse<T>: TConcat<TCons<H, TNil>>,
{
    type Output = Concat<Reverse<T>, TCons<H, TNil>>;
}

use static_assertions::{assert_impl_one, assert_impl_all};
use typenum::{Bit, Unsigned, Add1, B0, B1};
use typenum::consts::U0;
/// Type-level 'function' to calculate the length of a TList.
///
/// You can turn the result into a `usize` using `Len<List>::USIZE` or `Len<List>::to_usize()`.
///
/// (See [`typenum::Unsigned`].)
pub type Len<List> = <List as TLen>::Output;
pub trait TLen {
    type Output: Unsigned;
}

impl TLen for TNil {
    type Output = U0;
}

impl<H, T: TList> TLen for TCons<H, T>
where
    T: TLen,
    Len<T>: Add<B1>,
    Add1<Len<T>>: Unsigned,
{
    type Output = Add1<Len<T>>;
}

/// Type-level 'function' returning [`B1`] when the list is empty; [`B0`] otherwise.
///
/// You can turn the result into a `bool` using `IsEmpty<List>::BOOL` or `IsEmpty<List>::to_bool()`.
///
/// (See [`typenum::Bit`].)
pub type IsEmpty<List> = <List as TIsEmpty>::Output;
pub trait TIsEmpty {
    type Output: Bit;
}

impl TIsEmpty for TNil {
    type Output = B1;
}

impl<H, T: TList> TIsEmpty for TCons<H, T> {
    type Output = B0;
}

/// Constraint which only holds if a TList is a prefix of `Other`.
///
/// This is not a 'function', but rather a constraint you can use to make compiler errors more readable.
///
/// ```rust
/// use naperian::functional::tlist::*;
/// use typenum::consts::{U1, U2, U3, U4, U42};
///
/// static_assertions::assert_impl_all!(TList![U1, U2]: Prefix<TList![U1, U2, U3, U4]>);
/// static_assertions::assert_not_impl_any!(TList![U42]: Prefix<TList![U1, U2, U3, U4]>);
/// ```
pub trait Prefix<Other: TList> {}

// prefix [] _ = true
impl<Other: TList> Prefix<Other> for TNil {}

// prefix (h : ls) (h : rs) == prefix ls rs
impl<H, Ls: TList, Rs: TList> Prefix<TCons<H, Rs>> for TCons<H, Ls>
    where
    Ls: Prefix<Rs>,
{}


pub trait Compatible<Other: TList> {}
// compatible [] [] == true
impl Compatible<TNil> for TNil {}

// compatible [] (f : gs) == true
impl<F, GS: TList> Compatible<TCons<F, GS>> for TNil {}

// compatible (f : fs) [] == true
impl<F, FS: TList> Compatible<TNil> for TCons<F, FS> {}

// compatible (f : fs) (g : gs) == true
impl<F, FS: TList, GS: TList> Compatible<TCons<F, GS>> for TCons<F, FS>
where
    FS: Compatible<GS>,
{}

#[cfg(test)]
pub mod tests {
    // Since all of this is type-level code,
    // these tests run at compile-time.
    use super::*;
    use static_assertions::assert_type_eq_all;
    use typenum::consts::*;

    // First:
    assert_type_eq_all!(U1, First<TList![U1, U2]>);

    // Rest:
    assert_type_eq_all!(TList![U2], Rest<TList![U1, U2]>);

    // Last:
    assert_type_eq_all!(U2, Last<TList![U1, U2]>);

    // Inits:
    assert_type_eq_all!(TList![U1, U2], Inits<TList![U1, U2, U3]>);

    // Concat:
    assert_type_eq_all!(TList![U1, U2, U3], Concat<TList![U1], TList![U2, U3]>);

    // Reverse:
    assert_type_eq_all!(TCons<U3, TCons<U2, TCons<U1, TNil>>>, Reverse<TCons<U1, TCons<U2, TCons<U3, TNil>>>>);

    // Len:
    assert_type_eq_all!(U0, Len<TList![]>);
    assert_type_eq_all!(U1, Len<TList![usize]>);
    assert_type_eq_all!(U2, Len<TList![i32, usize]>);

    // IsEmpty:
    assert_type_eq_all!(B1, IsEmpty<TList![]>);
    assert_type_eq_all!(B0, IsEmpty<TList![i32]>);
}


