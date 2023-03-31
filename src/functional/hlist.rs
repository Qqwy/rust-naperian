use core::marker::PhantomData;
use core::ops::Add;

pub trait HList {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HNil;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HCons<H, T>(PhantomData<(H, T)>);

impl HList for HNil {}
impl<H, T: HList> HList for HCons<H, T> {}

#[macro_export]
macro_rules! HList {
    () => { $crate::functional::hlist::HNil };
    (...$Rest:ty) => { $Rest };
    ($A:ty) => { HList![$A,] };
    ($A:ty, $($tok:tt)*) => {
        $crate::functional::hlist::HCons<$A, HList![$($tok)*]>
    };
}

/// Type-level 'function' to return the first element of a HList
///
/// Only implemented for non-empty HLists.
pub type First<List> = <List as HFirst>::Output;
/// Implementation of [`First`].
pub trait HFirst: HList {
    type Output;
}

impl<H, Ts> HFirst for HCons<H, Ts>
where
    Ts: HList
{
    type Output = H;
}

/// Type-level 'function' to return the first element of a HList
///
/// Only implemented for non-empty HLists.
pub type Rest<List> = <List as HRest>::Output;
/// Implementation of [`First`].
pub trait HRest: HList {
    type Output;
}

impl<H, Ts> HRest for HCons<H, Ts>
where
    Ts: HList
{
    type Output = Ts;
}

/// Type-level 'function' to return the all elements but the last element of a HList
///
/// Only implemented for non-empty HLists.
pub type Last<List> = <List as HLast>::Output;
/// Implementation of [`Last`].
pub trait HLast: HList {
    type Output;
}

impl<H> HLast for HCons<H, HNil> {
    type Output = H;
}

impl<H, Ts> HLast for HCons<H, Ts>
    where
    Ts: HLast,
{
    type Output = Ts::Output;
}


/// Type-level 'function' to return the all elements but the last element of a HList
///
/// Only implemented for non-empty HLists.
pub type Inits<List> = <List as HInits>::Output;
/// Implementation of [`Inits`].
pub trait HInits: HList {
    type Output: HList;
}

impl<H> HInits for HCons<H, HNil> {
    type Output = HNil;
}

impl<H, Ts> HInits for HCons<H, Ts>
where
    Ts: HInits,
{
    type Output = HCons<H, Ts::Output>;
}


/// Type-level 'function' to concatenate two HLists.
pub type Concat<Lhs, Rhs> = <Lhs as HConcat<Rhs>>::Output;

pub trait HConcat<Rhs: HList>: HList
{
    type Output: HList;
}

impl<Rhs: HList> HConcat<Rhs> for HNil {
    type Output = Rhs;
}

impl<H, T, Rhs: HList> HConcat<Rhs> for HCons<H, T>
where
    T: HConcat<Rhs>,
{
    type Output = HCons<H, Concat<T, Rhs>>;
}

/// Type-level 'function' to reverse a HList.
pub type Reverse<List> = <List as HReverse>::Output;

/// Implementation of [`Reverse`].
pub trait HReverse {
    type Output: HList;
}

impl HReverse for HNil {
    type Output = HNil;
}

impl<H, T> HReverse for HCons<H, T>
    where
    T: HConcat<HCons<H, HNil>> + HReverse,
    Reverse<T>: HConcat<HCons<H, HNil>>,
{
    type Output = Concat<Reverse<T>, HCons<H, HNil>>;
}

use typenum::{Bit, Unsigned, Add1, B0, B1};
use typenum::consts::U0;
/// Type-level 'function' to calculate the length of a HList.
///
/// You can turn the result into a `usize` using `Len<List>::USIZE` or `Len<List>::to_usize()`.
///
/// (See [`typenum::Unsigned`].)
pub type Len<List> = <List as HLen>::Output;
pub trait HLen {
    type Output: Unsigned;
}

impl HLen for HNil {
    type Output = U0;
}

impl<H, T> HLen for HCons<H, T>
where
    T: HLen,
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
pub type IsEmpty<List> = <List as HIsEmpty>::Output;
pub trait HIsEmpty {
    type Output: Bit;
}

impl HIsEmpty for HNil {
    type Output = B1;
}

impl<H, T> HIsEmpty for HCons<H, T> {
    type Output = B0;
}



#[cfg(test)]
pub mod tests {
    // Since all of this is type-level code,
    // these tests run at compile-time.
    use super::*;
    use static_assertions::assert_type_eq_all;
    use typenum::consts::*;
    // First:
    assert_type_eq_all!(U1, First<HList![U1, U2]>);

    // Rest:
    assert_type_eq_all!(HList![U2], Rest<HList![U1, U2]>);

    // Last:
    assert_type_eq_all!(U2, Last<HList![U1, U2]>);

    // Inits:
    assert_type_eq_all!(HList![U1, U2], Inits<HList![U1, U2, U3]>);

    // Concat:
    assert_type_eq_all!(HList![U1, U2, U3], Concat<HList![U1], HList![U2, U3]>);

    // Reverse:
    assert_type_eq_all!(HCons<U3, HCons<U2, HCons<U1, HNil>>>, Reverse<HCons<U1, HCons<U2, HCons<U3, HNil>>>>);

    // Len:
    assert_type_eq_all!(U0, Len<HList![]>);
    assert_type_eq_all!(U1, Len<HList![usize]>);
    assert_type_eq_all!(U2, Len<HList![i32, usize]>);

    // IsEmpty:
    assert_type_eq_all!(B1, IsEmpty<HList![]>);
    assert_type_eq_all!(B0, IsEmpty<HList![i32]>);
}
