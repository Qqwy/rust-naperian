use crate::align::{align2, Maxed};
use crate::common::Array;
pub use crate::const_aliases::*;
pub use crate::functional::{
    Apply, Container, Mappable, Mappable2, Mappable3, Naperian, New, NewFrom,
};

use std::marker::PhantomData;

pub use crate::fin::Fin;

use generic_array::sequence::{Lengthen, Shorten};
use generic_array::{arr, ArrayLength, GenericArray};
use typenum::consts::*;
use typenum::operator_aliases::{Add1, Prod, Sub1};

use frunk::hlist::{HCons, HList, HNil};
use typenum::{NonZero, Unsigned};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
#[repr(transparent)]
pub struct Scalar<T>(pub(crate) T);
unsafe impl<T> Container for Scalar<T> {
    type Elem = T;
    type Containing<X> = Scalar<X>;
}

impl<T> New<T> for Scalar<T> {
    fn new(val: T) -> Self {
        Scalar::hreplicate(val)
    }
}

impl<T, Ts, N, Ns> New<T> for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: HList,
    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>> + Container,
    Ts::AmountOfElems: core::ops::Mul<N>,
    Prod<Ts::AmountOfElems, N>: ArrayLength,
    Ts::Orig: core::fmt::Debug,
    Ts::Rank: core::ops::Add<B1>,
    Add1<Ts::Rank>: ArrayLength,
    Ts::Rank: core::ops::Add<B1> + ArrayLength,
    Add1<Ts::Rank>: ArrayLength + core::ops::Sub<B1, Output = Ts::Rank>,
    Sub1<Add1<Ts::Rank>>: ArrayLength,
    Array<usize, Ts::Rank>: Lengthen<usize, Longer = GenericArray<usize, Add1<Ts::Rank>>>,
    T: Clone,
{
    fn new(elem_val: T) -> Self {
        Prism::hreplicate(elem_val)
    }
}

/// Conceptually, Ts is restricted to itself be a Hyper<Dimensions = Ns>
/// but putting this restriction on the struct makes it impossible to implement certain traits.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
#[repr(transparent)]
pub struct Prism<T, Ts, N, Ns>(
    pub(crate) Ts,
    pub(crate) core::marker::PhantomData<(T, N, Ns)>,
);
// where
// N: ArrayLength + NonZero,
// Ts: Hyper<Dimensions = Ns>,
// Ns: HList;

impl<T, Ts, N, Ns> core::fmt::Debug for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: HList,
    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>>,
    Ts::AmountOfElems: core::ops::Mul<N>,
    Prod<Ts::AmountOfElems, N>: ArrayLength,
    Ts::Orig: core::fmt::Debug,
    Ts::Rank: core::ops::Add<B1>,
    Add1<Ts::Rank>: ArrayLength,

    Ts::Rank: core::ops::Add<B1> + ArrayLength,
    Add1<Ts::Rank>: ArrayLength + core::ops::Sub<B1, Output = Ts::Rank>,
    Sub1<Add1<Ts::Rank>>: ArrayLength,
    Array<usize, Ts::Rank>: Lengthen<usize, Longer = GenericArray<usize, Add1<Ts::Rank>>>,
    T: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prism")
            .field("dimensions", &Self::dimensions())
            .field("contents", &self.inner())
            .finish()
    }
}

impl<T, Ts, N, Ns> Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: HList,
    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>>,
{
    pub fn build(vals: Ts) -> Self {
        Prism(vals, core::marker::PhantomData)
    }
}

unsafe impl<T, Ts, N, Ns> Container for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ts: Hyper<Dimensions = Ns> + Container,
    Ns: HList,
{
    type Elem = T;
    type Containing<X> = Prism<X, Ts::Containing<GenericArray<X, N>>, N, Ns>;
}

pub trait Hyper: Sized {
    type Dimensions: HList;
    type Elem;
    type Inner: Hyper;
    type Orig;
    type Rank: ArrayLength;

    // fn into_inner(&self) -> &Self::Inner;
    fn inner(&self) -> &Self::Orig;

    type AmountOfElems: ArrayLength;
    fn amount_of_elems(&self) -> usize {
        Self::AmountOfElems::to_usize()
    }

    fn rank() -> usize {
        Self::Rank::USIZE
    }

    fn dimensions() -> Array<usize, Self::Rank>;

    // fn dimensions() -> Array<usize, Self::Rank> {

    // }

    fn innermost_dimension(&self) -> usize;

    fn first(&self) -> &Self::Elem;

    fn hreplicate(elem: Self::Elem) -> Self;

    /// Reinterprets the backing memory as a one-dimensional array.
    ///
    /// This is a zero-cost operation. It compiles down to a no-op.
    fn into_flat(self) -> Array<Self::Elem, Self::AmountOfElems>;

    /// Reinterprets a one-dimensional array as this Hyper shape.
    ///
    /// This is a cheap operation. Currently it requires a single move.
    fn from_flat(arr: Array<Self::Elem, Self::AmountOfElems>) -> Self;

    /// Turns one Hyper into another.
    ///
    /// Sugar over calling Other::from_flat(self.into_flat()).
    /// c.f. [`Hyper::into_flat`] and [`Hyper::from_flat`].
    ///
    /// This is a cheap operation. Currently it requires a single move.
    fn reshape<Other>(self) -> Other
    where
        Other: Hyper<Elem = Self::Elem, AmountOfElems = Self::AmountOfElems>,
    {
        Other::from_flat(self.into_flat())
    }
}

impl<T> Hyper for Scalar<T> {
    type Dimensions = HNil;
    type Elem = T;
    type Inner = Self;
    type Orig = T;
    type AmountOfElems = U1;
    type Rank = U0;
    // fn into_inner(&self) -> &T {
    //     &self.0
    // }

    fn innermost_dimension(&self) -> usize {
        1
    }
    fn first(&self) -> &T {
        &self.0
    }
    fn inner(&self) -> &T {
        &self.0
    }

    fn dimensions() -> Array<usize, Self::Rank> {
        Default::default()
    }

    fn hreplicate(elem: Self::Elem) -> Self {
        Scalar(elem)
    }
    fn into_flat(self) -> Array<Self::Elem, U1> {
        arr![self.0]
    }
    fn from_flat(arr: Array<Self::Elem, U1>) -> Self {
        let (elem, _empty_arr) = arr.pop_front();
        Scalar(elem)
    }
}

impl<T, Ts, N, Ns> Hyper for Prism<T, Ts, N, Ns>
where
    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>>,
    Ns: HList,
    N: ArrayLength + NonZero,
    Ts::AmountOfElems: core::ops::Mul<N>,
    Ts::Rank: core::ops::Add<B1>,
    Add1<Ts::Rank>: ArrayLength,
    Prod<Ts::AmountOfElems, N>: ArrayLength,

    Ts::Rank: core::ops::Add<B1> + ArrayLength,
    Add1<Ts::Rank>: ArrayLength + core::ops::Sub<B1, Output = Ts::Rank>,
    Sub1<Add1<Ts::Rank>>: ArrayLength,
    Array<usize, Ts::Rank>: Lengthen<usize, Longer = GenericArray<usize, Add1<Ts::Rank>>>,
    T: Clone,
{
    type Dimensions = HCons<N, Ns>;
    type Elem = T;
    type Inner = Ts;
    type Orig = Ts::Orig;
    type Rank = Add1<Ts::Rank>;

    type AmountOfElems = Prod<Ts::AmountOfElems, N>;

    fn innermost_dimension(&self) -> usize {
        N::USIZE
    }

    fn first(&self) -> &Self::Elem {
        let ga = self.0.first();
        // Unwrap SAFETY: N is restricted to NonZero
        // Therefore, ga is always non-empty
        // No need to muck about with unsafe; compiler is smart enough here
        let first_row = ga.first().unwrap();
        first_row
    }

    fn inner(&self) -> &Self::Orig {
        self.0.inner()
    }

    fn dimensions() -> Array<usize, Self::Rank> {
        Ts::dimensions().append(N::USIZE)
    }

    fn hreplicate(elem: Self::Elem) -> Self {
        Prism(Ts::hreplicate(New::new(elem)), core::marker::PhantomData)
    }

    fn into_flat(self) -> Array<Self::Elem, Self::AmountOfElems> {
        // SAFETY: GenericArray has the following guarantees:
        // - It stores `Self::Elem` types consecutively in memory, with proper alignment
        // - the memory layout of GenericArray<GenericArray<T, N>, M> equals GenericArray<T, N * M>
        // Furthermore, Prism and Scalar are repr(transparent) so transmuting them to GenericArray is allowed.
        //
        // Note that we cannot use transmute because the compiler is not able to see that the sizes match
        // (even though they do!) c.f. https://github.com/rust-lang/rust/issues/47966
        unsafe { core::mem::transmute_copy(&self) }
    }

    fn from_flat(arr: Array<Self::Elem, Self::AmountOfElems>) -> Self {
        // SAFETY: See into_flat
        unsafe { core::mem::transmute_copy(&arr) }
    }
}

impl<T, A> Mappable<A> for Scalar<T> {
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> A) -> Self::Containing<A> {
        Scalar(fun(&self.0))
    }
    fn map_by_value(self, mut fun: impl FnMut(Self::Elem) -> A) -> Self::Containing<A> {
        Scalar(fun(self.0))
    }
}

impl<A, U> Mappable2<A, U> for Scalar<A> {
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        mut fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U> {
        Scalar(fun(&self.0, &rhs.0))
    }

    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        mut fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U> {
        Scalar(fun(self.0, rhs.0))
    }
}

impl<T, Ts, N, Ns, A> Mappable<A> for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ts: Hyper<Dimensions = Ns> + Mappable<Array<A, N>> + Container<Elem = Array<T, N>>,
    Ts::Containing<A>: Hyper<Dimensions = Ns>,
    Ns: HList,
{
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> A) -> Self::Containing<A> {
        let res = self.0.map(|arr| arr.map(&mut fun));
        Prism(res, core::marker::PhantomData)
    }

    fn map_by_value(self, mut fun: impl FnMut(Self::Elem) -> A) -> Self::Containing<A> {
        let res = self.0.map_by_value(|arr| arr.map_by_value(&mut fun));
        Prism(res, core::marker::PhantomData)
    }
}

impl<Ts, N, Ns, A, U> Mappable2<A, U> for Prism<A, Ts, N, Ns>
where
    Self: Hyper<Elem = A>,
    Self::Containing<U>: Hyper<Elem = U, AmountOfElems = <Self as Hyper>::AmountOfElems>,
    N: ArrayLength + NonZero,
    Ts: Hyper<Dimensions = Ns> + Mappable2<Array<A, N>, Array<U, N>>,
    Ts::Containing<A>: Hyper<Dimensions = Ns>,
    Ns: HList,
{
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        mut fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U> {
        let new_ts = self
            .0
            .map2(&rhs.0, |self_arr, rhs_arr| self_arr.map2(rhs_arr, &mut fun));
        Prism(new_ts, PhantomData)
    }

    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        mut fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U> {
        let new_ts = self.0.map2_by_value(rhs.0, |self_arr, rhs_arr| {
            self_arr.map2_by_value(rhs_arr, &mut fun)
        });
        Prism(new_ts, PhantomData)
    }
}

pub fn binary<As, Bs, AsAligned, BsAligned, Cs, A, B, C>(
    left: As,
    right: Bs,
    fun: impl Fn(A, B) -> C,
) -> Cs
where
    As: Hyper<Elem = A> + Maxed<Bs, AsAligned>,
    Bs: Hyper<Elem = B> + Maxed<As, BsAligned>,
    Cs: Hyper<Elem = C>,
    AsAligned: Container<Containing<B> = BsAligned>
        + Container<Containing<C> = Cs>
        + Hyper<Elem = A>
        + Mappable2<A, C>,
    BsAligned: Container + Hyper<Elem = B, AmountOfElems = AsAligned::AmountOfElems>,
{
    let (mleft, mright) = align2(left, right);
    mleft.map2_by_value(mright, fun)
}

pub trait AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C>
where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Container<Containing<B> = BsAligned>
        + Container<Containing<C> = Cs>
        + Hyper<Elem = A>
        + Mappable2<A, C>,
    BsAligned: Container + Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems>,
{
    /// Neccessarily only works by value because the two tensors need to be aligned.
    fn map2(self, right: Bs, fun: impl FnMut(A, B) -> C) -> Cs {
        let (mself, mright) = align2(self, right);
        mself.map2_by_value(mright, fun)
    }
}

impl<Bs, SelfAligned, BsAligned, Cs, A, B, C> AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C>
    for Scalar<A>
where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A>
        + Container<Containing<B> = BsAligned>
        + Mappable2<A, C>
        + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems>
        + Container<Containing<B> = BsAligned>,
{
}

impl<Bs, SelfAligned, BsAligned, Cs, A, B, C, Ts, N, Ns>
    AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C> for Prism<A, Ts, N, Ns>
where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A>
        + Container<Containing<B> = BsAligned>
        + Mappable2<A, C>
        + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems>
        + Container<Containing<B> = BsAligned>,
    N: ArrayLength + NonZero,
    Ns: HList,
{
}
