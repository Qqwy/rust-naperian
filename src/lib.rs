#![feature(type_name_of_val)]

pub mod common;
pub mod fin;
pub mod functional;
pub mod align;

use common::Array;
pub use functional::{Container, Mappable, Mappable2, Mappable3, New, NewFrom, Apply, Naperian};
use functional::pair::Pair;
use align::{Maxed, align2, Align};
use std::marker::PhantomData;

use fin::Fin;

use generic_array::sequence::{GenericSequence, Lengthen, Shorten};
use generic_array::{arr, ArrayLength, GenericArray};
use typenum::consts::*;
use typenum::operator_aliases::{Add1, Prod, Sub1};
use typenum::U;
use typenum::{NonZero, Unsigned};


/// Transpose a F<G<A>> into G<F<A>> provided both F and G implement [`Naperian`].
/// (and A: [`Clone`] since we need to copy a bunch of A's around.)
///
/// There is no need to implement this trait manually since there is a blanket implementation
/// for all types implementing Naperian.
pub trait NaperianTranspose<G, A: Clone>
where
    Self: Naperian<G>,
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{
    fn transpose(&self) -> G::Containing<Self::Containing<A>> {
        Naperian::tabulate(|x| Naperian::tabulate(|y| self.lookup(y).lookup(x).clone()))
    }
}

impl<G, A: Clone, Nap: ?Sized + Naperian<G>> NaperianTranspose<G, A> for Nap
where
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{
}

/// Anything that is a Dimension can be used one one rank of a hypercuboid.
///
///
/// Conceptually Dimension is a supertrait of [`Mappable`], [`Apply`], [`New`], [`Mappable2`], [`Mappable3`], [`IntoIterator`] and  [`Traversable`].
/// But making it a 'true' supertrait would require a very large number of (usually unconstrained and therefore un-inferrable) type parameters.
/// Instead, the other bounds are introduced only for the particular operations where they are required.
pub trait Dimension: Container {
    fn size(&self) -> usize;
}

impl<T> Dimension for Pair<T> {
    fn size(&self) -> usize {
        2
    }
}

impl<T, N: ArrayLength> Dimension for Array<T, N> {
    fn size(&self) -> usize {
        N::USIZE
    }
}

/// Version of innerp precisely following the Naperian paper.
///
/// This is evaluated more strictly than desired; it will first create an intermediate container
/// with all the products, and then sum the elements in this container.
pub fn innerp_orig<A, R>(a: &A, b: &A) -> R
where
    A: Mappable2<R, R> + Container<Containing<R> = A> + IntoIterator,
    R: std::iter::Sum<<A as std::iter::IntoIterator>::Item>,
    R: core::ops::Mul<R, Output = R> + Clone,
{
    let products = a.map2(b, |x: &R, y: &R| x.clone() * y.clone());

    products.into_iter().sum()
}

/// Calculate the inner product of two containers of the same shape.
///
/// The inner product is the sum of multiplying all elements pairwise.
pub fn innerp<'a, A, R: 'a>(a: &'a A, b: &'a A) -> R
where
    &'a A: IntoIterator<Item = &'a R>,
    A: Container<Containing<R> = A> + IntoIterator,
    &'a R: core::ops::Mul<&'a R, Output = R>,
    R: core::iter::Sum,
{
    a.into_iter().zip(b.into_iter()).map(|(x, y)| x * y).sum()
}

/// Calculates the matrix product of two matrices with the same element type `A`.
/// Given a f×g matrix and a g×h matrix, returns a f×h matrix.
///
/// This is implemented by first transforming both matrices to a common f×h×g representation,
/// and then mapping the inner product ([`innerp_orig`]) across the innermost g dimension to flatten it.
///
/// Compatibility of dimensions is fully determined at compile time.
// NOTE currently this uses innerp_orig as I could not get it working with the lifetime requirements of innerp.
// This could be improved in the future, for slightly less stack space usage.
pub fn matrixp<Fhga, Fha, Fga, Gha, Hga, Fa, Ga, Ha, A>(xss: &Fga, yss: &Gha) -> Fha
where
    Fhga: Container<Elem = Hga, Containing<Hga> = Fhga> + Container<Containing<Ha> = Fha>,
    Fhga: New<Fa::Containing<Gha::Containing<A>>> + Mappable2<Hga, Ha>,
    Fga: Container<Elem = Ga, Containing<Ga> = Fga> + Container<Containing<Hga> = Fhga>,
    Fga: Clone + Mappable<Hga>,
    Hga: Container<Containing<Ga> = Hga> + Container<Containing<A> = Ha>,
    Hga: New<Ga> + Mappable2<Ga, A>,
    Ga: Container<Elem = A, Containing<A> = Ga>,
    Ga: IntoIterator<Item = A> + Mappable2<A, A> + Naperian<A>,
    Gha: Container<Elem = Fa> + Container<Containing<A> = Ga>,
    Gha: Naperian<Fa, Log = Ga::Log>,
    Fa: Container<Elem = A, Containing<A> = Fa>,
    Fa: Naperian<A>,
    Fa::Containing<Gha::Containing<A>>: Naperian<Gha::Containing<A>, Log = Fa::Log>,
    A: Clone + std::iter::Sum + std::ops::Mul<Output = A>,
{
    let lifted_xss: Fhga = lifted_xss(xss);
    let lifted_yss: Fhga = lifted_yss(yss);
    lifted_xss.map2(&lifted_yss, |left: &Hga, right: &Hga| {
        left.map2(right, innerp_orig)
    })
}

fn lifted_xss<F, G, H>(xss: &F) -> F::Containing<H>
where
    F: Clone + Mappable<H> + Container<Elem = G>,
    G: Container,
    H: New<G> + Container,
{
    xss.clone().map_by_value(New::new)
}

fn lifted_yss<A, F, G, H>(yss: &F) -> H
where
    F: NaperianTranspose<G, A>,
    F::Containing<A>: Naperian<A, Log = F::Log>,
    G: Naperian<A>,
    G::Containing<F::Containing<A>>: Naperian<F::Containing<A>, Log = G::Log>,
    H: New<G::Containing<F::Containing<A>>>,
    A: Clone,
{
    New::new(yss.transpose())
}

use frunk::hlist::{HCons, HList, HNil};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
#[repr(transparent)]
pub struct Scalar<T>(T);
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
pub struct Prism<T, Ts, N, Ns>(Ts, core::marker::PhantomData<(T, N, Ns)>)
where
    N: ArrayLength + NonZero,
    // Ts: Hyper<Dimensions = Ns>,
    Ns: HList;

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

    fn dim(&self) -> usize;

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

    fn dim(&self) -> usize {
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

    fn dim(&self) -> usize {
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
        &self.0.inner()
    }

    fn dimensions() -> Array<usize, Self::Rank> {
        Ts::dimensions().append(N::USIZE)
    }

    fn hreplicate(elem: Self::Elem) -> Self {
        // todo!()
        // Prism::new(elem)
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

// TODO might be incorrect
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

// TODO might be incorrect
// impl<T, Ts, N, Ns, A> Mappable<A> for Prism<T, Ts, N, Ns>
// where
//     N: ArrayLength + NonZero,
//     Ts: Hyper<Dimensions = Ns> + Mappable<A> + Container<Elem=T>,
//     Ts::Containing<A>: Hyper<Dimensions = Ns>,
//     Ns: HList,
// {
//     fn map(&self, fun: impl FnMut(&Self::Elem) -> A) -> Self::Containing<A> {
//         let res = self.0.map(fun);// .map(|inner| inner.map(&mut fun));
//         Prism(res, core::marker::PhantomData)
//     }

//     fn map_by_value(self, fun: impl FnMut(Self::Elem) -> A) -> Self::Containing<A> {
//         let res = self.0.map_by_value(fun);// map_by_value(|inner| inner.map_by_value(&mut fun));
//         Prism(res, core::marker::PhantomData)
//     }
// }

// TODO might be incorrect
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
        let new_ts = self.0.map2(&rhs.0, |self_arr, rhs_arr| {
            self_arr.map2(rhs_arr, &mut fun)
        });
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
    AsAligned: Hyper<Elem = A> + Container<Containing<B> = BsAligned> + Mappable2<A, C> + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = AsAligned::AmountOfElems> + Container<Containing<B> = BsAligned>,
{
    let (mleft, mright) = align2(left, right);
    mleft.map2_by_value(mright, fun)
}

pub trait AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C>
    where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A> + Container<Containing<B> = BsAligned> + Mappable2<A, C> + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems> + Container<Containing<B> = BsAligned>,

{
    /// Neccessarily only works by value because the two tensors need to be aligned.
    fn map2(self, right: Bs, fun: impl FnMut(A, B) -> C) -> Cs {
        let (mself, mright) = align2(self, right);
        mself.map2_by_value(mright, fun)
    }
}

impl<Bs, SelfAligned, BsAligned, Cs, A, B, C> AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C> for Scalar<A>
where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A> + Container<Containing<B> = BsAligned> + Mappable2<A, C> + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems> + Container<Containing<B> = BsAligned>,
{}


impl<Bs, SelfAligned, BsAligned, Cs, A, B, C, Ts, N, Ns> AutoMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C> for Prism<A, Ts, N, Ns>
where
    Self: Hyper<Elem = A> + Maxed<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + Maxed<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A> + Container<Containing<B> = BsAligned> + Mappable2<A, C> + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems> + Container<Containing<B> = BsAligned>,
    N: ArrayLength + NonZero,
    Ns: HList,
{}


pub mod aliases {
    use super::{Array, HCons, HNil, Prism, Scalar};

    /// Vect, a Vector with a statically-known size.
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`Hyper`] trait which contains many common operations.
    pub type Vect<T, N> = Prism<T, Scalar<Array<T, N>>, N, HNil>;

    /// Mat, a Matrix with a statically-known dimensions (rows, colums).
    ///
    /// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`Hyper`] trait which contains many common operations.
    pub type Mat<T, Rows, Cols> = Prism<T, Vect<Array<T, Cols>, Rows>, Cols, HCons<Rows, HNil>>;

    /// Rank-3 tensors (slices, rows, columns).
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`Hyper`] trait which contains many common operations.
    pub type Tensor3<T, Slices, Rows, Cols> =
        Prism<T, Mat<Array<T, Cols>, Slices, Rows>, Cols, HCons<Rows, HCons<Slices, HNil>>>;
}

pub mod const_aliases {
    use super::{Array, HCons, HNil, Prism, Scalar};
    use typenum::U;

    /// Vect, a Vector with a statically-known size.
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`super::Hyper`] trait which contains many common operations.
    pub type Vect<T, const N: usize> = Prism<T, Scalar<Array<T, U<N>>>, U<N>, HNil>;

    /// Mat, a Matrix with a statically-known dimensions (rows, colums).
    ///
    /// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`super::Hyper`] trait which contains many common operations.
    pub type Mat<T, const ROWS: usize, const COLS: usize> =
        Prism<T, Vect<Array<T, U<COLS>>, ROWS>, U<COLS>, HCons<U<ROWS>, HNil>>;

    /// Vect, a Vector with a statically-known size.
    ///
    /// This is a type alias.
    /// During normal usage you do not need to understand the backing type,
    /// only that it implements the [`super::Hyper`] trait which contains many common operations.
    pub type Tensor3<T, const SLICES: usize, const ROWS: usize, const COLS: usize> = Prism<
        T,
        Mat<Array<T, U<COLS>>, SLICES, ROWS>,
        U<COLS>,
        HCons<U<ROWS>, HCons<U<SLICES>, HNil>>,
    >;
}
pub use const_aliases::*;

pub fn foo() -> Tensor3<usize, 2, 2, 3> {
    let v: Vect<usize, 3> = Prism::build(Scalar::new(arr![1, 2, 3]));
    let mat: Mat<usize, 2, 3> = Prism::build(Prism::build(Scalar::new(arr![
        arr![1, 2, 3],
        arr![4, 5, 6]
    ])));
    // let tens: Tensor3<usize, U2, U3, U1> = Prism::build(Prism::build(Prism::build(Scalar::new(arr![arr![arr![1,2,3], arr![4,5,6]]]))));
    let tens: Tensor3<usize, 2, 2, 3> =
        Prism::build(Prism::build(Prism::build(Scalar::new(arr![
            arr![arr![1, 2, 3], arr![4, 5, 6]],
            arr![arr![7, 8, 9], arr![10, 11, 12]]
        ]))));
    println!("{:?}", &mat);
    println!("{:?}", &tens);
    let flat: &Array<usize, U12> = unsafe { core::mem::transmute(&tens) };
    println!("flat: {:?}", &flat);
    // println!("{:?}", core::any::type_name_of_val(&tens));
    tens
}

pub fn alignment() {
    let v: Vect<usize, 3> = Prism::build(Scalar::new(arr![1, 2, 3]));
    let mat: Mat<usize, 2, 3> = Prism::build(Prism::build(Scalar::new(arr![
        arr![1, 2, 3],
        arr![4, 5, 6]
    ])));
    let mat2: Tensor3<usize, 3, 2, 3> = mat.align();
    // println!("mat: {:?}", mat);
    println!("mat2: {:?}", mat2);
}


// pub type Mat<T, Rows, Cols> = Prism<T  , Vect<Array<T, Rows>, Cols>, Cols, HCons<Rows, HNil>>;
// pub type Mat<T, R, C> = Prism<T, Prism<Array<T, C>, Scalar<Array<Array<T, C>, R>>, R, HNil>, C, HCons<R, HNil>>;
//                      Prism<T, Prism<Array<T, C>, Scalar<Array<Array<T, C>, R>>, R, HNil>, C, HCons<R, HNil>>
//                            Prism<i32, Prism<Array<i32, U3>, Scalar<Array<Array<i32, U3>, U2>>, U2, HNil>, U3, HCons<U2, HNil>>

// Prism<i32, Prism<Array<i32, UInt<UInt<UTerm, B1>, B1>>, Scalar<Array<Array<i32, UInt<UInt<UTerm, B1>, B1>>, UInt<UInt<UTerm, B1>, B0>>>, UInt<UInt<UTerm, B1>, B0>, HNil>, UInt<UInt<UTerm, B1>, B1>, HCons<UInt<UInt<UTerm, B1>, B0>, HNil>>

// Prism<T, Prism<Array<T, U3, Prism<Array<Array<T, U3, U2, Scalar<Array<Array<Array<T, U3, U2, U1>, U1, HNil>, U2, HCons<U1, HNil>>, U3, HCons<U2, HCons<U1, HNil>>>
// Prism<T, Prism<Array<T, C, Prism<Array<Array<T, C, R, Scalar<Array<Array<Array<T, C, R, S>, S, HNil>, R, HCons<S, HNil>>, C, HCons<R, HCons<S, HNil>>>

// Prism<i32, Prism<Array<i32, UInt<UInt<UTerm, B1>, B1>>, Prism<Array<Array<i32, UInt<UInt<UTerm, B1>, B1>>, UInt<UInt<UTerm, B1>, B0>>, Scalar<Array<Array<Array<i32, UInt<UInt<UTerm, B1>, B1>>, UInt<UInt<UTerm, B1>, B0>>, UInt<UTerm, B1>>>, UInt<UTerm, B1>, HNil>, UInt<UInt<UTerm, B1>, B0>, HCons<UInt<UTerm, B1>, HNil>>, UInt<UInt<UTerm, B1>, B1>, HCons<UInt<UInt<UTerm, B1>, B0>, HCons<UInt<UTerm, B1>, HNil>>>

// Prism<T, Prism<Array<T, U3>, Prism<Array<Array<T, U3>, U2>, Scalar<Array<Array<Array<T, U3>, U2>, U1>>, U1, HNil>, U2, HCons<U1, HNil>>, U3, HCons<U2, HCons<U1, HNil>>>

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary() {
        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let vec = Vect::<usize, 3>::from_flat(arr![10, 20, 30]);
        let res = mat.map2(vec, |x, y| x + y);
        println!("{:?}", res);

        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let matsum = (&mat).map2(&mat, |x, y| x + y);
        println!("matsum: {:?}", matsum);
    }

    #[test]
    fn alignment() {
        super::alignment();
    }

    #[test]
    fn foofoo() {
        let val = super::foo();
    }

    #[test]
    fn transpose() {
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let three_by_two: Array<GenericArray<_, _>, U2> = arr![v123, v456];
        println!("{three_by_two:?}");
        let two_by_three = three_by_two.transpose();
        println!("{two_by_three:?}");
        assert_eq!(two_by_three, arr![arr![1, 4], arr![2, 5], arr![3, 6]]);

        let pair_of_vecs = Pair(arr![1, 2, 3], arr![10, 20, 30]);
        let vec_of_pairs = pair_of_vecs.transpose();
        println!("{vec_of_pairs:?}");
        assert_eq!(vec_of_pairs, arr![Pair(1, 10), Pair(2, 20), Pair(3, 30)]);
    }

    // pub fn increase<'a>(m: &'a usize) -> State<'a, usize, usize>{
    //     State { run_state: Box::new(move |n| (m + n, m + n))}
    // }

    #[test]
    fn traversable() {
        let _pair = Pair(10, 20);
        // let res = pair.traverse(increase);
        // println!("{:?}", pair);
        // let pair_of_vecs = Pair(arr![1,2,3], arr![10, 20, 30]);
        // let transposed = pair_of_vecs.transpose();
        // println!("{:?}", transposed);
        // increase m = State (λn → (m + n, m + n))
    }

    #[test]
    fn innerprod() {
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let res = innerp(&v123, &v456);
        println!("{res:?}");
    }

    #[test]
    fn matrixprod() {
        let v123: Array<usize, _> = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let three_by_two: Array<GenericArray<_, _>, U2> = arr![v123, v456];
        let two_by_three = three_by_two.transpose();
        let res = matrixp(&two_by_three, &three_by_two);
        println!("{res:?}");
        assert_eq!(
            res,
            arr![arr![17, 22, 27], arr![22, 29, 36], arr![27, 36, 45]]
        );
    }

    #[test]
    fn hyper_map() {
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let two_by_three: Array<GenericArray<usize, _>, _> = arr![v123, v456];
        // let val = Prism::new(Scalar::new(two_by_three));
        let val = Scalar::new(two_by_three);
        println!("{:?}", val);
        println!("{:?}", val.dim());
        println!("{:?}", val.amount_of_elems());
        let val = Prism::build(Scalar::new(two_by_three));
        println!("{:?}", val);
        println!("{:?}", val.dim());
        println!("{:?}", val.amount_of_elems());
        let val = Prism::build(Prism::build(Scalar::new(two_by_three)));
        let first = val.first();
        println!("{:?}", val);
        println!("{:?}", val.dim());
        println!("{:?}", val.amount_of_elems());
        println!("{:?}", first);

        let xs = Prism::hreplicate(42usize);
        println!("xs: {:?}", xs);
        assert!(xs > val);
    }

    // #[test]
    // fn hyper_first() {
    //     super::hyper_first();
    // }

    #[test]
    fn flattening() {
        let flat = arr![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        println!("{:?}", &flat);
        let tens = Tensor3::<usize, 2, 2, 3>::from_flat(flat);
        println!("{:?}", &tens);

        let four_by_three: Mat<usize, 4, 3> = tens.reshape();
        println!("{:?}", &four_by_three);
        let three_by_four: Mat<usize, 3, 4> = four_by_three.reshape();
        println!("{:?}", &three_by_four);

        let flat2 = arr!["hello", "world", "how", "are", "you", "doing"];
        let mat = Mat::<&'static str, 3, 2>::from_flat(flat2);
        println!("{:?}", &mat);
    }
}

pub fn reshape_example(flat: Array<usize, U<20>>) -> Tensor3<usize, 2, 2, 5> {
    // let flat = arr![1,2,3,4,5,6,7,8,9,10,11,12];
    let tens = Tensor3::<usize, 2, 2, 5>::from_flat(flat);
    tens
    // println!("{:?}", &tens);
    // let vec: MatC<usize, 6, 2> = tens.reshape();

    // let four_by_three: Mat<usize, U4, U2> = tens.reshape();
    // println!("{:?}", &four_by_three);
    // let three_by_four: Mat<usize, U3, U4> = four_by_three.reshape();
    // println!("{:?}", &three_by_four);
}

pub fn matrixprod(
    two_by_three: Array<GenericArray<usize, U2>, U3>,
    three_by_two: Array<GenericArray<usize, U3>, U2>,
) -> Array<GenericArray<usize, U3>, U3> {
    matrixp(&two_by_three, &three_by_two)
}

pub fn hyper_first(v123: Array<usize, U3>, v456: GenericArray<usize, U3>) -> usize {
    // let v123 = arr![1, 2, 3];
    // let v456 = arr![4, 5, 6];
    let two_by_three: Array<GenericArray<usize, _>, _> = arr![v123, v456];
    let val = Prism::build(Prism::build(Scalar::new(two_by_three)));
    *val.first()
}

// pub fn innerprod(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
//     // use generic_array::arr;
//     // let v123 = arr![1,2,3];
//     // let v456 = arr![4,5,6];
//     innerp(&v123, &v456)
// }

// // pub fn innerprod_orig() {
// pub fn innerprod_orig(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
//     // use generic_array::arr;
//     // let v123 = arr![1,2,3];
//     // let v456 = arr![4,5,6];
//     innerp_orig(&v123, &v456)
// }
