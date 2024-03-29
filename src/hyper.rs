mod ops;

use crate::align::{align2, ShapeMatched};
use crate::common::Array;
pub use crate::const_aliases::*;
pub use crate::functional::{
    Apply, Container, Mappable, Mappable2, Mappable3, Naperian, New, NewFrom,
};

use core::marker::PhantomData;

pub use crate::fin::Fin;
use crate::functional::tlist::{TCons, TList, TNil, TReverse, TInits, Inits, Last, TLast};

use generic_array::sequence::{Lengthen, Shorten};
use generic_array::{arr, ArrayLength, GenericArray};
use typenum::consts::*;
use typenum::operator_aliases::{Add1, Prod, Sub1};

use typenum::{NonZero, Unsigned};

/// A Scalar, representing a single element of type T.
/// A rank-0 tensor.
///
/// There is no need to understand/use this type directly;
/// All useful methods are on the [`Hyper`] trait which Scalar implements.
///
/// Implementation of the 'Scalar' variant of the Hyper GADT from the Naperian paper.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
#[repr(transparent)]
pub struct Scalar<T>(pub(crate) T);

impl<T> core::fmt::Debug for Scalar<T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Scalar")
            .field(&core::any::type_name::<T>())
            .field(&self.0)
            .finish()
    }
}

// impl<T, N> Scalar<Array<T, N>>
// where
//     N: ArrayLength + NonZero,
// {
//     /// Turns a lower-dimension Hyper
//     /// whose element type is Array<T, N>
//     /// into a one-dimension-higher Hyper
//     /// whose element type is T,
//     /// with the new dimension being N.
//     ///
//     /// Inverse of [`Prism::lower`].
//     pub fn lift(self) -> Prism<T, Self, N, TNil>  {
//         Prism(self, core::marker::PhantomData)
//     }
// }

unsafe impl<T> Container for Scalar<T> {
    type Elem = T;
    type Containing<X> = Scalar<X>;
}

/// A Prism, a rank-N tensor made up of an array containing N rank-(N-1) tensors.
///
/// There is no need to understand this type directly;
/// All useful methods are on the [`Hyper`] trait which Prism implements.
///
/// # Internals
/// Here is what the internals of Prism mean:
///
/// - T: The element type
/// - Ts: A prism of Rank-(N-1) but whose element type is `Array<T, N>`.
/// - N: The innermost dimension of the tensor (which implements [`ArrayLength`]).
/// - Ns: A [`TList`] containing all other dimensions of the tensor.
///
/// Implementation of the 'Scalar' variant of the Hyper GADT from the Naperian paper.
/// Note that (at least currently) Prism will only use `Array<T, N>` as inner elements,
/// rather than 'anything that implements Dimension'.
/// So strictly speaking, this is an implementation
/// of the Prism2 variant of the Hyper2 type from the paper.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
#[repr(transparent)]
pub struct Prism<T, Ts, N, Ns>(
    pub(crate) Ts,
    pub(crate) core::marker::PhantomData<(T, N, Ns)>,
);

impl<T> New<T> for Scalar<T> {
    fn new(val: T) -> Self {
        Scalar::hreplicate(val)
    }
}

impl<T, Ts, N, Ns> New<T> for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: TList,
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

impl<T, Ts, N, Ns> core::fmt::Debug for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: TList,
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Prism")
            .field("type", &core::any::type_name::<T>())
            .field("dimensions", &Self::dimensions())
            .field("contents", &self.orig())
            .finish()
    }
}

impl<T, Ts, N, Ns> Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: TList,
    Ts: Hyper<Dimensions = Ns>,
{
    /// Lowers a rank-(K+1) Hyper
    /// into a rank-K Hyper
    /// by turning the element type from T into Array<T, N>.
    ///
    /// Inverse of [`Liftable::lift`].
    pub fn lower(self) -> Ts {
        self.0
    }
}

// impl<T, Ts, N, R, Rs> Prism<Array<T, N>, Ts, R, Rs>
// where
//     N: ArrayLength + NonZero,
// {
//     /// Turns a rank-K Hyper
//     /// whose element type is Array<T, C>
//     /// into a rank-(K+1) Hyper
//     /// whose element type is T.
//     /// The new (innermost) dimension is `C`.
//     ///
//     /// Inverse of [`Prism::lower`].
//     pub fn lift(self) -> Prism<T, Self, N, TCons<R, Rs>>  {
//         Prism(self, core::marker::PhantomData)
//     }
// }

unsafe impl<T, Ts, N, Ns> Container for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ts: Container,
    Ns: TList,
{
    type Elem = T;
    type Containing<X> = Prism<X, Ts::Containing<GenericArray<X, N>>, N, Ns>;
}

/// The meat of the crate. Hyper is implemented by any Vect, Mat, and other Tensor-like type.
///
/// The only two structs which implement [`Hyper`] are [`Scalar`] and [`Prism`].
///
/// But instead of manipulating those directly, it is nicer to use the following type aliases:
/// - [`Scalar`] for when you have a single value. <sub>(A single [`Scalar`])</sub>
/// - [`Vect`] A one-dimensional collection with fixed size. <sub>(A [`Prism`] of [`Scalar`]s)</sub>
/// - [`Mat`] A two-dimensional collection with rows and columns. <sub>(A [`Prism`] of [`Vect`]s)</sub>
/// - [`Tensor3`] A three-dimensional collection. <sub>(A [`Prism`] of [`Mat`]s)</sub>
/// - [`Tensor4`] A four-dimensional collection. <sub>(A [`Prism`] of [`Tensor3`]s)</sub>
///
/// # Implementation details
///
/// This is a Rust implementation of the the Hyper GADT from the Naperian paper.
///
///
/// Because Rust does not support GADTs, the type was turned into this trait,
/// and the two variants into the [`Scalar`] and [`Prism`] structs.
///
/// Note that (at least currently) Prism will only use `Array<T, N>` as inner elements,
/// rather than 'anything that implements Dimension'.
/// So strictly speaking, this is an implementation
/// of the 'Hyper2' type from the paper.
pub trait Hyper: Sized {
    /// A type-level list of type-level numbers representing the dimensions of this tensor.
    ///
    /// (c.f. [`typenum::Unsigned`])
    type Dimensions: TList;

    /// The element type of the structure.
    type Elem;

    /// Reference to what nested Hyper is contained in higher-rank Hypers.
    type Inner: Hyper;

    /// Reference to the innermost 'concrete' type contained.
    ///
    /// - For [`Scalar`], this is `Elem`.
    /// - For [`Vect`], this is `Array<Elem, N>`
    /// - For [`Mat`], this is `Array<Array<Elem, Cols>, Rows>`
    /// - For [`Tensor3`], this is `Array<Array<Array<Elem, Cols>, Rows>, Slices>`
    /// - ... etc.
    type Orig;

    /// Type-level number representing the rank (number of dimensions) of the Hyper.
    type Rank: ArrayLength;

    /// The total amount of elements in this Hyper.
    /// The product of [`Self::Dimensions`].
    type AmountOfElems: ArrayLength;

    // fn into_inner(&self) -> &Self::Inner;
    /// Returns a reference to the innermost form of this Hyper;
    ///
    /// c.f. [`Self::Orig`].
    fn orig(&self) -> &Self::Orig;
    fn into_orig(self) -> Self::Orig;

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
    /// Sugar over calling `Other::from_flat(self.into_flat())`.
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

pub trait Liftable {
    type Lifted;

    /// Turns a lower-dimension Hyper
    /// whose element type is Array<T, N>
    /// into a one-dimension-higher Hyper
    /// whose element type is T,
    /// with the new dimension being N.
    ///
    /// Inverse of [`Prism::lower`].
    fn lift(self) -> Self::Lifted;
}

impl<T, N> Liftable for Scalar<Array<T, N>>
where
    N: ArrayLength + NonZero,
{
    type Lifted = Prism<T, Self, N, TNil>;
    fn lift(self) -> Self::Lifted {
        Prism(self, PhantomData)
    }
}

impl<T, Ts, N, R, Rs> Liftable for Prism<Array<T, N>, Ts, R, Rs>
where
    N: ArrayLength + NonZero,
    Rs: TList,
{
    type Lifted = Prism<T, Self, N, TCons<R, Rs>>;
    fn lift(self) -> Self::Lifted {
        Prism(self, PhantomData)
    }
}

impl<T> Hyper for Scalar<T> {
    type Dimensions = TNil;
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
    fn orig(&self) -> &T {
        &self.0
    }

    fn into_orig(self) -> T {
        self.0
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
    Ns: TList,
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
    type Dimensions = TCons<N, Ns>;
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

    fn orig(&self) -> &Self::Orig {
        self.0.orig()
    }

    fn into_orig(self) -> Self::Orig {
        self.0.into_orig()
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
    Ns: TList,
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
    Ns: TList,
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

/// Map a binary (two-parameter) function over two [`Hyper`]s.
///
/// This method will work with `As` and `Bs` being Hypers of different ranks,
/// which will automatically be aligned before calling the mapping function.
///
/// Freestanding version of [`HyperMappable2::map2`].
pub fn binary<As, Bs, AsAligned, BsAligned, Cs, A, B, C>(
    left: As,
    right: Bs,
    fun: impl Fn(A, B) -> C,
) -> Cs
where
    As: Hyper<Elem = A> + ShapeMatched<Bs, AsAligned>,
    Bs: Hyper<Elem = B> + ShapeMatched<As, BsAligned>,
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

/// Map a binary (two-parameter) function over two [`Hyper`]s.
///
/// The difference between this trait and [`Mappable2`],
/// is that this trait will work with `As` and `Bs` being Hypers of different ranks,
/// which will automatically be aligned before calling the mapping function.
///
/// The advantage is flexibility.
/// The disadvantage is that [`HyperMappable2::map2`] by necessity can only work by-value.
pub trait HyperMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C>
where
    Self: Hyper<Elem = A> + ShapeMatched<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + ShapeMatched<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Container<Containing<B> = BsAligned>
        + Container<Containing<C> = Cs>
        + Hyper<Elem = A>
        + Mappable2<A, C>,
    BsAligned: Container + Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems>,
{
    /// Map a binary (two-parameter) function over two [`Hyper`]s.
    ///
    /// Neccessarily works by value because the two tensors need to be aligned,
    /// which requires copying some of their elements.
    ///
    /// Trait version of the freestanding function [`binary`].
    fn map2(self, right: Bs, fun: impl FnMut(A, B) -> C) -> Cs {
        let (mself, mright) = align2(self, right);
        mself.map2_by_value(mright, fun)
    }
}

impl<Bs, SelfAligned, BsAligned, Cs, A, B, C>
    HyperMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C> for Scalar<A>
where
    Self: Hyper<Elem = A> + ShapeMatched<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + ShapeMatched<Self, BsAligned>,
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
    HyperMappable2<Bs, SelfAligned, BsAligned, Cs, A, B, C> for Prism<A, Ts, N, Ns>
where
    Self: Hyper<Elem = A> + ShapeMatched<Bs, SelfAligned>,
    Bs: Hyper<Elem = B> + ShapeMatched<Self, BsAligned>,
    Cs: Hyper<Elem = C>,
    SelfAligned: Hyper<Elem = A>
        + Container<Containing<B> = BsAligned>
        + Mappable2<A, C>
        + Container<Containing<C> = Cs>,
    BsAligned: Hyper<Elem = B, AmountOfElems = SelfAligned::AmountOfElems>
        + Container<Containing<B> = BsAligned>,
    N: ArrayLength + NonZero,
    Ns: TList,
{
}

/// Trait implemented for every rank-2 or higher Tensor.
/// Transposes the outermost two dimensions.
///
/// So
/// - a [`Mat<T, X, Y>`] is transposed into a [`Mat<T, Y, X>`].
/// - a [`Tensor3<T, X, Y, Z>`] is transposed into a [`Tensor3<T, Y, X, Z>`].
/// - etc.
///
/// TODO possibility to map over innermore dimensions.
///
/// ```rust
/// use naperian::hyper::HyperTranspose;
/// use naperian::{Mat, Hyper};
/// use generic_array::arr;
/// let mat2x3 = Mat::<usize, 2, 3>::from_flat(arr![1,2,3,4,5,6]);
/// let mat3x2 = mat2x3.transpose();
/// println!("{:?}", mat3x2);
/// ```
pub trait HyperTranspose: Hyper {
    type Transposed;
    fn transpose(self) -> Self::Transposed;
}

use crate::NaperianTranspose;

impl<T, Tts, Tts2, N, Ns, N2, N2s> HyperTranspose
    for Prism<T, Prism<Array<T, N>, Tts, N2, N2s>, N, Ns>
where
    T: Clone,
    N: NonZero + ArrayLength,
    N2: NonZero + ArrayLength,
    Ns: TList,
    N2s: TList,
    Self: Hyper<Inner = Prism<Array<T, N>, Tts, N2, N2s>>,
    Tts: Hyper<Dimensions = N2s, Elem = Array<Array<T, N>, N2>>
        + Container<Elem = Array<Array<T, N>, N2>, Containing<Array<Array<T, N2>, N>> = Tts2>
        + Mappable<Array<Array<T, N2>, N>>,
    Tts2: Hyper<Dimensions = N2s, Elem = Array<Array<T, N2>, N>, AmountOfElems = Tts::AmountOfElems>
        + Liftable,
    Prism<Array<T, N>, Tts, N2, N2s>: Hyper<Dimensions = Ns>,
    Tts2::Lifted: Liftable<Lifted = Prism<T, Prism<Array<T, N2>, Tts2, N, N2s>, N2, TCons<N, N2s>>>,
{
    type Transposed = Prism<T, Prism<Array<T, N2>, Tts2, N, N2s>, N2, TCons<N, N2s>>;
    fn transpose(self) -> Self::Transposed {
        let inner = self.lower().lower();
        let res = inner.map(NaperianTranspose::transpose);
        res.lift().lift()
    }
}

/// A Scalar can be iterated; it yields its single contained value.
impl<T> IntoIterator for Scalar<T> {
    type Item = T;
    type IntoIter = core::iter::Once<T>;
    fn into_iter(self) -> Self::IntoIter {
        core::iter::once(self.0)
    }
}

/// Iterating over a Prism will iterate over the outermost dimension:
///
/// - for [`Vect`], iteration is over the individual elements.
/// - for [`Mat`], iteration is over the _rows_.
/// - for [`Tensor3`], iteration is over the _slices_.
/// - for [`Tensor4`], iteration is over the _blocks_.
impl<T, Ts, N, Ns> IntoIterator for Prism<T, Ts, N, Ns>
where
    Self: Hyper<Elem = T>,
    Ts: Hyper<Elem = Array<T, N>, Dimensions = Ns>,
    N: ArrayLength + NonZero,
    Ns: TList,
    Ts::Orig: IntoIterator,
{
    type Item = <<Ts as Hyper>::Orig as IntoIterator>::Item;
    type IntoIter = <<Ts as Hyper>::Orig as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.lower().into_orig().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::println;

    #[test]
    fn transpose_example() {
        let mat2x3 = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let mat3x2: Mat<usize, 3, 2> = mat2x3.transpose();
        println!("{:?}", mat3x2);
    }

    #[test]
    fn iteration() {
        let mat: Mat<usize, 2, 3> = [[1, 2, 3], [4, 5, 6]].into();
        for row in mat.clone() {
            println!("{:?}", row);
        }

        for col in mat.transpose() {
            println!("{:?}", col);
        }
    }

    #[test]
    fn index_example() {
        let mat2x3 = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let zero: Fin<_> = Fin::cnew::<0>(); // Fin::new(3).unwrap();
        println!("rows: {:?}", mat2x3.rows());
        let res = &mat2x3[zero];
        println!(" mat[0]: {:?}", res);

        println!("cols: {:?}", mat2x3.into_columns());

        let t3 = Tensor3::<usize, 2, 3, 4>::from_flat(arr![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]);
        let zero: Fin<_> = Fin::cnew::<0>(); // Fin::new(3).unwrap();
        println!("t3[0]: {:?}", t3[zero]);
        println!("t3[0]: {:?}", t3[zero][Fin::cnew::<0>()]);
        println!("t3[0][0]: {:?}", t3[zero].rows()[0]);
    }
}

use core::ops::{Deref, Index, DerefMut};
impl<T> Deref for Scalar<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Scalar<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Helper trait to modify the inner dimensions of a Hyper
pub trait WithDimensions<Dims: TList> {
    type Output;
}

impl<T, Any: TList> WithDimensions<Any> for Scalar<T> {
    type Output = Self;
}

impl<T, Ts, N, Ns, D, Ds> WithDimensions<TCons<D, Ds>> for Prism<T, Ts, N, Ns>
where
    Ts: WithDimensions<Ds>,
    Ns: TList,
    Ds: TList,
{
    type Output = Prism<T, <Ts as WithDimensions<Ds>>::Output, D, Ds>;
}

/// Helper trait to extract a rank-N slice from a rank-(N+1) tensor.
pub trait Slice {
    type Output;
}

impl<T, Ts, N, Ns> Slice for Prism<T, Ts, N, Ns>
    where
    Ns: TList,
    TCons<N, Ns>: TInits,
    Ts: WithDimensions<Inits<TCons<N, Ns>>>,
    <Ts as WithDimensions<Inits<TCons<N, Ns>>>>::Output: Container,
{
    type Output = <<Ts as WithDimensions<Inits<TCons<N, Ns>>>>::Output as Container>::Containing<T>;
}

#[cfg(test)]
pub mod slice_tests {
    use super::*;
    static_assertions::assert_type_eq_all!(Scalar<usize>, <Vect<usize, 3> as Slice>::Output);
    static_assertions::assert_type_eq_all!(Vect<usize, 3>, <Mat<usize, 2, 3> as Slice>::Output);
    static_assertions::assert_type_eq_all!(Mat<usize, 4, 5>, <Tensor3<usize, 3, 4, 5> as Slice>::Output);
}

impl<T, Ts, N, Ns> Index<Fin<N>> for Prism<T, Ts, N, Ns>
    where
    Self: Slice,
    N: Unsigned,
    Ns: TList,
    TCons<N, Ns>: TLast,
    Last<TCons<N, Ns>>: ArrayLength + NonZero,
{
    type Output = <Self as Slice>::Output;
    fn index(&self, index: Fin<N>) -> &Self::Output {
        let arr: &Array<Self::Output, Last<TCons<N, Ns>>> = unsafe { core::mem::transmute(&self.0) };
        &arr[index.val()]
    }
}
