use core::{marker::PhantomData, fmt::Debug, ops::{Add, Mul}, hash::Hash};
use generic_array::{ArrayLength, sequence::{Lengthen, GenericSequence}, functional::FunctionalSequence};
use typenum::{B1, Add1, Prod, U1};

use crate::{functional::{Container, New, NewFrom, Mappable, Mappable2}, align::ShapeMatched};
// use super::{Liftable, Lowerable};
use super::shape_of::{ShapeOf, TShapeOf, MatchingDimensions, Alignable, align2};
// use super::scalar::Scalar;
use crate::{functional::tlist::{TList, TCons, TNil, First, Rest, TRest, TFirst}, common::Array};


#[repr(transparent)]
pub struct Tensor<T, Dims: TShapeOf>(pub(crate) ShapeOf<T, Dims>, pub(crate) PhantomData<Dims>);

impl<T: Debug, Dims> Debug for Tensor<T, Dims>
    where
    ShapeOf<T, Dims>: Debug,
    Dims: TShapeOf,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tensor")
            .field("type", &crate::helper::debug_typename::<T>())
            .field("dims", &Dims::dimensions())
            .field("contents", &self.0)
            .finish()
    }
}

pub type Scalar<T> = Tensor<T, TNil>;
pub type Vect<T, N> = Tensor<T, TList![N]>;
pub type Mat<T, R, C> = Tensor<T, TList![C, R]>;
pub type Tensor3<T, S, R, C> = Tensor<T, TList![C, R, S]>;

impl<T: Hash, Dims> Hash for Tensor<T, Dims>
where
    ShapeOf<T, Dims>: Hash,
    Dims: TShapeOf,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T: Clone, Dims> Clone for Tensor<T, Dims>
where
    ShapeOf<T, Dims>: Clone,
    Dims: TShapeOf,
{
    fn clone(&self) -> Self {
        Tensor(self.0.clone(), PhantomData)
    }
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

impl<Rhs, T: PartialEq<Rhs>, Dims> PartialEq<Tensor<Rhs, Dims>> for Tensor<T, Dims>
    where
    ShapeOf<T, Dims>: PartialEq<ShapeOf<Rhs, Dims>>,
    Dims: TShapeOf,
{
    fn eq(&self, other: &Tensor<Rhs, Dims>) -> bool {
        self.0.eq(&other.0)
    }
    fn ne(&self, other: &Tensor<Rhs, Dims>) -> bool {
        self.0.ne(&other.0)
    }
}

impl<T: Eq, Dims> Eq for Tensor<T, Dims>
where
    ShapeOf<T, Dims>: Eq,
    Dims: TShapeOf,
{ }


impl<Rhs, T: PartialOrd<Rhs>, Dims> PartialOrd<Tensor<Rhs, Dims>> for Tensor<T, Dims>
where
    ShapeOf<T, Dims>: PartialOrd<ShapeOf<Rhs, Dims>>,
    Dims: TShapeOf,
{
    fn partial_cmp(&self, other: &Tensor<Rhs, Dims>) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl<T: Ord, Dims> Ord for Tensor<T, Dims>
where
    ShapeOf<T, Dims>: Ord,
    Dims: TShapeOf,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}


impl<T, N, Ns> Tensor<Array<T, N>, Ns>
where
    N: ArrayLength,
    Ns: TShapeOf,
    TCons<N, Ns>: TShapeOf<Output<T> = ShapeOf<Array<T, N>, Ns>>,
    // Add1<Ns::Rank>: ArrayLength + Add<B1>,
    // Array<usize, Ns::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ns::Rank>>>,
{
    /// Lifts a (K-1)-dimensional `Tensor<Array<T, N>, TList![Ns]>` into a K-dimensional `Tensor<T, TList![N, ...Ns]>`
    pub fn lift(self) -> Tensor<T, TCons<N, Ns>> {
        Tensor(self.0, PhantomData)
    }
}

// /// Turns a `Vect<T, N>` into a `Scalar<Array<T, N>>`
// ///
// /// In other words: lowers a 1-dimensional `Tensor<T, TList![N]>` into a `Scalar<Array<T, N>>`
// impl<T, N> Lowerable for Tensor<T, TList![N]>
// where
//     N: ArrayLength,
// {
//     type Lowered = Scalar<Array<T, N>>;
//     fn lower(self) -> Scalar<Array<T, N>> {
//         Scalar(self.0)
//     }
// }

impl<T, N, Ns> Tensor<T, TCons<N, Ns>>
    where
    N: ArrayLength + Mul<Ns::HSize>,
    Ns: TShapeOf,
    Add1<Ns::Rank>: ArrayLength + Add<B1>,
    Prod<N, Ns::HSize>: ArrayLength,
    Array<usize, Ns::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ns::Rank>>>,
{
    /// Lowers a K-dimensional `Tensor<T, TList![N, ...Ns]>` into a (K-1)-dimensional `Tensor<Array<T, N>, Ns>`
    pub fn lower(self) -> Tensor<Array<T, N>, Ns> {
        Tensor(self.0, PhantomData)
    }
}

unsafe impl<T, Dims> Container for Tensor<T, Dims>
    where
    Dims: TShapeOf,
{
    type Elem = T;
    type Containing<X> = Tensor<X, Dims>;
}

impl<T: Clone, Dims> New<T> for Tensor<T, Dims>
where
    Dims: TShapeOf,
{
    fn new(elem_val: T) -> Self {
        Tensor(Dims::hreplicate(elem_val), PhantomData)
    }
}

impl<T, A> Mappable<A> for Tensor<T, TNil>
{
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> A) -> Self::Containing<A> {
        Tensor(fun(&self.0), PhantomData)
    }
    fn map_by_value(self, mut fun: impl FnMut(Self::Elem) -> A) -> Self::Containing<A> {
        Tensor(fun(self.0), PhantomData)
    }
}

impl<T, D, Ds, A> Mappable<A> for Tensor<T, TCons<D, Ds>>
    where
    TCons<D, Ds>: TShapeOf,
ShapeOf<T, TCons<D, Ds>>: Mappable<A> + Container<Elem = T, Containing<A> = ShapeOf<A, TCons<D, Ds>>>,
    Ds: TList,
{
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> A) -> Self::Containing<A> {
        Tensor(self.0.map(&mut fun), PhantomData)
    }
    fn map_by_value(self, mut fun: impl FnMut(Self::Elem) -> A) -> Self::Containing<A> {
        Tensor(self.0.map_by_value(&mut fun), PhantomData)
    }
}

// trait AlignedMappable2: Container {
//     type Output<U>;
//     fn map2<R, U>(self, rhs: Self::Containing<R>, fun: impl FnMut(Self::Elem, R) -> U) -> Self::Output<U>;
// }

// impl<T> AlignedMappable2 for Tensor<T, TNil>
// {
//     type Output<U> = Tensor<U, TNil>;
//     fn map2<R, U>(self, rhs: Tensor<R, TNil>, mut fun: impl FnMut(Self::Elem, R) -> U) -> Self::Output<U>
//     {
//         Tensor(fun(self.0, rhs.0), PhantomData)
//     }
// }

// impl<T, D, Ds, R> AlignedMappable2 for Tensor<T, TCons<D, Ds>>
// where
//     D: ArrayLength + Mul<Ds::HSize>,
//     Ds: TShapeOf,
//     ShapeOf<Array<T, D>, Ds>: FunctionalSequence<Array<T, D>>,
//     Add1<Ds::Rank>: ArrayLength + Add<B1>,
//     Prod<D, Ds::HSize>: ArrayLength,
//     Array<usize, Ds::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ds::Rank>>>,
// {
//     type Output<U> = Tensor<U, TCons<D, Ds>>;
//     fn map2<U>(self, rhs: Self::Containing<R>, fun: impl FnMut(Self::Elem, R) -> U) -> Self::Output<U> {
//         Tensor(self.0.zip(rhs.0, fun), PhantomData)
//     }
// }

impl<T, Dims> Tensor<T, Dims>
where
    Dims: TShapeOf,
{
    pub fn into_flat(self) -> Array<T, Dims::HSize> {
        // SAFETY: memory representation is guaranteed to match
        unsafe {core::mem::transmute_copy(&self)}
    }

    pub fn from_flat(array: Array<T, Dims::HSize>) -> Self {
        // SAFETY: memory representation is guaranteed to match
        unsafe {core::mem::transmute_copy(&array)}
    }
}

pub trait Mappable2b<R, Rhs = <Self as Container>::Containing<R>>
where
    Self: Container,
    Rhs: Container<Elem = R>,
{
    type Output<U>: Container<Elem = U>;
    fn map2<U>(self, rhs: Rhs, fun: impl FnMut(Self::Elem, R) -> U) -> Self::Output<U>;
}

impl<L: Clone, R: Clone, Dims, Dims2, DimsAligned> Mappable2b<R, Tensor<R, Dims2>> for Tensor<L, Dims>
where
    Dims: TShapeOf + MatchingDimensions<Dims2, Output = DimsAligned> + Alignable<DimsAligned>,
    Dims2: TShapeOf + MatchingDimensions<Dims, Output = DimsAligned> + Alignable<DimsAligned>,
    DimsAligned: TShapeOf,
{
    type Output<U> = Tensor<U, DimsAligned>;
    fn map2<U>(self, bs: Tensor<R, Dims2>, mut fun: impl FnMut(<Self as Container>::Elem, R) -> U) -> Self::Output<U> {
        use typenum::Unsigned;
        // Decide implementation based on the amount of elements in the aligned tensors.
        // TODO benchmark to:
        // - Find proper cutoff point
        // - Double-check how smart the compiler actually is for this kind of stuff.
        // - The kind of element type (Copy vs non-Copy) might also greatly matter!
        // if DimsAligned::HSize::USIZE < 1024 {
        if true {
            // If small, create both aligned tensors in full and operate on them as arrays directly.
            // For small arrays, this allows the compiler extra optimizations.
            // For large arrays, this would grow code size and the amount of memcpys
            let (self_aligned, bs_aligned) = align2(self, bs);
            let self_arr = self_aligned.into_flat();
            let bs_arr = bs_aligned.into_flat();
            let us_arr = self_arr.zip(bs_arr, fun);
            Tensor::from_flat(us_arr)
        } else {
            // If large, iterate over the two arrays lazily, repeating the elements in the smaller one.
            // Compiler can optimize this less (which is visible for small arrays)
            // But for large arrays it means we can skip building
            // the intermediate large array only to immediately consume it afterwards.
            let self_arr = self.into_flat();
            let bs_arr = bs.into_flat();
            let self_iter = self_arr.into_iter().cycle().take(DimsAligned::HSize::USIZE);
            let bs_iter = bs_arr.into_iter().cycle().take(DimsAligned::HSize::USIZE);
            let us_arr = self_iter.zip(bs_iter).map(|(l, r)| fun(l, r)).collect();
            Tensor::from_flat(us_arr)
        }
    }
}

#[cfg(test)]
mod test {
    use typenum::{U2, U3, U40};

    use crate::helper::type_name_of_val;

    use super::*;
    #[test]
    fn map2() {
        use generic_array::arr;
        let left: Scalar<_> = Scalar::from_flat(arr![10]);
        // let left: Vect<usize, _> = Vect::from_flat(arr![10, 20]);
        let right: Mat<usize, U40, U2> = Mat::from_flat(arr![1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10]);
        let added = left.map2(right, Add::add);
        std::println!("{:?}", &added);
        std::println!("{:?}", &type_name_of_val(&added));
    }
}

// impl<T, D, Ds, A, U> Mappable2<A, U> for Tensor<T, TCons<D, Ds>>
//     where
//     TCons<D, Ds>: TShapeOf,
//     ShapeOf<T, TCons<D, Ds>>: Mappable2<A, U> + Container<Elem = T> + Container<Containing<A> = ShapeOf<A, TCons<D, Ds>>> + Container<Containing<U> = ShapeOf<U, TCons<D, Ds>>>,
//     Ds: TList,
// {
//     fn map2_by_value<B>(
//         self,
//         rhs: Self::Containing<B>,
//         mut fun: impl FnMut(A, B) -> U,
//     ) -> Self::Containing<U> {
//         let new_arr = self.0.map2_by_value(rhs.0, &mut fun);
//         Tensor(new_arr, PhantomData)
//     }

//     fn map2<'b, B: 'b>(
//         &self,
//         rhs: &'b Self::Containing<B>,
//         fun: impl FnMut(&A, &'b B) -> U,
//     ) -> Self::Containing<U> {
//         todo!()
//     }
// }

use typenum::consts::*; //{U4, U8, U16384};
extern crate alloc;
type Type = alloc::string::String;
type R = U16; // U16384;
type C = U16;
pub fn mulexample(mat: Mat<Type, R, C>, vec: Vect<Type, C>) -> Mat<Type, R, C> {
    mat.map2(vec, |x, y| x + &y)
}
