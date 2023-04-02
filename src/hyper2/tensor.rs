use core::{marker::PhantomData, fmt::Debug, ops::{Add, Mul}, hash::Hash};
use generic_array::{ArrayLength, sequence::Lengthen};
use typenum::{B1, Add1, Prod};

// use super::{Liftable, Lowerable};
use super::shape_of::{ShapeOf, TShapeOf};
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

impl<T> Scalar<T> {
    pub fn new(val: T) -> Self {
        Self(val, PhantomData)
    }
}

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

