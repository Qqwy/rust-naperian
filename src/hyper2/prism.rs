use core::{marker::PhantomData, fmt::Debug, ops::Add, hash::Hash};
use generic_array::{ArrayLength, sequence::Lengthen};
use typenum::{B1, Add1};

use super::{Liftable, Lowerable};
use super::scalar::Scalar;
use crate::{functional::tlist::{TList, TCons, TNil, First, Rest, TRest, TFirst}, common::Array};


#[repr(transparent)]
pub struct Prism<T, Dims: NonEmptyDims>(pub(crate) ShapeOf<T, Dims>, pub(crate) PhantomData<Dims>);

impl<T: Debug, Dims: NonEmptyDims> Debug for Prism<T, Dims>
    where
    ShapeOf<T, Dims>: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Prism")
            .field("type", &crate::helper::debug_typename::<T>())
            .field("dims", &Dims::dimensions())
            .field("contents", &self.0)
            .finish()
    }
}

impl<T: Hash, Dims: NonEmptyDims> Hash for Prism<T, Dims>
where
    ShapeOf<T, Dims>: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T: Clone, Dims: NonEmptyDims> Clone for Prism<T, Dims>
where
    ShapeOf<T, Dims>: Clone
{
    fn clone(&self) -> Self {
        Prism(self.0.clone(), PhantomData)
    }
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

impl<Rhs, T: PartialEq<Rhs>, Dims: NonEmptyDims> PartialEq<Prism<Rhs, Dims>> for Prism<T, Dims>
    where
    ShapeOf<T, Dims>: PartialEq<ShapeOf<Rhs, Dims>>,
{
    fn eq(&self, other: &Prism<Rhs, Dims>) -> bool {
        self.0.eq(&other.0)
    }
    fn ne(&self, other: &Prism<Rhs, Dims>) -> bool {
        self.0.ne(&other.0)
    }
}

impl<T: Eq, Dims: NonEmptyDims> Eq for Prism<T, Dims>
where
    ShapeOf<T, Dims>: Eq,
{
}


impl<Rhs, T: PartialOrd<Rhs>, Dims: NonEmptyDims> PartialOrd<Prism<Rhs, Dims>> for Prism<T, Dims>
where
    ShapeOf<T, Dims>: PartialOrd<ShapeOf<Rhs, Dims>>,
{
    fn partial_cmp(&self, other: &Prism<Rhs, Dims>) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl<T: Ord, Dims: NonEmptyDims> Ord for Prism<T, Dims>
where
    ShapeOf<T, Dims>: Ord,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

pub type ShapeOf<T, Dims> = <Dims as TShapeOf>::Output<T>;
pub trait TShapeOf: TList {
    type Output<T>;
    type Rank: ArrayLength + Add<B1>;
    fn dimensions() -> Array<usize, Self::Rank>;
}

impl TShapeOf for TNil {
    type Rank = typenum::U0;
    type Output<T> = T;
    fn dimensions() -> Array<usize, Self::Rank> {
        Default::default()
    }
}

impl<D: ArrayLength, Ds: TShapeOf> TShapeOf for TCons<D, Ds>
    where
    Add1<Ds::Rank>: ArrayLength + Add<B1>,
    Array<usize, Ds::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ds::Rank>>>,
{
    type Output<T> = ShapeOf<Array<T, D>, Ds>;
    type Rank = Add1<Ds::Rank>;
    fn dimensions() -> Array<usize, Self::Rank> {
        Ds::dimensions().append(D::USIZE)
    }
}

pub trait NonEmptyDims
where
    Self: TShapeOf + TFirst + TRest,
{}

impl<D: ArrayLength, Ds: TShapeOf> NonEmptyDims for TCons<D, Ds>
    where
    Add1<Ds::Rank>: ArrayLength + Add<B1>,
    Array<usize, Ds::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ds::Rank>>>,
{}

impl<T, N, Ns> Liftable for Prism<Array<T, N>, Ns>
where
    N: ArrayLength,
Add1<Ns::Rank>: ArrayLength + Add<B1>,
Array<usize, Ns::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ns::Rank>>>,
// Ns: TShapeOf<Output<Array<T, N>> = Array<ShapeOf<T, Ns>, N>>,
    Ns: NonEmptyDims,
TCons<N, Ns>: TShapeOf<Output<T> = ShapeOf<Array<T, N>, Ns>>,

{
    type Lifted = Prism<T, TCons<N, Ns>>;
    fn lift(self) -> Self::Lifted {
        Prism(self.0, PhantomData)
    }
}


impl<T, N: ArrayLength> Lowerable for Prism<T, TCons<N, TNil>>
{
    type Lowered = Scalar<Array<T, N>>;
    fn lower(self) -> Scalar<Array<T, N>> {
        Scalar(self.0)
    }
}


impl<T, N: ArrayLength, Ns: NonEmptyDims> Lowerable for Prism<T, TCons<N, Ns>>
    where
    Add1<Ns::Rank>: ArrayLength + Add<B1>,
    Array<usize, Ns::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ns::Rank>>>,
{
    type Lowered = Prism<Array<T, N>, Ns>;
    fn lower(self) -> Prism<Array<T, N>, Ns> {
        Prism(self.0, PhantomData)
    }
}
