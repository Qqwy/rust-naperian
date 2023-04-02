use core::ops::Mul;
use core::{marker::PhantomData, fmt::Debug, ops::Add, hash::Hash};
use generic_array::{ArrayLength, sequence::Lengthen};
use generic_array::sequence::GenericSequence;
use typenum::{B1, Add1, Prod};

// use super::{Liftable, Lowerable};
// use super::scalar::Scalar;
use crate::{functional::tlist::{TList, TCons, TNil, First, Rest, TRest, TFirst}, common::Array};
use crate::functional::New;

use super::tensor::{Tensor, Scalar};

/// type-level 'function' turning a [`trait@TList`] of `N` dimensions into the correctly-shaped N-dimensional array:
///
/// ```rust
/// use naperian::hyper2::shape_of::ShapeOf;
///
/// use naperian::common::Array;
/// use naperian::functional::tlist::*;
/// use typenum::consts::{U1, U2, U3, U4, U5};
/// use static_assertions::assert_type_eq_all as type_eq;
///
/// // 0 dimensions: single value
/// type_eq!(ShapeOf<i32, TList![]>, i32);
///
/// // 1 dimensions: one-dimensional array
/// type_eq!(ShapeOf<u64, TList![U3]>, Array<u64, U3>);
///
/// // 2 dimensions: two-dimensional array
/// type_eq!(ShapeOf<u64, TList![U2, U4]>, Array<Array<u64, U2>, U4>);
///
/// // 3 dimensions:
/// type_eq!(ShapeOf<i8, TList![U2, U4, U5]>, Array<Array<Array<i8, U2>, U4>, U5>);
/// // etc.
/// ```
pub type ShapeOf<T, Dims> = <Dims as TShapeOf>::Output<T>;
pub trait TShapeOf: TList {
    type Output<T>;

    type Rank: ArrayLength + Add<B1>;
    type HSize: ArrayLength;
    fn dimensions() -> Array<usize, Self::Rank>;

    fn hreplicate<T: Clone>(elem: T) -> Self::Output<T>;
}

impl TShapeOf for TNil {
    type Rank = typenum::U0;
    type HSize = typenum::U1;
    type Output<T> = T;
    fn dimensions() -> Array<usize, Self::Rank> {
        Default::default()
    }

    fn hreplicate<T: Clone>(elem: T) -> Self::Output<T> {
        elem
    }
}

impl<D: ArrayLength, Ds: TShapeOf> TShapeOf for TCons<D, Ds>
    where
    Add1<Ds::Rank>: ArrayLength + Add<B1>,
    Array<usize, Ds::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ds::Rank>>>,
    D: Mul<Ds::HSize>,
    Prod<D, Ds::HSize>: ArrayLength,
{
    type Output<T> = ShapeOf<Array<T, D>, Ds>;
    type Rank = Add1<Ds::Rank>;
    type HSize = Prod<D, Ds::HSize>;
    fn dimensions() -> Array<usize, Self::Rank> {
        Ds::dimensions().append(D::USIZE)
    }

    fn hreplicate<T: Clone>(elem: T) -> Self::Output<T> {
        Ds::hreplicate(New::new(elem))
    }
}

// pub trait NonEmptyDims
// where
//     Self: TShapeOf + TFirst + TRest,
// {}

// impl<D: ArrayLength, Ds: TShapeOf> NonEmptyDims for TCons<D, Ds>
//     where
//     Add1<Ds::Rank>: ArrayLength + Add<B1>,
//     Array<usize, Ds::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ds::Rank>>>,
// {}

pub trait Alignable<GtEq>: Sized
    where
    Self: TShapeOf,
    GtEq: TShapeOf,
{
    fn align<T: Clone>(tensor: Tensor<T, Self>) -> Tensor<T, GtEq>;
}

impl Alignable<TNil> for TNil {
    fn align<T: Clone>(scalar: Scalar<T>) -> Scalar<T> {
        scalar
    }
}

impl<F, Fs, Gs> Alignable<TCons<F, Gs>> for TCons<F, Fs>
where
    F: ArrayLength + Mul<Fs::HSize> + Mul<Gs::HSize>,
    Fs: TShapeOf,
    Gs: TShapeOf,
    Fs: Alignable<Gs>,
    Prod<F, Fs::HSize>: ArrayLength,
    Prod<F, Gs::HSize>: ArrayLength,
    Add1<Fs::Rank>: ArrayLength + Add<B1>,
    Array<usize, Fs::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Fs::Rank>>>,
    Add1<Gs::Rank>: ArrayLength + Add<B1>,
    Array<usize, Gs::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Gs::Rank>>>,
{
    fn align<T: Clone>(tensor: Tensor<T, Self>) -> Tensor<T, TCons<F, Gs>> {
        Alignable::<Gs>::align(tensor.lower()).lift()
    }
}

impl<F, Fs> Alignable<TCons<F, Fs>> for TNil
where
    F: ArrayLength + Mul<Fs::HSize>,
    Fs: TShapeOf,
    Prod<F, Fs::HSize>: ArrayLength,
    Add1<Fs::Rank>: ArrayLength + Add<B1>,
    Array<usize, Fs::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Fs::Rank>>>,
{
    fn align<T: Clone>(scalar: Scalar<T>) -> Tensor<T, TCons<F, Fs>> {
        Tensor(TCons::<F, Fs>::hreplicate(scalar.0), PhantomData)
    }
}

pub trait MatchingDimensions<Other> {
    type Output: TList;
}

impl MatchingDimensions<TNil> for TNil {
    type Output = TNil;
}

impl<D, Ds: TList> MatchingDimensions<TNil> for TCons<D, Ds> {
    type Output = TCons<D, Ds>;
}


impl<D, Ds: TList> MatchingDimensions<TCons<D, Ds>> for TNil {
    type Output = TCons<D, Ds>;
}

impl<D, Ds: TList, Ds2: TList> MatchingDimensions<TCons<D, Ds2>> for TCons<D, Ds>
where
    Ds: MatchingDimensions<Ds2>,
{
    type Output = TCons<D, <Ds as MatchingDimensions<Ds2>>::Output>;
}

pub fn align2<T: Clone, Left, Right, LeftAligned, RightAligned>(left: Tensor<T, Left>, right: Tensor<T, Right>) -> (Tensor<T, LeftAligned>, Tensor<T, RightAligned>)
    where
    Left: MatchingDimensions<Right, Output = LeftAligned> + Alignable<LeftAligned>,
    Right: MatchingDimensions<Left, Output = RightAligned> + Alignable<RightAligned>,
    LeftAligned: TShapeOf,
    RightAligned: TShapeOf,
{
    (Alignable::align(left), Alignable::align(right))
}

#[cfg(test)]
mod tests {
    use std::println;

    use crate::hyper2::tensor::{Mat, Vect};
    use generic_array::arr;
    use typenum::{U2, U3};

    use super::*;
    #[test]
    fn align() {
        // let left: Vect<usize, U3> = Scalar::new(arr![1,2,3]).lift();
        let left: Mat<usize, _, _> = Scalar::new(arr![arr![1,2,3], arr![4,5,6]]).lift().lift();
        let right: Vect<usize, _> = Scalar::new(arr![4,5,6, 7]).lift();
        // let right  = Scalar::new(4);
        let (la, ra) = align2(left, right);
        println!("{:?}", &la);
        println!("{:?}", &ra);
    }
}
