use core::ops::Mul;
use core::{marker::PhantomData, fmt::Debug, ops::Add, hash::Hash};
use generic_array::{ArrayLength, sequence::Lengthen};
use generic_array::sequence::GenericSequence;
use typenum::{B1, Add1, Prod};

// use super::{Liftable, Lowerable};
// use super::scalar::Scalar;
use crate::{functional::tlist::{TList, TCons, TNil, First, Rest, TRest, TFirst}, common::Array};
use crate::functional::New;

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
