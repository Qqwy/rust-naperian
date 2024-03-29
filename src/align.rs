//! Helper code to align Tensors of different (but compatible) shapes.

use super::{Hyper, Prism, Scalar};
use crate::common::Array;
use crate::functional::tlist::{Compatible, TCons, TList};
use core::marker::PhantomData;
use generic_array::sequence::Lengthen;
use generic_array::ArrayLength;
use typenum::NonZero;
use typenum::{Add1, Prod, Sub1, B1};

/// Helper trait to align a tensor with another.
///
/// Rust implementation of the 'Alignable' typeclass from the Naperian paper.
///
/// This means that if Self is a Tensor whose rank is lower,
/// that its elements will be repeaded for each of the higher ranks of the other.
///
/// The definition only makes sense (and the impls guard against this)
/// if Other is equal or larger in rank than Self.
pub trait Alignable<Other>
where
    Self: Hyper,
    Other: Hyper<Elem = Self::Elem>,
{
    /// Aligns self to the shape of Other by repeating the inner elements.
    ///
    /// It is unlikely that you need to call this method directly.
    /// You probably want to use the free function [`align2`] instead.
    fn align(self) -> Other;
}

impl<T, Other, ORank> Alignable<Other> for Scalar<T>
where
    Other: Hyper<Elem = Self::Elem, Rank = ORank>,
    ORank: typenum::IsGreaterOrEqual<Self::Rank, Output = B1>,
{
    fn align(self) -> Other {
        Other::hreplicate(self.0)
    }
}

impl<T, Ts, N, Ns, Other, DimensionsRest> Alignable<Other> for Prism<T, Ts, N, Ns>
where
    DimensionsRest: TList,
    Other: Hyper<Elem = Self::Elem, Dimensions = TCons<N, DimensionsRest>>,
    Other::Inner: Hyper<Elem = Array<T, N>>,

    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>> + Alignable<Other::Inner>,
    Ns: TList,
    N: ArrayLength + NonZero,
    Ts::AmountOfElems: core::ops::Mul<N>,
    Ts::Rank: core::ops::Add<B1>,
    Add1<Ts::Rank>: ArrayLength,
    Prod<Ts::AmountOfElems, N>: ArrayLength,

    Ts::Rank: core::ops::Add<B1> + ArrayLength,
    Add1<Ts::Rank>: ArrayLength + core::ops::Sub<B1, Output = Ts::Rank>,
    Sub1<Add1<Ts::Rank>>: ArrayLength,
    Array<usize, Ts::Rank>: Lengthen<usize, Longer = Array<usize, Add1<Ts::Rank>>>,
    T: Clone,
{
    fn align(self) -> Other {
        let res: Prism<T, _, N, Other::Dimensions> = Prism(Ts::align(self.0), PhantomData);
        unsafe { core::mem::transmute_copy(&res) }
    }
}

/// Helper trait to align two tensors whose inner ranks are the same.
///
/// This is a Rust implementation of the 'Max' type family from the Naperian paper.
///
/// - (Scalar<T>, Scalar<U>) -> Scalar<T>
/// - (Scalar<T>, Prism<U, Us>) -> Prism<T, Us>
/// - (Prism<T, Ts>, Scalar<U>) -> Prism<T, Ts>
/// - (Prism<T, Ts>, Prism<U, Us>) -> Prism<T, MatchingShape<Ts, Us>>
pub trait MatchingShape<Other> {
    type Output;
}
impl<T, U> MatchingShape<Scalar<U>> for Scalar<T> {
    type Output = Scalar<T>;
}
impl<U, T, Ts, N, Ns> MatchingShape<Scalar<U>> for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: TList,
{
    type Output = Prism<T, Ts, N, Ns>;
}

impl<U, T, Ts, N, Ns> MatchingShape<Prism<T, Ts, N, Ns>> for Scalar<U>
where
    N: ArrayLength + NonZero,
    Ns: TList,
{
    type Output = Prism<U, Ts, N, Ns>;
}

impl<U, T, Ts, Ts2, N, Ns, Ns2> MatchingShape<Prism<U, Ts, N, Ns>> for Prism<T, Ts2, N, Ns2>
where
    N: ArrayLength + NonZero,
    Ns: TList,
    Ns2: TList,
    Ns: Compatible<Ns2>,
    Ts: MatchingShape<Ts2>,
    <Ts as MatchingShape<Ts2>>::Output: Hyper,
{
    type Output = Prism<
        U,
        <Ts as MatchingShape<Ts2>>::Output,
        N,
        <<Ts as MatchingShape<Ts2>>::Output as Hyper>::Dimensions,
    >;
}

/// Helper subtrait to make trait bounds more readable.
///
/// As: ShapeMatched<Bs, AsAligned> means that 'AsAligned' is a Hyper just like As but having been aligned with the dimensions of Bs.
pub trait ShapeMatched<Other, SelfAligned>:
    MatchingShape<Other, Output = SelfAligned> + Alignable<SelfAligned>
where
    SelfAligned: Hyper<Elem = <Self as Hyper>::Elem>,
{
}
impl<T, Other, SelfAligned> ShapeMatched<Other, SelfAligned> for T
where
    Self: MatchingShape<Other, Output = SelfAligned> + Alignable<SelfAligned>,
    SelfAligned: Hyper<Elem = <Self as Hyper>::Elem>,
{
}

/// Aligns two Tensors whose inner dimension match.
///
/// - If Left and Right are of equal rank, they are returned unchanged.
/// - If Left is of lower rank than Right, Left's contents are repeated for the higher ranks.
/// - Otherwise, Right is of lower rank than Left. Right's contents are repeated for the higher ranks.
pub fn align2<Left, Right, LeftAligned, RightAligned>(
    left: Left,
    right: Right,
) -> (LeftAligned, RightAligned)
where
    Left: Hyper<Elem = LeftAligned::Elem> + ShapeMatched<Right, LeftAligned>,
    Right: Hyper<Elem = RightAligned::Elem> + ShapeMatched<Left, RightAligned>,
    LeftAligned: Hyper,
    RightAligned: Hyper,
{
    (left.align(), right.align())
}

// TODO temporary code so we can use cargo asm
pub fn hypermax() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::println;

    #[test]
    fn hypermax() {
        use crate::const_aliases::{Mat, Tensor3, Vect};
        use generic_array::arr;
        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let tens =
            Tensor3::<usize, 2, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        type Aligned = <Vect<usize, 1> as MatchingShape<Mat<usize, 2, 1>>>::Output;
        let res = core::any::type_name::<Aligned>();
        println!("MatchingShape: {res:?}");
        let (mat_aligned, tens_aligned) = align2(mat, tens);
        // let mat_aligned: MatchingShape = mat.align();
        // let tens_aligned: MatchingShape = tens.align();
        println!("mat_aligned: {:?}", &mat_aligned);
        println!("tens_aligned: {:?}", &tens_aligned);
    }
}
