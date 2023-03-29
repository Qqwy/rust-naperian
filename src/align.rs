//! Helper code to align Tensors of different (but compatible) shapes.

use super::{Hyper, Prism, Scalar};
use crate::common::Array;
use core::marker::PhantomData;
use frunk::hlist::{HCons, HList};
use generic_array::sequence::{Lengthen};
use generic_array::ArrayLength;
use typenum::NonZero;
use typenum::{Add1, Prod, Sub1, B1};

pub trait Align<Other>
where
    Self: Hyper,
    Other: Hyper<Elem = Self::Elem>,
{
    fn align(self) -> Other;
}

impl<T, Other, ORank> Align<Other> for Scalar<T>
where
    Other: Hyper<Elem = Self::Elem, Rank = ORank>,
    ORank: typenum::IsGreaterOrEqual<Self::Rank, Output = B1>,
{
    fn align(self) -> Other {
        Other::hreplicate(self.0)
    }
}

impl<T, Ts, N, Ns, Other, DimensionsRest> Align<Other> for Prism<T, Ts, N, Ns>
where
    DimensionsRest: HList,
    Other: Hyper<Elem = Self::Elem, Dimensions = HCons<N, DimensionsRest>>,
    Other::Inner: Hyper<Elem = Array<T, N>>,

    Ts: Hyper<Dimensions = Ns, Elem = Array<T, N>> + Align<Other::Inner>,
    Ns: HList,
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
        // TODO: Probably incorrect
        let res: Prism<T, _, N, Other::Dimensions> = Prism(Ts::align(self.0), PhantomData);
        unsafe { core::mem::transmute_copy(&res) }
    }
}

pub trait HyperMax<Other> {
    type Output;
}
impl<T, U> HyperMax<Scalar<U>> for Scalar<T> {
    type Output = Scalar<T>;
}
impl<U, T, Ts, N, Ns> HyperMax<Scalar<U>> for Prism<T, Ts, N, Ns>
where
    N: ArrayLength + NonZero,
    Ns: HList,
{
    type Output = Prism<T, Ts, N, Ns>;
}

impl<U, T, Ts, N, Ns> HyperMax<Prism<T, Ts, N, Ns>> for Scalar<U>
where
    N: ArrayLength + NonZero,
    Ns: HList,
{
    type Output = Prism<U, Ts, N, Ns>;
}

impl<U, T, Ts, Ts2, N, Ns, Ns2> HyperMax<Prism<U, Ts, N, Ns>> for Prism<T, Ts2, N, Ns2>
where
    N: ArrayLength + NonZero,
    Ns: HList,
    Ns2: HList,
    Ts: HyperMax<Ts2>,
    <Ts as HyperMax<Ts2>>::Output: Hyper,
{
    type Output = Prism<
        U,
        <Ts as HyperMax<Ts2>>::Output,
        N,
        <<Ts as HyperMax<Ts2>>::Output as Hyper>::Dimensions,
    >;
}

pub fn align2<Left, Right, LMax, RMax>(left: Left, right: Right) -> (LMax, RMax)
where
    Left: Hyper<Elem = LMax::Elem> + Maxed<Right, LMax>,
    Right: Hyper<Elem = RMax::Elem> + Maxed<Left, RMax>,
    LMax: Hyper,
    RMax: Hyper,
{
    (left.align(), right.align())
}

/// Helper subtrait to make trait bounds more readable.
/// As: Maxed<Bs, AsAligned> means that 'AsAligned' is a Hyper just like As but having been aligned with the dimensions of Bs.
pub trait Maxed<Other, SelfAligned>:
    HyperMax<Other, Output = SelfAligned> + Align<SelfAligned>
where
    SelfAligned: Hyper<Elem = <Self as Hyper>::Elem>,
{
}
impl<T, Other, SelfAligned> Maxed<Other, SelfAligned> for T
where
    Self: HyperMax<Other, Output = SelfAligned> + Align<SelfAligned>,
    SelfAligned: Hyper<Elem = <Self as Hyper>::Elem>,
{
}

// TODO temporary code so we can use cargo asm
pub fn hypermax() {
    use crate::const_aliases::{Mat, Tensor3, Vect};
    use generic_array::arr;
    let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
    let tens = Tensor3::<usize, 2, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    type Max = <Vect<usize, 1> as HyperMax<Mat<usize, 2, 1>>>::Output;
    let res = std::any::type_name::<Max>();
    println!("Max: {res:?}");
    let (mat_aligned, tens_aligned) = align2(mat, tens);
    // let mat_aligned: Max = mat.align();
    // let tens_aligned: Max = tens.align();
    println!("mat_aligned: {:?}", &mat_aligned);
    println!("tens_aligned: {:?}", &tens_aligned);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypermax() {
        super::hypermax();
    }
}
