//! Contains trait implementations for the functional traits
//! for GenericArray
//!

use crate::common::Array;
use crate::fin::Fin;
use generic_array::ArrayLength;
use generic_array::sequence::GenericSequence;

use super::{Container, Mappable, Mappable2, Mappable3, Apply, New, NewFrom, Naperian};

unsafe impl<T, N> Container for Array<T, N>
where
    N: ArrayLength,
{
    type Elem = T;
    type Containing<X> = Array<X, N>;
}

impl<T, U, N> Mappable<U> for Array<T, N>
where
    N: ArrayLength,
{
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> U) -> Self::Containing<U> {
        Array::generate(|pos| {
            let val = &self[pos];
            fun(val)
        })
    }
    fn map_by_value(self, fun: impl FnMut(Self::Elem) -> U) -> Self::Containing<U> {
        self.into_iter().map(fun).collect()
    }
}

impl<T: Clone, N: ArrayLength> New<T> for Array<T, N> {
    fn new(elem_val: T) -> Self {
        Array::generate(|_pos| elem_val.clone())
    }
}

impl<T, N: ArrayLength> NewFrom<T> for Array<T, N> {
    fn new_from(mut fun: impl FnMut() -> T) -> Self {
        Array::generate(|_pos| fun())
    }
}

impl<A, B, F: Fn(&A) -> B, N: ArrayLength> Apply<A, B, F> for Array<F, N> {
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B> {
        Array::generate(|pos| {
            let fun = &self[pos];
            let val = &vals[pos];
            fun(val)
        })
    }
}

impl<A, U, N: ArrayLength> Mappable2<A, U> for Array<A, N> {
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        mut fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U> {
        Array::generate(|pos| {
            let left = &self[pos];
            let right = &rhs[pos];
            fun(left, right)
        })
    }

    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        mut fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U> {
        self.into_iter()
            .zip(rhs)
            .map(|(left, right)| fun(left, right))
            .collect()
    }
}

impl<A, U, N: ArrayLength> Mappable3<A, U> for Array<A, N> {
    fn map3<B, C>(
        &self,
        second: &Self::Containing<B>,
        third: &Self::Containing<C>,
        mut fun: impl FnMut(&A, &B, &C) -> U,
    ) -> Self::Containing<U> {
        Array::generate(|pos| {
            let one = &self[pos];
            let two = &second[pos];
            let three = &third[pos];
            fun(one, two, three)
        })
    }

    fn map3_by_value<B, C>(
        self,
        second: Self::Containing<B>,
        third: Self::Containing<C>,
        mut fun: impl FnMut(A, B, C) -> U,
    ) -> Self::Containing<U> {
        self.into_iter()
            .zip(second)
            .zip(third)
            .map(|((one, two), three)| fun(one, two, three))
            .collect()
    }
}

impl<T, N: ArrayLength> Naperian<T> for Array<T, N>
where
    Self: Container<Containing<Fin<N>> = Array<Fin<N>, N>>,
{
    type Log = Fin<N>;
    fn lookup(&self, index: Self::Log) -> &T {
        &self[index.val()]
    }
    fn positions() -> Array<Self::Log, N> {
        Array::generate(|pos| {
            // SAFETY: pos is in range [0..N)
            unsafe { Fin::new_unchecked(pos) }
        })
    }
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self {
        Array::generate(|pos| {
            // SAFETY: pos is in range [0..N)
            let fin = unsafe { Fin::new_unchecked(pos) };
            fun(fin)
        })
    }
}
