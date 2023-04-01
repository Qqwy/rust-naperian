pub mod scalar;
pub mod prism;

use core::{marker::PhantomData, ops::Add};

use generic_array::{ArrayLength, sequence::Lengthen};
use scalar::Scalar;
use prism::{Prism, NonEmptyDims, ShapeOf};
use typenum::{NonZero, B1, Add1};

use crate::{common::Array, functional::tlist::{TList, TCons, Rest, Reverse, TReverse, TConcat}};

use self::prism::TShapeOf;


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
    type Lifted = Prism<T, TList![N]>;
    fn lift(self) -> Self::Lifted {
        Prism(self.0, PhantomData)
    }
}

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

impl<T, N: ArrayLength> Prism<T, TList![N]>
{
    pub fn lower(self) -> Scalar<Array<T, N>> {
        Scalar(self.0)
    }
}

impl<T, D: ArrayLength, Ds: NonEmptyDims> Prism<T, TCons<D, Ds>>
    where
    TCons<D, Ds>: NonEmptyDims,
    <TCons<D, Ds> as TShapeOf>::Rank: ArrayLength + Add<B1>,
    TCons<D, Ds>: TShapeOf<Output<T> = ShapeOf<Array<T, D>, Ds>>,
{
    pub fn lower(self) -> Prism<Array<T, D>, Ds> {
        Prism(self.0, PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use typenum::{U3, U2};
    extern crate std;
    use std::println;

    use super::*;
    #[test]
    fn lift_and_lower() {
        use generic_array::arr;
        let arrs = arr![arr![1,2,3], arr![4,5,6]];
        let scalar = Scalar(arrs);
        println!("{:?}", &scalar);
        let vect = scalar.lift();
        println!("{:?}", &vect);
        type Foo = ShapeOf<usize, TList![U3, U2]>;
        println!("{:?}", core::any::type_name::<Foo>());
        let mat = vect.lift();
        println!("{:?}", &mat);
    }
}

