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

pub trait Lowerable {
    type Lowered;
    fn lower(self) -> Self::Lowered;
}


#[cfg(test)]
mod tests {
    use typenum::{U3, U2};
    extern crate std;
    use std::{println, dbg};

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
        println!("{:?}", &crate::helper::type_name_of_val(&mat));
        let vect2 = mat.lower();
        println!("{:?}", &vect2);
        let scalar2 = vect2.lower();
        println!("{:?}", &scalar2);
    }
}

