pub mod shape_of;
// pub mod scalar;
pub mod tensor;

use shape_of::{ShapeOf};
// use scalar::Scalar;
use tensor::{Tensor, Scalar, Vect, Mat};

use crate::{common::Array, functional::tlist::{TList, TCons, Rest, Reverse, TReverse, TConcat}};

use core::{marker::PhantomData, ops::Add};
use generic_array::{ArrayLength, sequence::Lengthen};
use typenum::{NonZero, B1, Add1};


// pub trait Liftable {
//     type Lifted;

//     /// Turns a lower-dimension Hyper
//     /// whose element type is Array<T, N>
//     /// into a one-dimension-higher Hyper
//     /// whose element type is T,
//     /// with the new dimension being N.
//     ///
//     /// Inverse of [`Prism::lower`].
//     fn lift(self) -> Self::Lifted;
// }

// pub trait Lowerable {
//     type Lowered;
//     fn lower(self) -> Self::Lowered;
// }


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
        let scalar: Scalar<Array<Array<usize, U3>, U2>> = Scalar::new(arrs);
        println!("{:?}", &scalar);
        let vect: Vect<Array<usize, U3>, U2> = scalar.lift();
        println!("{:?}", &vect);
        let mattie: Mat<usize, U2, U3> = vect.lift();
        println!("{:?}", &mattie);
        println!("{:?}", &crate::helper::type_name_of_val(&mattie));
        let vect2 = mattie.lower();
        println!("{:?}", &vect2);
        let scalar2 = vect2.lower();
        println!("{:?}", &scalar2);
    }
}
