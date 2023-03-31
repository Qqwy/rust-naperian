//! Naperian is a library to work with tensors, also known as N-dimensional arrays, in an ergonomic **and** type-safe manner.
//!
//! It **supports stable rust** (MSRV = 1.65.0) and works in `no_std` environments.
//!
//! # Type-Safe
//!
//! Naperian is fully type-safe.
//! Instead of only encoding the element type (which most collection types do),
//! the full list of dimensions of each tensor is encoded in the tensor's type as well.
//!
//! By keeping track of the the full list of dimensions:
//! - incorrect usage turns into compile-time errors.
//! - No bounds-checks or other runtime checks are necessary.
//!
//! # Ergonomic
//!
//! Naperian implements what is known as 'automatic alignment' or 'rank-polymorphic broadcasting',
//! just like you might know from languages like APL or Julia, or Python's NumPy library.
//!
//! It means that you can freely mix single values, lower-rank tensors and higher-rank tensors (as long as their lower dimensions match) in your calculations:
//!
//! ```rust
//! use naperian::aliases::{Mat, Vect};
//! let mat = Mat::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//! let vec = Vect::from([1, 2, 3]);
//! // vec automatically is repeated for each row of mat:
//! let res = mat + vec - 1;
//! assert_eq!(res, [[1,3,5], [4,6,8], [7,9,11]].into());
//! ```
//!
//! If the smaller-rank tensor's lower dimensions do not match the lower dimensions of the the higher-rank tensor, this results in a compile-time error:
//! ```compile_fail
//! use naperian::aliases::{Mat, Vect};
//! let mat = Mat::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//! let vec = Vect::from([1, 2, 3, 4, 5]);
//! let res = mat * vec; // <- Compile error!
//! ```
//!
//! ## Wow! How does this work?
//! Using a languages' ['normal'](https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system) type system to track dimensions and perform automatic alignment
//! is a technique introduced by the paper [APLicative Programming with Naperian Functors](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/aplicative.pdf) written by Jeremy Gibbons.
//! The paper implements the technique in Haskell. While Rust is not Haskell,
//! its type system is similar enough to make the same technique work...
//!
//!
//! ### Caveats
//! ... with a few caveats, that is.
//!
//! #### Rust's type system does not support higher-kindred types (HKTs).
//! These can be emulated by (ab?)using traits with generic associated types (GATs).
//!
//! While this works, it does lead to a large and ever-increasing amount of type constraints.
//! #### Rust does not support 'GADTs' nor 'type families'
//! GADTs (generic algebraic data types; <sub>not to be confused with GATs, they are very different!</sub>) are essentially an enum whose individual variants are constrained by different type variables, and after construction these type variables are hidden.
//!
//! In Rust, this needs to be emulated by building separate structs for each of the variants, all of which implement a shared trait.
//!
//! Similarly, type families, a Haskell construct to enable 'type-level functions', are emulated with a dedicated trait containing a GAT.
//!
//! Both emulated approaches in Rust introduce more verbose type constraint bounds.
//!
//! #### Support for type-level linked lists and type-level numbers
//! Type-level lists are supported using the [`frunk`] crate's HList type.
//! (though since we only use HList at the type level we might roll our own at some point).
//!
//! Type-level unsigned integers are supported using the [`typenum`] crate.
//! (_Until the [generic_const_exprs](https://github.com/rust-lang/rust/issues/76560)_ feature is ever finished and stabilized,
//! const generic numbers cannot be used in the crate's internals._)
//!
//! These are both examples of GADTs emulated using (in this case recursive) structs implementing a shared trait.
//! #### Prepare for large compiler error messages
//! Because of these implementation constraints, the types the Naperian tensors end up having are rather complex.
//! This makes some compiler errors messages not very fun to look at.
//!
//! Luckily, you won't have to touch any of this machinery when writing your code.
//! Referring to the [`Vect`], [`Mat`], [`Tensor3`], etc. aliases is usually good enough :-).
#![no_std]

#[cfg(test)]
extern crate std;
#[cfg(test)]
use std::println;

pub mod aliases;
pub mod align;
pub mod common;
pub mod const_aliases;
pub mod fin;
pub mod functional;
pub mod hyper;
pub mod paper;
pub mod compat;


use align::Alignable;
use common::Array;

use functional::{Container, Mappable2, Naperian, NaperianTranspose};

#[doc(inline)]
pub use functional::{Mappable, New};

#[doc(inline)]
pub use hyper::{Hyper, Liftable, HyperMappable2};
use hyper::{Prism, Scalar};
use paper::innerp_orig;

#[doc(inline)]
pub use const_aliases::*;

use core::iter::Sum;

use core::ops::Mul;

use generic_array::{arr, GenericArray};
use typenum::consts::*;

use typenum::U;

use functional::hlist::{HCons, HNil};

/// Calculate the inner product (also known as the dot product) of two collections of the same shape.
///
/// The inner product is the sum of multiplying all elements pairwise.
///
/// Its implementation is different from the one mentioned in the Naperian paper,
/// and has bounds which are on one hand simpler, but on the other do require some lifetime tracking.
///
/// (See the original definition as [`paper::innerp_orig`]).
pub fn innerp<'a, A, R: 'a>(a: &'a A, b: &'a A) -> R
where
    &'a A: IntoIterator<Item = &'a R>,
    A: Container<Containing<R> = A> + IntoIterator,
    &'a R: Mul<&'a R, Output = R>,
    R: Sum,
{
    a.into_iter().zip(b.into_iter()).map(|(x, y)| x * y).sum()
}

/// Calculates the matrix product of two matrices with the same element type `A`.
/// Given a f×g matrix and a g×h matrix, returns a f×h matrix.
///
/// This is implemented by first transforming both matrices to a common f×h×g representation,
/// and then mapping the inner product ([`innerp_orig`]) across the innermost g dimension to flatten it.
///
/// Compatibility of dimensions is fully determined at compile time.
// NOTE currently this uses innerp_orig as I could not get it working with the lifetime requirements of innerp.
// This could be improved in the future, for slightly less stack space usage.
pub fn matrixp<Fhga, Fha, Fga, Gha, Hga, Fa, Ga, Ha, A>(xss: &Fga, yss: &Gha) -> Fha
where
    Fhga: Container<Elem = Hga, Containing<Hga> = Fhga> + Container<Containing<Ha> = Fha>,
    Fhga: New<Fa::Containing<Gha::Containing<A>>> + Mappable2<Hga, Ha>,
    Fga: Container<Elem = Ga, Containing<Ga> = Fga> + Container<Containing<Hga> = Fhga>,
    Fga: Clone + Mappable<Hga>,
    Hga: Container<Containing<Ga> = Hga> + Container<Containing<A> = Ha>,
    Hga: New<Ga> + Mappable2<Ga, A>,
    Ga: Container<Elem = A, Containing<A> = Ga>,
    Ga: IntoIterator<Item = A> + Mappable2<A, A> + Naperian<A>,
    Gha: Container<Elem = Fa> + Container<Containing<A> = Ga>,
    Gha: Naperian<Fa, Log = Ga::Log>,
    Fa: Container<Elem = A, Containing<A> = Fa>,
    Fa: Naperian<A>,
    Fa::Containing<Gha::Containing<A>>: Naperian<Gha::Containing<A>, Log = Fa::Log>,
    A: Clone + core::iter::Sum + core::ops::Mul<Output = A>,
{
    let lifted_xss: Fhga = lifted_xss(xss);
    let lifted_yss: Fhga = lifted_yss(yss);
    lifted_xss.map2(&lifted_yss, |left: &Hga, right: &Hga| {
        left.map2(right, innerp_orig)
    })
}

fn lifted_xss<F, G, H>(xss: &F) -> F::Containing<H>
where
    F: Clone + Mappable<H> + Container<Elem = G>,
    G: Container,
    H: New<G> + Container,
{
    xss.clone().map_by_value(New::new)
}

fn lifted_yss<A, F, G, H>(yss: &F) -> H
where
    F: NaperianTranspose<G, A>,
    F::Containing<A>: Naperian<A, Log = F::Log>,
    G: Naperian<A>,
    G::Containing<F::Containing<A>>: Naperian<F::Containing<A>, Log = G::Log>,
    H: New<G::Containing<F::Containing<A>>>,
    A: Clone,
{
    New::new(yss.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn from_testing() {
        let mat: aliases::Mat<_, _, _> = [[1,2,3], [4,5,6]].into();
        let vec: aliases::Vect<_, _> = [1,2,3].into();
        let res = mat + vec;
        assert_eq!(res, [[2, 4, 6], [5, 7, 9]].into());
    }

    #[test]
    fn binary() {
        use hyper::HyperMappable2;
        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let vec = Vect::<usize, 3>::from_flat(arr![10, 20, 30]);
        let res = mat.map2(vec, |x, y| x + y);
        println!("{:?}", res);

        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let matsum = (&mat).map2(&mat, |x, y| x + y);
        println!("matsum: {:?}", matsum);
    }

    #[test]
    fn transpose() {
        use crate::functional::pair::Pair;
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let three_by_two: Array<GenericArray<_, _>, U2> = arr![v123, v456];
        println!("{three_by_two:?}");
        let two_by_three = three_by_two.transpose();
        println!("{two_by_three:?}");
        assert_eq!(two_by_three, arr![arr![1, 4], arr![2, 5], arr![3, 6]]);

        let pair_of_vecs = Pair(arr![1, 2, 3], arr![10, 20, 30]);
        let vec_of_pairs = pair_of_vecs.transpose();
        println!("{vec_of_pairs:?}");
        assert_eq!(vec_of_pairs, arr![Pair(1, 10), Pair(2, 20), Pair(3, 30)]);
    }

    // pub fn increase<'a>(m: &'a usize) -> State<'a, usize, usize>{
    //     State { run_state: Box::new(move |n| (m + n, m + n))}
    // }

    #[test]
    fn traversable() {
        use crate::functional::pair::Pair;
        let _pair = Pair(10, 20);
        // let res = pair.traverse(increase);
        // println!("{:?}", pair);
        // let pair_of_vecs = Pair(arr![1,2,3], arr![10, 20, 30]);
        // let transposed = pair_of_vecs.transpose();
        // println!("{:?}", transposed);
        // increase m = State (λn → (m + n, m + n))
    }

    #[test]
    fn innerprod() {
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let res = innerp(&v123, &v456);
        println!("{res:?}");
    }

    #[test]
    fn matrixprod() {
        let v123: Array<usize, _> = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let three_by_two: Array<GenericArray<_, _>, U2> = arr![v123, v456];
        let two_by_three = three_by_two.transpose();
        let res = matrixp(&two_by_three, &three_by_two);
        println!("{res:?}");
        assert_eq!(
            res,
            arr![arr![17, 22, 27], arr![22, 29, 36], arr![27, 36, 45]]
        );
    }

    #[test]
    fn hyper_map() {
        let v123 = arr![1, 2, 3];
        let v456 = arr![4, 5, 6];
        let two_by_three: Array<GenericArray<usize, _>, _> = arr![v123, v456];
        // let val = Prism::new(Scalar::new(two_by_three));
        let val = Scalar::new(two_by_three);
        println!("{:?}", val);
        println!("{:?}", val.innermost_dimension());
        println!("{:?}", val.amount_of_elems());
        let val = Scalar::new(two_by_three).lift();
        println!("{:?}", val);
        println!("{:?}", val.innermost_dimension());
        println!("{:?}", val.amount_of_elems());
        let val = Scalar::new(two_by_three).lift().lift();
        let first = val.first();
        println!("{:?}", val);
        println!("{:?}", val.innermost_dimension());
        println!("{:?}", val.amount_of_elems());
        println!("{:?}", first);

        let xs = Prism::hreplicate(42usize);
        println!("xs: {:?}", xs);
        assert!(xs > val);
    }

    // #[test]
    // fn hyper_first() {
    //     super::hyper_first();
    // }

    #[test]
    fn flattening() {
        let flat = arr![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        println!("{:?}", &flat);
        let tens = Tensor3::<usize, 2, 2, 3>::from_flat(flat);
        println!("{:?}", &tens);

        let four_by_three: Mat<usize, 4, 3> = tens.reshape();
        println!("{:?}", &four_by_three);
        let three_by_four: Mat<usize, 3, 4> = four_by_three.reshape();
        println!("{:?}", &three_by_four);

        let flat2 = arr!["hello", "world", "how", "are", "you", "doing"];
        let mat = Mat::<&'static str, 3, 2>::from_flat(flat2);
        println!("{:?}", &mat);
    }
}

#[doc(hidden)]
pub fn reshape_example(flat: Array<usize, U<20>>) -> Tensor3<usize, 2, 2, 5> {
    // let flat = arr![1,2,3,4,5,6,7,8,9,10,11,12];

    Tensor3::<usize, 2, 2, 5>::from_flat(flat)
    // println!("{:?}", &tens);
    // let vec: MatC<usize, 6, 2> = tens.reshape();

    // let four_by_three: Mat<usize, U4, U2> = tens.reshape();
    // println!("{:?}", &four_by_three);
    // let three_by_four: Mat<usize, U3, U4> = four_by_three.reshape();
    // println!("{:?}", &three_by_four);
}

#[doc(hidden)]
pub fn matrixprod(
    two_by_three: Array<GenericArray<usize, U2>, U3>,
    three_by_two: Array<GenericArray<usize, U3>, U2>,
) -> Array<GenericArray<usize, U3>, U3> {
    matrixp(&two_by_three, &three_by_two)
}

#[doc(hidden)]
pub fn hyper_first(v123: Array<usize, U3>, v456: GenericArray<usize, U3>) -> usize {
    // let v123 = arr![1, 2, 3];
    // let v456 = arr![4, 5, 6];
    let two_by_three: Array<GenericArray<usize, _>, _> = arr![v123, v456];
    let val = Scalar::new(two_by_three).lift().lift();
    *val.first()
}

#[doc(hidden)]
pub fn innerprod(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
    // use generic_array::arr;
    // let v123 = arr![1,2,3];
    // let v456 = arr![4,5,6];
    innerp(&v123, &v456)
}

// pub fn innerprod_orig() {
#[doc(hidden)]
pub fn innerprod_orig(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
    // use generic_array::arr;
    // let v123 = arr![1,2,3];
    // let v456 = arr![4,5,6];
    innerp_orig(&v123, &v456)
}

// fn foo() {
//     let mat = Mat::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//     let vec = Vect::from([1, 2, 3, 4, 5]);
//     let res = mat * vec; // <- Compile error!
// }
