#![feature(type_name_of_val)]

pub mod aliases;
pub mod align;
pub mod common;
pub mod const_aliases;
pub mod fin;
pub mod functional;
pub mod hyper;
pub mod paper;

use align::Alignable;
use align::Maxed;
use common::Array;

use functional::{Container, Mappable, Mappable2, Naperian, New};

use hyper::AutoMappable2;
#[doc(inline)]
pub use hyper::Hyper;
use hyper::{Prism, Scalar};
use paper::innerp_orig;

#[doc(inline)]
pub use const_aliases::*;

use std::iter::Sum;

use std::ops::Mul;

use generic_array::{arr, GenericArray};
use typenum::consts::*;

use typenum::U;

use frunk::hlist::{HCons, HNil};

/// Transpose a `F<G<A>>` into `G<F<A>>` provided both `F` and `G` implement [`Naperian`].
/// (and A: [`Clone`] since we need to copy a bunch of `A`'s around.)
///
/// There is no need to implement this trait manually since there is a blanket implementation
/// for all types implementing Naperian.
pub trait NaperianTranspose<G, A: Clone>
where
    Self: Naperian<G>,
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{
    fn transpose(&self) -> G::Containing<Self::Containing<A>> {
        Naperian::tabulate(|x| Naperian::tabulate(|y| self.lookup(y).lookup(x).clone()))
    }
}

impl<G, A: Clone, Nap: ?Sized + Naperian<G>> NaperianTranspose<G, A> for Nap
where
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{
}

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
    A: Clone + std::iter::Sum + std::ops::Mul<Output = A>,
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

#[doc(hidden)]
pub fn foo() -> Tensor3<usize, 2, 2, 3> {
    let _v: Vect<usize, 3> = Prism::build(Scalar::new(arr![1, 2, 3]));
    let mat: Mat<usize, 2, 3> = Prism::build(Prism::build(Scalar::new(arr![
        arr![1, 2, 3],
        arr![4, 5, 6]
    ])));
    // let tens: Tensor3<usize, U2, U3, U1> = Prism::build(Prism::build(Prism::build(Scalar::new(arr![arr![arr![1,2,3], arr![4,5,6]]]))));
    let tens: Tensor3<usize, 2, 2, 3> =
        Prism::build(Prism::build(Prism::build(Scalar::new(arr![
            arr![arr![1, 2, 3], arr![4, 5, 6]],
            arr![arr![7, 8, 9], arr![10, 11, 12]]
        ]))));
    println!("{:?}", &mat);
    println!("{:?}", &tens);
    let flat: &Array<usize, U12> = unsafe { core::mem::transmute(&tens) };
    println!("flat: {:?}", &flat);
    // println!("{:?}", core::any::type_name_of_val(&tens));
    tens
}

#[doc(hidden)]
pub fn alignment() {
    let _v: Vect<usize, 3> = Prism::build(Scalar::new(arr![1, 2, 3]));
    let mat: Mat<usize, 2, 3> = Prism::build(Prism::build(Scalar::new(arr![
        arr![1, 2, 3],
        arr![4, 5, 6]
    ])));
    let mat2: Tensor3<usize, 3, 2, 3> = mat.align();
    // println!("mat: {:?}", mat);
    println!("mat2: {mat2:?}");
}

use core::ops::Sub;
impl<A, B, C, HypA, HypB, HypC, HypAAligned, HypBAligned> Sub<HypB> for Scalar<A>
where
    Self: Container<Containing<A> = HypA> + AutoMappable2<HypB, HypAAligned, HypBAligned, HypC, A, B, C>,
    HypA: Hyper<Elem = A>,
    HypB: Hyper<Elem = B>,
    A: Sub<B, Output = C>,
    HypB: Maxed<HypA, HypBAligned> + align::Max<Scalar<A>, Output = HypBAligned>,
    HypA: Maxed<HypB, HypAAligned> + align::Max<HypB, Output = HypAAligned>,
    HypAAligned: Hyper<Elem = A> + Container<Containing<B> = HypBAligned> + Container<Containing<C> = HypC> + Mappable2<A, C>,
    HypBAligned: Hyper<Elem = B, AmountOfElems = HypAAligned::AmountOfElems> + Container<Containing<B> = HypBAligned>,
    HypC: Hyper<Elem = C, AmountOfElems = HypAAligned::AmountOfElems>,
{
    type Output = HypC;

    fn sub(self, rhs: HypB) -> HypC {
        self.map2(rhs, Sub::sub)
    }
}

impl<A, B, C, HypA, HypB, HypC, HypAAligned, HypBAligned, As, N, Ns> Sub<HypB> for Prism<A, As, N, Ns>
where
    Self: Container<Containing<A> = HypA> + AutoMappable2<HypB, HypAAligned, HypBAligned, HypC, A, B, C>,
    HypA: Hyper<Elem = A>,
    HypB: Hyper<Elem = B>,
    A: Sub<B, Output = C>,
    HypB: Maxed<HypA, HypBAligned> + align::Max<Self, Output = HypBAligned>,
    HypA: Maxed<HypB, HypAAligned> + align::Max<HypB, Output = HypAAligned>,
    HypAAligned: Hyper<Elem = A> + Container<Containing<B> = HypBAligned> + Container<Containing<C> = HypC> + Mappable2<A, C>,
    HypBAligned: Hyper<Elem = B, AmountOfElems = HypAAligned::AmountOfElems> + Container<Containing<B> = HypBAligned>,
    HypC: Hyper<Elem = C, AmountOfElems = HypAAligned::AmountOfElems>,
{
    type Output = HypC;

    fn sub(self, rhs: HypB) -> HypC {
        self.map2(rhs, Sub::sub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_subtraction() {
        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let scalar = Scalar::new(1);
        let res = mat - scalar;
        println!("{:?}", res);
    }

    #[test]
    fn binary() {
        use hyper::AutoMappable2;
        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let vec = Vect::<usize, 3>::from_flat(arr![10, 20, 30]);
        let res = mat.map2(vec, |x, y| x + y);
        println!("{:?}", res);

        let mat = Mat::<usize, 2, 3>::from_flat(arr![1, 2, 3, 4, 5, 6]);
        let matsum = (&mat).map2(&mat, |x, y| x + y);
        println!("matsum: {:?}", matsum);
    }

    #[test]
    fn alignment() {
        super::alignment();
    }

    #[test]
    fn foofoo() {
        let val = super::foo();
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
        let val = Prism::build(Scalar::new(two_by_three));
        println!("{:?}", val);
        println!("{:?}", val.innermost_dimension());
        println!("{:?}", val.amount_of_elems());
        let val = Prism::build(Prism::build(Scalar::new(two_by_three)));
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

pub fn matrixprod(
    two_by_three: Array<GenericArray<usize, U2>, U3>,
    three_by_two: Array<GenericArray<usize, U3>, U2>,
) -> Array<GenericArray<usize, U3>, U3> {
    matrixp(&two_by_three, &three_by_two)
}

pub fn hyper_first(v123: Array<usize, U3>, v456: GenericArray<usize, U3>) -> usize {
    // let v123 = arr![1, 2, 3];
    // let v456 = arr![4, 5, 6];
    let two_by_three: Array<GenericArray<usize, _>, _> = arr![v123, v456];
    let val = Prism::build(Prism::build(Scalar::new(two_by_three)));
    *val.first()
}

// pub fn innerprod(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
//     // use generic_array::arr;
//     // let v123 = arr![1,2,3];
//     // let v456 = arr![4,5,6];
//     innerp(&v123, &v456)
// }

// // pub fn innerprod_orig() {
// pub fn innerprod_orig(v123: Array<usize, U10>, v456: GenericArray<usize, U10>) -> usize {
//     // use generic_array::arr;
//     // let v123 = arr![1,2,3];
//     // let v456 = arr![4,5,6];
//     innerp_orig(&v123, &v456)
// }
