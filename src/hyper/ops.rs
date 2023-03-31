use concat_idents::concat_idents;
use core::ops::{Add, Sub, Mul, Div, Rem, BitAnd, BitOr, BitXor, Shr, Shl};
use crate::compat::{TensorCompatible, Elem, Tensor};
use crate::functional::{Container, Mappable2, Mappable};
use crate::functional::tlist::{TList, TCons, Compatible};
use super::{Scalar, Prism, Hyper, ShapeMatched, HyperMappable2};
use crate::align;

macro_rules! impl_binop {
    ($op_trait:ident::$op:ident) => {
        concat_idents!(binop_trait_impl = $op_trait, Impl {
            trait binop_trait_impl<T: TensorCompatible, HL = <T as TensorCompatible>::Kind> {
                type Output;
                fn binop(self, rhs: T) -> Self::Output;
            }

            // Elem *op* Scalar
            impl<A, B> binop_trait_impl<B, Elem> for Scalar<A>
            where
                B: TensorCompatible<Kind = Elem>,
                A: $op_trait<B>,
            {
                type Output = Scalar<<A as $op_trait<B>>::Output>;
                fn binop(self, rhs: B) -> Self::Output {
                    // println!("Boom");
                    Scalar($op_trait::$op(self.0, rhs))
                }
            }

            // Scalar *op* Elem
            impl<A, B> binop_trait_impl<Scalar<B>, Elem> for A
            where
                A: TensorCompatible<Kind = Elem>,
                A: $op_trait<B>,
            {
                type Output = Scalar<<A as $op_trait<B>>::Output>;

                fn binop(self, rhs: Scalar<B>) -> Self::Output {
                    // println!("Boom");
                    Scalar($op_trait::$op(self, rhs.0))
                }
            }



            // Tensor *op* Scalar
            impl<HypA, HypB, HypC, HypAAligned, HypBAligned, A, B, C> binop_trait_impl<HypB, Tensor> for Scalar<A>
            where
                A: $op_trait<B, Output = C>,
                HypB: TensorCompatible<Kind = Tensor>,
                Self: Container<Containing<A> = HypA> + HyperMappable2<HypB, HypAAligned, HypBAligned, HypC, A, B, C>,
                HypA: Hyper<Elem = A>,
                HypB: Hyper<Elem = B>,
                HypB: ShapeMatched<HypA, HypBAligned> + align::MatchingShape<Scalar<A>, Output = HypBAligned>,
                HypA: ShapeMatched<HypB, HypAAligned> + align::MatchingShape<HypB, Output = HypAAligned>,
                HypAAligned: Hyper<Elem = A> + Container<Containing<B> = HypBAligned> + Container<Containing<C> = HypC> + Mappable2<A, C>,
                HypBAligned: Hyper<Elem = B, AmountOfElems = HypAAligned::AmountOfElems> + Container<Containing<B> = HypBAligned>,
                HypC: Hyper<Elem = C, AmountOfElems = HypAAligned::AmountOfElems>,
            {
                type Output = HypC;
                fn binop(self, rhs: HypB) -> HypC {
                    self.map2(rhs, $op_trait::$op)
                }
            }

            // Elem *op* Prism
            impl<A, B, As, N, Ns> binop_trait_impl<B, Elem> for Prism<A, As, N, Ns>
            where
                A: $op_trait<B>,
                B: TensorCompatible<Kind = Elem>,
                Self: Mappable<<A as $op_trait<B>>::Output> + Container<Elem = A>,
                Ns: TList,
                B: Clone,
            {
                type Output = <Self as Container>::Containing<<A as $op_trait<B>>::Output>;
                fn binop(self, rhs: B) -> Self::Output {
                    self.map_by_value(|x| $op_trait::$op(x, rhs.clone()))
                }
            }

            // Prism *op* Elem
            impl<A, B, Bs, N, Ns> binop_trait_impl<Prism<B, Bs, N, Ns>, Elem> for A
            where
                A: $op_trait<B>,
                A: TensorCompatible<Kind = Elem>,
                Prism<B, Bs, N, Ns>: Mappable<<A as $op_trait<B>>::Output> + Container<Elem = B>,
                Ns: TList,
                A: Clone,
            {
                type Output = <Prism<B, Bs, N, Ns> as Container>::Containing<<A as $op_trait<B>>::Output>;
                fn binop(self, rhs: Prism<B, Bs, N, Ns>) -> Self::Output {
                    rhs.map_by_value(|x| $op_trait::$op(self.clone(), x))
                }
            }

            // Tensor *op* Prism
            impl<HypA, HypB, HypC, HypAAligned, HypBAligned, A, B, C, As, N, Ns> binop_trait_impl<HypB, Tensor> for Prism<A, As, N, Ns>
            where
                A: $op_trait<B, Output = C>,
                HypB: TensorCompatible<Kind = Tensor>,
                Self: Container<Containing<A> = HypA> + HyperMappable2<HypB, HypAAligned, HypBAligned, HypC, A, B, C>,
                HypA: Hyper<Elem = A>,
                HypB: Hyper<Elem = B>,
                HypA::Dimensions: Compatible<HypB::Dimensions>,
                HypB: ShapeMatched<HypA, HypBAligned> + align::MatchingShape<Self, Output = HypBAligned>,
                HypA: ShapeMatched<HypB, HypAAligned> + align::MatchingShape<HypB, Output = HypAAligned>,
                HypAAligned: Hyper<Elem = A> + Container<Containing<B> = HypBAligned> + Container<Containing<C> = HypC> + Mappable2<A, C>,
                HypBAligned: Hyper<Elem = B, AmountOfElems = HypAAligned::AmountOfElems> + Container<Containing<B> = HypBAligned>,
                HypC: Hyper<Elem = C, AmountOfElems = HypAAligned::AmountOfElems>,
                Ns: TList,
            {
                type Output = HypC;
                fn binop(self, rhs: HypB) -> HypC {
                    self.map2(rhs, $op_trait::$op)
                }
            }

            impl<A, B, Kind, C> $op_trait<B> for Scalar<A>
            where
                Self: binop_trait_impl<B, Kind, Output = C>,
                B: TensorCompatible<Kind = Kind>,
            {
                type Output = C;
                fn $op(self, rhs: B) -> C {
                    binop_trait_impl::binop(self, rhs)
                }
            }

            impl<A, B, C, Kind, As, N, Ns> $op_trait<B> for Prism<A, As, N, Ns>
            where
                Self: binop_trait_impl<B, Kind, Output = C>,
                B: TensorCompatible<Kind = Kind>,
                Ns: TList,
            {
                type Output = C;
                fn $op(self, rhs: B) -> C {
                    binop_trait_impl::binop(self, rhs)
                }
            }
        });
    }
}

impl_binop!(Add::add);
impl_binop!(Sub::sub);
impl_binop!(Mul::mul);
impl_binop!(Div::div);
impl_binop!(Rem::rem);
impl_binop!(BitAnd::bitand);
impl_binop!(BitOr::bitor);
impl_binop!(BitXor::bitxor);
impl_binop!(Shr::shr);
impl_binop!(Shl::shl);
