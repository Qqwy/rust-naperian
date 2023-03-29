//! Code which follows the Naperian paper closely.
//!
//! This module is not intended for normal every-day usage.
//!
//! Included in the crate for two reasons:
//! 1. Sometimes used internally by the public-facing API
//! 2. It can help you to understand how this crate is built/how faithful the implementation is to the paper.
use core::ops::Mul;
use core::iter::Sum;
use crate::functional::{Container, Mappable2};

/// Version of innerp precisely following the Naperian paper.
///
/// This is evaluated more strictly than desired; it will first create an intermediate container
/// with all the products, and then sum the elements in this container.
pub fn innerp_orig<A, R>(a: &A, b: &A) -> R
where
    A: Mappable2<R, R> + Container<Containing<R> = A> + IntoIterator,
    R: Sum<<A as std::iter::IntoIterator>::Item>,
    R: Mul<R, Output = R> + Clone,
{
    let products = a.map2(b, |x: &R, y: &R| x.clone() * y.clone());

    products.into_iter().sum()
}
