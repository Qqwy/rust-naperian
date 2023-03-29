use super::{Scalar, Prism};

pub enum Elem{}
pub enum Tensor{}
pub trait TensorCompatibilityOption {}
impl TensorCompatibilityOption for Elem {}
impl TensorCompatibilityOption for Tensor {}

/// Trait to allow binary operators to be used
/// where one of the elements is a non-tensor element.
///
///
/// If you ever encounter the error "the trait `TensorCompatible` is not implemented for `MyType`", then you need to implement this trait for your struct and the binary operator will work.
///
/// # How?
///
/// The following implementation should suffice.
/// At some point a deriving macro for this trait would be a nice feature to add to this library :-).
///
/// ```ignore
/// impl naperian::compat::TensorCompatible for MyType {
///     type Kind = naperian::compat::Elem;
/// }
/// ```
///
/// # Why?
///
/// This trait (and the other machinery in this module) exists to
/// work around a limitation in the Rust compiler:
/// Working with two mutually exclusive blanket implementations of a trait.
///
/// In this case, for each binary operator, we have two implementations of the binary operator:
/// - One for Tensor <-> Tensor
/// - One for Tensor <-> NonTensor.
///
/// It is impossible for these two implementations to _not_ overlap,
/// unless a technique like this is used.
///
/// See [this blog article](https://geo-ant.github.io/blog/2021/mutually-exclusive-traits-rust/) for more info.
///
/// # I want to use a struct that I have no control over
/// You cannot implement this trait for structs outside of your own code
/// (because of the so-called orphan instances rule.)
///
/// But what you can always do, is to wrap the structs in a call to [`Scalar::new`].
/// This will always work.
pub trait TensorCompatible {
    type Kind: TensorCompatibilityOption;
}


trait IsTensor: TensorCompatible<Kind = Tensor> {}
trait IsElem: TensorCompatible<Kind = Elem> {}

impl<T: TensorCompatible<Kind = Elem>> IsElem for T {}
impl<T: TensorCompatible<Kind = Tensor>> IsTensor for T {}

impl<T> TensorCompatible for Scalar<T> {
    type Kind = Tensor;
}
impl<T, Ts, N, Ns> TensorCompatible for Prism<T, Ts, N, Ns> {
    type Kind = Tensor;
}

impl TensorCompatible for usize {
    type Kind = Elem;
}

impl TensorCompatible for isize {
    type Kind = Elem;
}

impl TensorCompatible for i32 {
    type Kind = Elem;
}

impl TensorCompatible for i64 {
    type Kind = Elem;
}

impl TensorCompatible for u32 {
    type Kind = Elem;
}

impl TensorCompatible for u64 {
    type Kind = Elem;
}

impl TensorCompatible for i8 {
    type Kind = Elem;
}

impl TensorCompatible for u8 {
    type Kind = Elem;
}
