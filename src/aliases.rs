use generic_array::ArrayLength;
use typenum::NonZero;

use super::{Array, HCons, HNil, Prism};
use super::functional::New;
use super::hyper::Liftable;

/// A single scalar value.
///
/// This is a rank-0 tensor.
pub type Scalar<T> = super::Scalar<T>;

/// Vect, a Vector with a statically-known size.
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Vect<T, N> = Prism<T, Scalar<Array<T, N>>, N, HNil>;

/// Mat, a Matrix with a statically-known dimensions (rows, colums).
///
/// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Mat<T, Rows, Cols> = Prism<T, Vect<Array<T, Cols>, Rows>, Cols, HCons<Rows, HNil>>;

/// Rank-3 tensors (slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Tensor3<T, Slices, Rows, Cols> =
    Prism<T, Mat<Array<T, Cols>, Slices, Rows>, Cols, HCons<Rows, HCons<Slices, HNil>>>;

/// Rank-4 tensors (blocks, slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor4<T, Blocks, Slices, Rows, Cols> = Prism<
    T,
    Tensor3<Array<T, Cols>, Blocks, Slices, Rows>,
    Cols,
    HCons<Rows, HCons<Slices, HCons<Blocks, HNil>>>,
>;

impl<T, N> From<Array<T, N>> for Vect<T, N>
where
    N: ArrayLength + NonZero
{
    fn from(value: Array<T, N>) -> Self {
        Scalar::new(value).lift()
    }
}

impl<T, Rows, Cols> From<Array<Array<T, Cols>, Rows>> for Mat<T, Rows, Cols>
where
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
{
    fn from(value: Array<Array<T, Cols>, Rows>) -> Self {
        Scalar::new(value).lift().lift()
    }
}

impl<T, Slices, Rows, Cols> From<Array<Array<Array<T, Cols>, Rows>, Slices>> for Tensor3<T, Slices, Rows, Cols>
where
    Slices: ArrayLength + NonZero,
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
{
    fn from(value: Array<Array<Array<T, Cols>, Rows>, Slices>) -> Self {
        Scalar::new(value).lift().lift().lift()
    }
}

impl<T, Blocks, Slices, Rows, Cols> From<Array<Array<Array<Array<T, Cols>, Rows>, Slices>, Blocks>> for Tensor4<T, Blocks, Slices, Rows, Cols>
where
    Blocks: ArrayLength + NonZero,
    Slices: ArrayLength + NonZero,
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
{
    fn from(value: Array<Array<Array<Array<T, Cols>, Rows>, Slices>, Blocks>) -> Self {
        Scalar::new(value).lift().lift().lift().lift()
    }
}

use super::hyper::{Hyper, HyperTranspose};
impl<T, Rows, Cols, Ts> Mat<T, Rows, Cols>
where
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
    Self: Hyper<Inner = Ts>,
Vect<Array<T, Cols>, Rows>: super::Hyper<Dimensions = HCons<Rows, HNil>, Orig = Array<Vect<T, Cols>, Rows>>,
{
    pub fn rows(self) -> Array<Vect<T, Cols>, Rows> {
        self.lower().into_orig()
    }
}

impl<T: Clone, Rows, Cols, Ts> Mat<T, Rows, Cols>
where
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
    Self: Hyper + HyperTranspose<Transposed = Mat<T, Cols, Rows>>,
    Mat<T, Cols, Rows>: Hyper<Inner = Ts>,
Vect<Array<T, Rows>, Cols>: super::Hyper<Dimensions = HCons<Cols, HNil>, Orig = Array<Vect<T, Rows>, Cols>>,
{
    pub fn cols(self) -> Array<Vect<T, Rows>, Cols> {
        self.transpose().rows()
    }
}
