use generic_array::ArrayLength;
use typenum::NonZero;

use super::functional::New;
use super::hyper::Liftable;
use super::{Array, Prism, TCons, TNil};

/// A single scalar value.
///
/// This is a rank-0 tensor.
pub type Scalar<T> = super::Scalar<T>;

/// Vect, a Vector with a statically-known size.
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Vect<T, N> = Prism<T, Scalar<Array<T, N>>, N, TNil>;

/// Mat, a Matrix with a statically-known dimensions (rows, colums).
///
/// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Mat<T, Rows, Cols> = Prism<T, Vect<Array<T, Cols>, Rows>, Cols, TCons<Rows, TNil>>;

/// Rank-3 tensors (slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`Hyper`] trait which contains many common operations.
pub type Tensor3<T, Slices, Rows, Cols> =
    Prism<T, Mat<Array<T, Cols>, Slices, Rows>, Cols, TCons<Rows, TCons<Slices, TNil>>>;

/// Rank-4 tensors (blocks, slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor4<T, Blocks, Slices, Rows, Cols> = Prism<
    T,
    Tensor3<Array<T, Cols>, Blocks, Slices, Rows>,
    Cols,
    TCons<Rows, TCons<Slices, TCons<Blocks, TNil>>>,
>;

impl<T, N> From<Array<T, N>> for Vect<T, N>
where
    N: ArrayLength + NonZero,
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

impl<T, Slices, Rows, Cols> From<Array<Array<Array<T, Cols>, Rows>, Slices>>
    for Tensor3<T, Slices, Rows, Cols>
where
    Slices: ArrayLength + NonZero,
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
{
    fn from(value: Array<Array<Array<T, Cols>, Rows>, Slices>) -> Self {
        Scalar::new(value).lift().lift().lift()
    }
}

impl<T, Blocks, Slices, Rows, Cols> From<Array<Array<Array<Array<T, Cols>, Rows>, Slices>, Blocks>>
    for Tensor4<T, Blocks, Slices, Rows, Cols>
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
impl<T, Rows, Cols> Mat<T, Rows, Cols>
where
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
    Self: Hyper,
    Vect<Array<T, Cols>, Rows>: super::Hyper<Dimensions = TCons<Rows, TNil>>,
{
    /// Returns a reference to this [`Mat`], viewed as an array of its rows.
    ///
    /// Since a [`Mat`] is stored in row-major order, this can be done without moving elements around.
    pub fn rows(&self) -> &Array<Vect<T, Cols>, Rows> {
        unsafe { core::mem::transmute(&self.0) } // self.0.orig()
    }

    /// Consumes this [`Mat`], turning it into an array of its rows.
    ///
    /// Since a [`Mat`] is stored in row-major order, this can be done without moving elements around.
    pub fn into_rows(self) -> Array<Vect<T, Cols>, Rows> {
        unsafe { core::mem::transmute_copy(&self.0) }
        // self.lower().into_orig()
    }
}

impl<T: Clone, Rows, Cols> Mat<T, Rows, Cols>
where
    Rows: ArrayLength + NonZero,
    Cols: ArrayLength + NonZero,
    Self: Hyper + HyperTranspose<Transposed = Mat<T, Cols, Rows>>,
    Mat<T, Cols, Rows>: Hyper,
    Vect<Array<T, Rows>, Cols>:
        super::Hyper<Dimensions = TCons<Cols, TNil>, Orig = Array<Vect<T, Rows>, Cols>>,
{
    /// Consumes this [`Mat`], turning it into an array of its columns.
    ///
    /// This requires moving elements around.
    pub fn into_columns(self) -> Array<Vect<T, Rows>, Cols> {
        self.transpose().into_rows()
    }
}
