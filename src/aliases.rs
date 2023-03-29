
use super::{Array, HCons, HNil, Prism, Scalar};

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
