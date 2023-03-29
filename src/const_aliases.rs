
use super::{Array, HCons, HNil, Prism};
use typenum::U;

/// A single scalar value.
///
/// This is a rank-0 tensor.
pub type Scalar<T> = super::Scalar<T>;

/// Vect, a Vector with a statically-known size.
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Vect<T, const N: usize> = Prism<T, Scalar<Array<T, U<N>>>, U<N>, HNil>;

/// Mat, a Matrix with a statically-known dimensions (rows, colums).
///
/// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Mat<T, const ROWS: usize, const COLS: usize> =
    Prism<T, Vect<Array<T, U<COLS>>, ROWS>, U<COLS>, HCons<U<ROWS>, HNil>>;

/// Rank-3 tensors (slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor3<T, const SLICES: usize, const ROWS: usize, const COLS: usize> =
    Prism<T, Mat<Array<T, U<COLS>>, SLICES, ROWS>, U<COLS>, HCons<U<ROWS>, HCons<U<SLICES>, HNil>>>;

/// Rank-4 tensors (blocks, slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor4<T, const BLOCKS: usize, const SLICES: usize, const ROWS: usize, const COLS: usize> =
    Prism<T, Tensor3<Array<T, U<COLS>>, BLOCKS, SLICES, ROWS>, U<COLS>, HCons<U<ROWS>, HCons<U<SLICES>, HCons<U<BLOCKS>, HNil>>>>;
