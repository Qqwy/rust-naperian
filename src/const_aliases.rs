
use super::{Array, HCons, HNil, Prism, Scalar};
use typenum::U;

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

/// Vect, a Vector with a statically-known size.
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor3<T, const SLICES: usize, const ROWS: usize, const COLS: usize> =
    Prism<T, Mat<Array<T, U<COLS>>, SLICES, ROWS>, U<COLS>, HCons<U<ROWS>, HCons<U<SLICES>, HNil>>>;
