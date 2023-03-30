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
pub type Tensor4<
    T,
    const BLOCKS: usize,
    const SLICES: usize,
    const ROWS: usize,
    const COLS: usize,
> = Prism<
    T,
    Tensor3<Array<T, U<COLS>>, BLOCKS, SLICES, ROWS>,
    U<COLS>,
    HCons<U<ROWS>, HCons<U<SLICES>, HCons<U<BLOCKS>, HNil>>>,
>;


use crate::functional::New;
use crate::hyper::Liftable;
use typenum::{NonZero, ToUInt, Const};
use generic_array::ArrayLength;
impl<T, const N: usize> From<[T; N]> for Vect<T, N>
where
    Const<N>: ToUInt,
    U::<N>: ArrayLength + NonZero
{
    fn from(value: [T; N]) -> Self {
        // SAFETY: Memory layout of [T; N] and GenericArray<T, U<N>>
        // is guaranteed to be is the same.
        let ga: Array<T, U<N>> = unsafe { core::mem::transmute_copy(&value) };
        Scalar::new(ga).lift()
    }
}

// impl<T, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]> for Mat<T, ROWS, COLS>
// where
//     Const<ROWS>: ToUInt,
//     Const<COLS>: ToUInt,
//     U::<ROWS>: ArrayLength + NonZero,
//     U::<COLS>: ArrayLength + NonZero,
// {
//     fn from(value: Array<Array<T, U::<COLS>>, U::<ROWS>>) -> Self {
//         Scalar::new(value).lift().lift()
//     }
// }

// impl<T, Slices, Rows, Cols> From<Array<Array<Array<T, Cols>, Rows>, Slices>> for Tensor3<T, Slices, Rows, Cols>
// where
//     Slices: ArrayLength + NonZero,
//     Rows: ArrayLength + NonZero,
//     Cols: ArrayLength + NonZero,
// {
//     fn from(value: Array<Array<Array<T, Cols>, Rows>, Slices>) -> Self {
//         Scalar::new(value).lift().lift().lift()
//     }
// }

// impl<T, Blocks, Slices, Rows, Cols> From<Array<Array<Array<Array<T, Cols>, Rows>, Slices>, Blocks>> for Tensor4<T, Blocks, Slices, Rows, Cols>
// where
//     Blocks: ArrayLength + NonZero,
//     Slices: ArrayLength + NonZero,
//     Rows: ArrayLength + NonZero,
//     Cols: ArrayLength + NonZero,
// {
//     fn from(value: Array<Array<Array<Array<T, Cols>, Rows>, Slices>, Blocks>) -> Self {
//         Scalar::new(value).lift().lift().lift().lift()
//     }
// }
