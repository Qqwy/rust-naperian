use super::{Array, Prism, TCons, TNil};
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
pub type Vect<T, const N: usize> = Prism<T, Scalar<Array<T, U<N>>>, U<N>, TNil>;

/// Mat, a Matrix with a statically-known dimensions (rows, colums).
///
/// Matrices are stored in [Row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Mat<T, const ROWS: usize, const COLS: usize> =
    Prism<T, Vect<Array<T, U<COLS>>, ROWS>, U<COLS>, TCons<U<ROWS>, TNil>>;

/// Rank-3 tensors (slices, rows, columns).
///
/// This is a type alias.
/// During normal usage you do not need to understand the backing type,
/// only that it implements the [`super::Hyper`] trait which contains many common operations.
pub type Tensor3<T, const SLICES: usize, const ROWS: usize, const COLS: usize> =
    Prism<T, Mat<Array<T, U<COLS>>, SLICES, ROWS>, U<COLS>, TCons<U<ROWS>, TCons<U<SLICES>, TNil>>>;

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
    TCons<U<ROWS>, TCons<U<SLICES>, TCons<U<BLOCKS>, TNil>>>,
>;

use crate::functional::{Mappable, New};
use crate::hyper::Liftable;
use generic_array::ArrayLength;
use typenum::{Const, NonZero, ToUInt};
impl<T, const N: usize> From<[T; N]> for Vect<T, N>
where
    Const<N>: ToUInt,
    U<N>: ArrayLength + NonZero,
{
    fn from(value: [T; N]) -> Self {
        Scalar::new(generic_array_from(value)).lift()
    }
}

impl<T, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]> for Mat<T, ROWS, COLS>
where
    Const<ROWS>: ToUInt,
    Const<COLS>: ToUInt,
    U<ROWS>: ArrayLength + NonZero,
    U<COLS>: ArrayLength + NonZero,
{
    fn from(value: [[T; COLS]; ROWS]) -> Self {
        let arr = generic_array_from(value).map_by_value(generic_array_from);
        Scalar::new(arr).lift().lift()
    }
}

impl<T, const SLICES: usize, const ROWS: usize, const COLS: usize> From<[[[T; COLS]; ROWS]; SLICES]>
    for Tensor3<T, SLICES, ROWS, COLS>
where
    Const<SLICES>: ToUInt,
    Const<COLS>: ToUInt,
    Const<ROWS>: ToUInt,
    U<SLICES>: ArrayLength + NonZero,
    U<ROWS>: ArrayLength + NonZero,
    U<COLS>: ArrayLength + NonZero,
{
    fn from(value: [[[T; COLS]; ROWS]; SLICES]) -> Self {
        let arr = generic_array_from(value)
            .map_by_value(generic_array_from)
            .map_by_value(|rows| rows.map_by_value(generic_array_from));
        Scalar::new(arr).lift().lift().lift()
    }
}

impl<T, const BLOCKS: usize, const SLICES: usize, const ROWS: usize, const COLS: usize>
    From<[[[[T; COLS]; ROWS]; SLICES]; BLOCKS]> for Tensor4<T, BLOCKS, SLICES, ROWS, COLS>
where
    Const<BLOCKS>: ToUInt,
    Const<SLICES>: ToUInt,
    Const<COLS>: ToUInt,
    Const<ROWS>: ToUInt,
    U<BLOCKS>: ArrayLength + NonZero,
    U<SLICES>: ArrayLength + NonZero,
    U<ROWS>: ArrayLength + NonZero,
    U<COLS>: ArrayLength + NonZero,
{
    fn from(value: [[[[T; COLS]; ROWS]; SLICES]; BLOCKS]) -> Self {
        let arr = generic_array_from(value)
            .map_by_value(generic_array_from)
            .map_by_value(|slices| {
                slices
                    .map_by_value(generic_array_from)
                    .map_by_value(|rows| rows.map_by_value(generic_array_from))
            });

        Scalar::new(arr).lift().lift().lift().lift()
    }
}

// Needed until https://github.com/fizyk20/generic-array/issues/140 is fixed.
// GenericArray only has some hard-coded From/Into impls at this time.
fn generic_array_from<T, const N: usize>(arr: [T; N]) -> Array<T, U<N>>
where
    Const<N>: ToUInt,
    U<N>: ArrayLength,
{
    // SAFETY: Memory layout of [T; N] and GenericArray<T, U<N>>
    // is guaranteed to be is the same.
    unsafe { core::mem::transmute_copy(&arr) }
}
