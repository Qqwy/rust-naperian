use core::{any::TypeId, marker::PhantomData};

use generic_array::ArrayLength;
use typenum::NonZero;

use crate::{functional::tlist::{TCons, TNil}, hyper2::Prism, common::Array};
use super::Liftable;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
#[repr(transparent)]
pub struct Scalar<T>(pub(crate) T);

impl<T> core::fmt::Debug for Scalar<T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Scalar")
            .field(&crate::helper::debug_typename::<T>())
            .field(&self.0)
            .finish()
    }
}

impl<T, N> Liftable for Scalar<Array<T, N>>
where
    N: ArrayLength + NonZero,
{
    type Lifted = Prism<T, TCons<N, TNil>>;
    fn lift(self) -> Self::Lifted {
        Prism(self.0, PhantomData)
    }
}
