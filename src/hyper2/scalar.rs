use core::any::TypeId;

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

