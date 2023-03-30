use typenum::consts::*;
use typenum::Unsigned;

pub trait IsTrue {}
impl IsTrue for B1 {}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Fin<N: Unsigned> {
    val: usize,
    _phantom: core::marker::PhantomData<N>,
}

impl<N: Unsigned> core::fmt::Debug for Fin<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fin")
            .field("val", &self.val)
            .field("bound", &N::USIZE)
            .finish()
    }
}

use const_format::formatcp;
impl<N: Unsigned> Fin<N> {
    /// Creates a new Fin from a usize. Fallible, checked at runtime.
    ///
    /// A BoundError is returned when the usize is larger than the maximum bound (`N`).
    #[inline(always)]
    pub const fn new(val: usize) -> Result<Self, BoundError<N>> {
        if val < N::USIZE {
            Ok(Fin {
                val,
                _phantom: core::marker::PhantomData,
            })
        } else {
            Err(BoundError::<N>::new())
        }
    }

    /// Creates a new Fin from a usize.
    ///
    /// # Safety
    /// The caller is responsible for making sure that `val` is smaller than `N::USIZE`.
    pub unsafe fn new_unchecked(val: usize) -> Self {
        Fin {
            val,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Creates a new Fin from an unsigned typenum.
    ///
    /// Outcome is always valid as too large values result in a compile error.
    #[inline(always)]
    pub const fn tnew<Val>() -> Self
    where
        Val: Unsigned + typenum::IsLess<N>,
        typenum::Le<Val, N>: IsTrue,
    {
        Fin {
            val: Val::USIZE,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Creates a new Fin from an compile-time constant usize.
    ///
    /// Outcome is always valid as too large values result in a compile error.
    #[inline(always)]
    pub const fn cnew<const VAL: usize>() -> Self
    where
        typenum::Const<VAL>: typenum::ToUInt,
        typenum::U<VAL>: Unsigned + typenum::IsLess<N>,
        typenum::Le<typenum::U<VAL>, N>: IsTrue,
    {
        Fin {
            val: VAL,
            _phantom: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub const fn bound() -> usize {
        N::USIZE
    }

    /// Returns the bounded value stored inside as a normal, unbounded, usize
    pub fn val(&self) -> usize {
        self.val
    }
}

#[derive(Debug)]
pub struct BoundError<N: Unsigned> {
    _phantom: core::marker::PhantomData<N>,
}

impl<N: Unsigned> BoundError<N> {
    pub const fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<N: Unsigned> TryInto<Fin<N>> for usize {
    type Error = BoundError<N>;
    fn try_into(self) -> Result<Fin<N>, Self::Error> {
        Fin::<N>::new(self)
    }
}

pub trait UnsignedExt: Unsigned {
    #[inline(always)]
    fn fin<N: Unsigned>() -> Fin<N>
    where
        Self: Unsigned + typenum::IsLess<N>,
        typenum::Le<Self, N>: IsTrue,
    {
        Fin::<N>::tnew::<Self>()
    }
}

impl<Index: Unsigned> UnsignedExt for Index {}

pub fn fin<const VAL: usize, N: Unsigned>() -> Fin<N>
where
    typenum::Const<VAL>: typenum::ToUInt,
    typenum::U<VAL>: Unsigned + typenum::IsLess<N>,
    typenum::Le<typenum::U<VAL>, N>: IsTrue,
{
    Fin::<N>::cnew::<VAL>()
}
