use core::marker::PhantomData;

pub(crate) fn debug_typename<T>() -> TypeName<T> {
    TypeName::<T>::of()
}

pub(crate) struct TypeName<T>(PhantomData<T>);

impl<T> TypeName<T> {
    pub(crate) fn of() -> Self {
        TypeName(PhantomData)
    }
}


impl<T> core::fmt::Debug for TypeName<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", simplified_typename::<T>())
    }
}

#[cfg(not(feature = "alloc"))]
pub(crate) fn simplified_typename<T: ?Sized>() -> &'static str {
    core::any::type_name::<T>()
}

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub(crate) fn simplified_typename<T: ?Sized>() -> alloc::string::String {
    extern crate tynm;
    tynm::type_name::<T>()
}


#[cfg(feature = "alloc")]
pub fn type_name_of_val<T: ?Sized>(_val: &T) -> alloc::string::String {
    simplified_typename::<T>()
}


#[cfg(not(feature = "alloc"))]
pub fn type_name_of_val<T: ?Sized>(_val: &T) -> &'static str {
    simplified_typename::<T>()
}
