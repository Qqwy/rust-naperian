#![feature(type_name_of_val)]
use sealed::sealed;
use typenum::Unsigned;
use typenum::consts::*;
use typenum::{IsLess, Bit};
use core::ops::Add;
use std::fmt::Debug;

// TODO Make optional based on const-generics feature
use typenum::{U, Const, ToUInt};

#[sealed]
pub trait TVec<T>: Sized {
    type Len: Unsigned;
    fn len(&self) -> usize {
        Self::Len::to_usize()
    }

    fn prepend(self, h: T) -> TVecCons<T, Self> {
        TVecCons {
            head: h,
            tail: self
        }
    }

    // fn lookup<Index: Unsigned + IsLess<Self::Len> + typenum::Cmp<typenum::UTerm>>(self) -> T;
}

// pub trait Lookup<T, Index>: TVec<T> {
//     fn lookup(self) -> T;
// }

// impl<T, Tail, Index> Lookup<T, Index> for TVecCons<T, Tail>
// where
//     Tail: TVec<T>,
//     <Tail as TVec<T>>::Len: core::ops::Add<B1>,
//     <<Tail as TVec<T>>::Len as Add<typenum::B1>>::Output: Unsigned,
// {
//     fn lookup(self) -> T {
//         self.head
//     }
// }

// impl<T, Tail, Index> Lookup<T, Index> for TVecCons<T, Tail>
// where
//     Index: Unsigned + IsLess<Self::Len>,
//     Tail: TVec<T>,
//     <Tail as TVec<T>>::Len: core::ops::Add<B1>,
//     <<Tail as TVec<T>>::Len as Add<typenum::B1>>::Output: Unsigned,
// {
//     fn lookup(self) -> T {
//         // use typenum::UTerm;
//         // if <op!(Index == UTerm)>::to_bool() {
            
//         // }
//     }
// }

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecNil;

#[sealed]
impl<T> TVec<T> for TVecNil {
    type Len = typenum::U0;
    // fn lookup<Index: IsLess<Self::Len>>(self) -> T {
    //     unreachable!()
    // }
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecCons<T, Tail>{
    pub head: T,
    pub tail: Tail,
}

#[sealed]
impl<T, Tail: TVec<T>> TVec<T> for TVecCons<T, Tail>
where
    <Tail as TVec<T>>::Len: core::ops::Add<B1>,
    <<Tail as TVec<T>>::Len as Add<typenum::B1>>::Output: Unsigned,
{
    type Len = typenum::Add1<Tail::Len>;

    // fn lookup<Index: Unsigned + IsLess<Self::Len> + typenum::Cmp<typenum::UTerm>>(self) -> T {
    //     if <op!(cmp(Index, U0) == typenum::Greater)>::to_bool() {
    //         todo!()
    //     } else {
    //         todo!()
    //     }
    // }
}

pub trait TVecReplicate<T>: TVec<T> {
    fn replicate(elem: T) -> Self;
}

impl<T> TVecReplicate<T> for TVecNil {
    fn replicate(_elem: T) -> Self {
        TVecNil
    }
}

impl<T: Clone, Tail: TVecReplicate<T>> TVecReplicate<T> for TVecCons<T, Tail>
where
    <Tail as TVec<T>>::Len: core::ops::Add<B1>,
    <<Tail as TVec<T>>::Len as Add<typenum::B1>>::Output: Unsigned,
{
    fn replicate(elem: T) -> Self {
        TVecCons {
            head: elem.clone(),
            tail: Tail::replicate(elem)
        }
    }
}

#[sealed]
pub trait TVecMappable<Mapper> {
    type Output;
    fn map(self, mapper: Mapper) -> Self::Output;
}

#[sealed]
impl<F> TVecMappable<F> for TVecNil {
    type Output = TVecNil;

    fn map(self, _fun: F) -> Self::Output {
        TVecNil
    }
}

#[sealed]
impl<F, R, T, Tail> TVecMappable<F> for TVecCons<T, Tail>
where
    F: Fn(T) -> R,
    Tail: TVecMappable<F>,
{
    type Output = TVecCons<R, <Tail as TVecMappable<F>>::Output>;
    fn map(self, fun: F) -> Self::Output {
        let TVecCons {head, tail} = self;
        TVecCons {
            head: fun(head),
            tail: tail.map(fun),
        }
    }
}

#[sealed]
pub trait TVecZippable<Other> {
    type Zipped;

    fn zip(self, other: Other) -> Self::Zipped;
}

#[sealed]
impl TVecZippable<TVecNil> for TVecNil {
    type Zipped = TVecNil;
    fn zip(self, _other: TVecNil) -> Self::Zipped {
        TVecNil
    }
}

#[sealed]
impl<T, T2, Tail, Tail2> TVecZippable<TVecCons<T2, Tail2>> for TVecCons<T, Tail>
where
    Tail: TVecZippable<Tail2>
{
    type Zipped = TVecCons<(T, T2), Tail::Zipped>;
    fn zip(self, other: TVecCons<T2, Tail2>) -> Self::Zipped {
        TVecCons {
            head: (self.head, other.head),
            tail: self.tail.zip(other.tail)
        }
    }
}

#[sealed]
pub trait TVecZipWith<Other, F>: TVecZippable<Other> {
    type Output;
    fn zip_with(self, other: Other, fun: F) -> Self::Output;
}

#[sealed]
impl<F> TVecZipWith<TVecNil, F> for TVecNil {
    type Output = TVecNil;
    fn zip_with(self, _other: TVecNil, _fun: F) -> Self::Output {
        TVecNil
    }
}


#[sealed]
impl<T, T2, Tail, Tail2, F, R> TVecZipWith<TVecCons<T2, Tail2>, F> for TVecCons<T, Tail>
    where
    F: Fn(T, T2) -> R,
    Tail: TVecZipWith<Tail2, F>,
{
    type Output = TVecCons<R, <Tail as TVecZipWith<Tail2, F>>::Output>;
    fn zip_with(self, other: TVecCons<T2, Tail2>, fun: F) -> Self::Output {
        TVecCons {
            head: fun(self.head, other.head),
            tail: self.tail.zip_with(other.tail, fun),
        }
    }
}

pub trait Naperian<T> {
    type Log: Copy;
    // fn lookup(self) -> fn(Self::Log) -> Self::T;
    // fn lookup<Fun: Fn(Self::Log) -> Self::T>(self) -> Fun;
    // fn lookup(self) -> Box<dyn Fn(Self::Log) -> &Self::T>;
    fn lookup(&self, index: Self::Log) -> &T;
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self;
    // fn positions<Nap: Naperian<T = Self::Log>>() -> Nap;

    // fn pure(init_fun: impl Fn() -> T) -> Self
    // where
    //     Self: Sized
    // {
    //     Self::tabulate(|_pos| init_fun())
    // }

    // fn ap<U, R, Vals: Naperian<U, Log = Self::Log>>(&self, vals: Vals) -> impl Naperian<R>
    //     where
    //     T: Fn(&U) -> R,
    //     Self: Sized
    // {
    //     Self::tabulate(|pos| {
    //         let fun = self.lookup(pos);
    //         let val = vals.lookup(pos);
    //         fun(val)
    //     })
    // }
}

pub trait NaperianPos {
    fn positions() -> Self;
}

pub unsafe trait ArrayLength2: Unsigned {
    type ArrayType<T>;
}

unsafe impl ArrayLength2 for typenum::UTerm {
    #[doc(hidden)]
    type ArrayType<T> = [T; 0];
}

/// Internal type used to generate a struct of appropriate size
#[allow(dead_code)]
#[repr(C)]
#[doc(hidden)]
pub struct GArrayImplEven<T, U> {
    parent1: U,
    parent2: U,
    _marker: PhantomData<T>,
}

impl<T: Clone, U: Clone> Clone for GArrayImplEven<T, U> {
    fn clone(&self) -> GArrayImplEven<T, U> {
        GArrayImplEven {
            parent1: self.parent1.clone(),
            parent2: self.parent2.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Copy, U: Copy> Copy for GArrayImplEven<T, U> {}


unsafe impl<N: ArrayLength2> ArrayLength2 for typenum::UInt<N, B0> {
    #[doc(hidden)]
    type ArrayType<T> = GArrayImplEven<T, N::ArrayType<T>>;
}

unsafe impl<N: ArrayLength2> ArrayLength2 for typenum::UInt<N, B1> {
    #[doc(hidden)]
    type ArrayType<T> = GArrayImplOdd<T, N::ArrayType<T>>;
}


/// Internal type used to generate a struct of appropriate size
#[allow(dead_code)]
#[repr(C)]
#[doc(hidden)]
pub struct GArrayImplOdd<T, U> {
    parent1: U,
    parent2: U,
    data: T,
}

impl<T: Clone, U: Clone> Clone for GArrayImplOdd<T, U> {
    fn clone(&self) -> GArrayImplOdd<T, U> {
        GArrayImplOdd {
            parent1: self.parent1.clone(),
            parent2: self.parent2.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T: Copy, U: Copy> Copy for GArrayImplOdd<T, U> {}


/// Struct representing a generic array - `GenericArray<T, N>` works like [T; N]
#[allow(dead_code)]
#[repr(transparent)]
pub struct GArray<T, U: ArrayLength2> {
    data: U::ArrayType<T>,
}


unsafe impl<T: Send, N: ArrayLength2> Send for GArray<T, N> {}
unsafe impl<T: Sync, N: ArrayLength2> Sync for GArray<T, N> {}


impl<T, N> core::ops::Deref for GArray<T, N>
where
    N: ArrayLength2,
{
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self as *const Self as *const T, N::USIZE) }
    }
}

impl<T, N> core::ops::DerefMut for GArray<T, N>
where
    N: ArrayLength2,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut T, N::USIZE) }
    }
}


// pub struct Garr<T, N: Unsigned> {
//     data: [T; N::USIZE],
//     marker: PhantomData<N>,
// }


// pub fn positions<T, Nap: Naperian<T, Log = T>>() -> Nap {
//     Nap::tabulate(|x| x)
// }

pub trait HKT<U> {
    type Current; // Current content type; ex: for Vec<A> this would be A
    type Target; // Type with T swapped with U; ex: for Vec<A> this would be Vec<U>
}

macro_rules! derive_hkt {
    ($t:ident) => {
        impl<T, U> HKT<U> for $t<T> {
            type Current = T;
            type Target = $t<U>;
        }
    }
}

derive_hkt!(Vec);
derive_hkt!(Option);
derive_hkt!(Box);

pub trait Generic1 {
    type I;
    type New<Ego>;
}


impl<T, U, N> HKT<U> for GenericArray<T, N>
    where
        N: ArrayLength<T> + ArrayLength<U>
{
    type Current = T;
    type Target = GenericArray<U, N>;
}

macro_rules! derive_generic1 {
    ($t:ident) => {
        impl<T> Generic1 for $t<T> {
            type I = T;
            type New<X> = $t<X>;
        }
    }
}
derive_generic1!(Vec);
derive_generic1!(Option);
derive_generic1!(Box);

// impl<T, N> Generic1 for GenericArray<T, N>
// where
//     N: ArrayLength<T>,
// {
//     type I = T;
//     type New<X> = GenericArray<X, N>;
// }

trait Plug<B> {
    type R;
}

impl<A, B> Plug<B> for Option<A> {
    type R = Option<B>;
}

impl<A, B, N> Plug<B> for GenericArray<A, N>
    where
    N: ArrayLength<A> + ArrayLength<B>,
{
    type R = GenericArray<B, N>;
}

impl<A, B, N> Plug<B> for GArray<A, N>
where
    N: ArrayLength2,
{
    type R = GArray<B, N>;
}


trait ApplicativeX<'a, 'f, A: 'a, B, F: Fn(&'a A) -> B + 'f>: Plug<A> + Plug<B> + Plug<F> {
    fn ap<>(funs: &'f <Self as Plug<F>>::R, vals: &'a <Self as Plug<A>>::R) -> <Self as Plug<B>>::R
    where Self: Plug<A> + Plug<B> + Plug<F>;

    // fn pure(x: Self::I) -> Self
    // where Self: Generic1;
}

impl<'a, 'f, A: 'a, B, F: Fn(&'a A) -> B + 'f> ApplicativeX<'a, 'f, A, B, F> for Option<A> {
    // fn pure(x: <Self as Generic1>::I) -> Self {
    //     Some(x)
    // }

    fn ap(funs: &'f <Self as Plug<F>>::R, vals: &'a <Self as Plug<A>>::R) -> <Self as Plug<B>>::R {
        match (funs, vals) {
            (Some(f), Some(v)) => Some(f(v)),
            (_, _) => None,
        }
    }
}

// impl<A, B, F: Fn(&A) -> B, N> ApplicativeX<A, B, F> for GenericArray<A, N>
//     where
//     N: ArrayLength<A> + ArrayLength<B> + ArrayLength<F>,
//     Self: Plug<A> + Plug<B> + Plug<F>,
//     <Self as Plug<A>>::R: IntoIterator<Item = A>,
//     <Self as Plug<F>>::R: IntoIterator<Item = F>,
// {
//     fn ap(funs: &<Self as Plug<F>>::R, vals: &<Self as Plug<A>>::R) -> <Self as Plug<B>>::R
//     {
//         GenericArray::from_iter(funs.into_iter().zip(vals.into_iter()).map(|(f, v)| f(&v)))
//     }
// }


/// Creates an array one element at a time using a mutable iterator
/// you can write to with `ptr::write`.
///
/// Increment the position while iterating to mark off created elements,
/// which will be dropped if `into_inner` is not called.
#[doc(hidden)]
pub struct GArrayBuilder<T, N: ArrayLength2> {
    array: core::mem::MaybeUninit<GArray<T, N>>,
    position: usize,
}

impl<T, N: ArrayLength2> GArrayBuilder<T, N> {
    #[doc(hidden)]
    #[inline]
    pub unsafe fn new() -> GArrayBuilder<T, N> {
        GArrayBuilder {
            array: core::mem::MaybeUninit::uninit(),
            position: 0,
        }
    }

    /// Creates a mutable iterator for writing to the array using `ptr::write`.
    ///
    /// Increment the position value given as a mutable reference as you iterate
    /// to mark how many elements have been created.
    #[doc(hidden)]
    #[inline]
    pub unsafe fn iter_position(&mut self) -> (core::slice::IterMut<T>, &mut usize) {
        (
            (&mut *self.array.as_mut_ptr()).iter_mut(),
            &mut self.position,
        )
    }

    /// When done writing (assuming all elements have been written to),
    /// get the inner array.
    #[doc(hidden)]
    #[inline]
    pub unsafe fn into_inner(self) -> GArray<T, N> {
        let array = core::ptr::read(&self.array);

        core::mem::forget(self);

        array.assume_init()
    }
}

impl<T, N: ArrayLength2> Drop for GArrayBuilder<T, N> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() {
            unsafe {
                for value in &mut (&mut *self.array.as_mut_ptr())[..self.position] {
                    core::ptr::drop_in_place(value);
                }
            }
        }
    }
}



impl<T, N> FromIterator<T> for GArray<T, N>
where
    N: ArrayLength2,
{
    fn from_iter<I>(iter: I) -> GArray<T, N>
    where
        I: IntoIterator<Item = T>,
    {
        unsafe {
            let mut destination = GArrayBuilder::new();

            {
                let (destination_iter, position) = destination.iter_position();

                iter.into_iter()
                    .zip(destination_iter)
                    .for_each(|(src, dst)| {
                        core::ptr::write(dst, src);

                        *position += 1;
                    });
            }

            if destination.position < N::USIZE {
                from_iter_length_fail(destination.position, N::USIZE);
            }

            destination.into_inner()
        }
    }
}

#[inline(never)]
#[cold]
fn from_iter_length_fail(length: usize, expected: usize) -> ! {
    panic!(
        "GenericArray::from_iter received {} elements but expected {}",
        length, expected
    );
}

impl<'a, 'f, A: 'a, B, F: Fn(&'a A) -> B + 'f, N> ApplicativeX<'a, 'f, A, B, F> for GenericArray<A, N>
where
    N: ArrayLength<A> + ArrayLength<B> + ArrayLength<F>,
    Self: Plug<A> + Plug<B> + Plug<F>,
    &'a<Self as Plug<A>>::R: IntoIterator<Item = &'a A>,
    &'f<Self as Plug<F>>::R: IntoIterator<Item = &'f F> + 'f,
    <Self as Plug<B>>::R: FromIterator<B>,
{
    fn ap(funs: &'f <Self as Plug<F>>::R, vals: &'a <Self as Plug<A>>::R) -> <Self as Plug<B>>::R
    {
        FromIterator::from_iter(funs.into_iter().zip(vals.into_iter()).map(|(f, v)| f(&v)))
    }
}

impl<'a, 'f, A: 'a, B, F: Fn(&A) -> B + 'f, N> ApplicativeX<'a, 'f, A, B, F> for GArray<A, N>
    where
    N: ArrayLength2,
    Self: Plug<A> + Plug<B> + Plug<F>,
    &'a <Self as Plug<A>>::R: IntoIterator<Item = &'a A>,
    &'f <Self as Plug<F>>::R: IntoIterator<Item = &'f F> + 'f,
    <GArray<A, N> as Plug<B>>::R: FromIterator<B>,
{
    fn ap(funs: &'f<Self as Plug<F>>::R, vals: &'a<Self as Plug<A>>::R) -> <Self as Plug<B>>::R
    {
        FromIterator::from_iter(funs.into_iter().zip(vals.into_iter()).map(|(f, v)| f(&v)))
    }
}


trait Functor<U>: HKT<U> {
    fn map<F>(&self, f: F) -> Self::Target
        where
        F: Fn(&Self::Current) -> U;
}

impl<T, U> Functor<U> for Option<T> {
    fn map<F>(&self, f: F) -> Self::Target
    where
        F: Fn(&Self::Current) -> U {
        match self {
            None => None,
            Some(value) => Some(f(value))
        }
    }
}

impl<T, U, N> Functor<U> for GenericArray<T, N>
    where
    N: ArrayLength<T> + ArrayLength<U>
{
    fn map<F>(&self, f: F) -> Self::Target
    where
        F: Fn(&Self::Current) -> U {
        GenericArray::from_iter(self.into_iter().map(f))
    }
}

trait Pure<U>: Functor<U> {
    fn pure_(value: U) -> Self::Target where Self: HKT<U, Current=U>;
}

trait Apply<B>: Functor<B> {
    fn ap<F>(&self, funs: <Self as HKT<F>>::Target) -> <Self as HKT<B>>::Target
    where
        Self: HKT<F>,
        F: Fn(&<Self as HKT<B>>::Current) -> B,
    ;
}

trait Applicative<U>: Functor<U> {
    fn ap<F>(&self, funs: <Self as HKT<F>>::Target) -> <Self as HKT<U>>::Target
    where
        F: Fn(&<Self as HKT<U>>::Current) -> U,
        Self: HKT<F>
        ;
}

impl<T, U> Pure<U> for Option<T> {
    fn pure_(value: U) -> Self::Target {
        Some(value)
    }
}

// impl<T, B> Apply<B> for Option<T> {
//     fn ap<F>(&self, funs: <Self as HKT<F>>::Target) -> <Self as HKT<B>>::Target
//     where
//         Self: HKT<F>,
//         <Self as HKT<F>>::Target: Option<F>,
//         F: Fn(&<Self as HKT<B>>::Current) -> B, {
//         match (self, funs) {
//             (Some(val), Some(fun)) => Some(fun(val)),
//             (_, _) => None,
//         }
//     }
// }

// impl<T, U> Applicative<U> for Option<T> {
//     fn ap<F>(&self, funs: <Self as HKT<F>>::Target) -> Option<U>
//     where
//         Self: HKT<F>,
//         F: Fn(&<Self as HKT<U>>::Current) -> U {
//         match self {
//             None => None,
//             Some(val) => {
//                 match funs {
//                     None => None,
//                     Some(fun) => {
//                         Some(fun(val))
//                     }
//                 }
//             }
//         }
//     }
// }


#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn example1() {
        let v123 = TVecNil.prepend(3).prepend(2).prepend(1);
        println!("{:?}", std::any::type_name_of_val(&v123));
        println!("{:?}", v123);
        println!("{:?}", v123.len());
    }

    #[test]
    fn example2() {
        let v123 = TVecNil.prepend(3).prepend(2).prepend(1);

        let v456 = v123.map(|x| x + 3);
        println!("{:?}", std::any::type_name_of_val(&v456));
        println!("{:?}", v456);
        println!("{:?}", v456.len());
    }

    #[test]
    fn lift_square() {
        let v123 = TVecNil.prepend(3).prepend(2).prepend(1);
        let result = v123.zip(TVecReplicate::replicate(3));

        println!("{:?}", std::any::type_name_of_val(&result));
        println!("{:?}", result);
        println!("{:?}", result.len());

        let result2 = result.map(|(x, y)| x + y);

        println!("{:?}", std::any::type_name_of_val(&result2));
        println!("{:?}", result2);
        println!("{:?}", result2.len());

        let result3 = v123.zip_with(TVecReplicate::replicate(3), std::ops::Add::add);
        println!("{:?}", std::any::type_name_of_val(&result3));
        println!("{:?}", result3);
        println!("{:?}", result3.len());
    }
}

// /// Peano arithmetic
// #[sealed]
// pub trait Nat {
//     fn reify() -> usize;
// }

// pub struct Z;
// pub struct S<N: Nat>(PhantomData<N>);

// #[sealed]
// impl Nat for Z {
//     fn reify() -> usize { 0 }
// }

// #[sealed]
// impl<N: Nat> Nat for S<N>{
//     fn reify() -> usize { 1 + N::reify() }
// }

// pub trait TVec<N: Nat> {
//     fn cons<T>(self, elem: T) -> TVecCons<T, S<N>, N, Self>
//         where Self: TVec<S<N>> + Sized
//     {
//         TVecCons(elem, self, PhantomData, PhantomData)
//     }

//     fn len() -> usize {
//         N::reify()
//     }
// }

// pub struct TVecNil;
// pub struct TVecCons<H, N: Nat, TailN: Nat, Tail: TVec<TailN>>(H, Tail, PhantomData<N>, PhantomData<TailN>);

// impl TVec<Z> for TVecNil {}
// impl<H, N: Nat, Tail: TVec<N>> TVec<S<N>> for TVecCons<H, S<N>, N, Tail> {}


// #[sealed]
// pub trait TVec<Size: Nat, T> {
//     fn len() -> usize;
// }

// pub struct TVecNil;

// pub struct TVecCons<Size: Nat, T, Tail: TVec<Size, T>>(PhantomData<Size>, T, Tail);

// #[sealed]
// impl<T> TVec<Z, T> for TVecNil {
//     fn len() -> usize { Z::reify() }
// }

// #[sealed]
// impl<Size: Nat, T, Tail: TVec<Size, T>> TVec<S<Size>, T> for TVecCons<Size, T, Tail> {
//     fn len() -> usize { Size::reify() }
// }








// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }

use core::marker::PhantomData;
use generic_array::{GenericArray, ArrayLength};

pub trait IsTrue {}
impl IsTrue for B1 {}


#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Fin<N: Unsigned> {
    val: usize,
    _phantom: PhantomData<N>,
}

impl<N: Unsigned> core::fmt::Debug for Fin<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = format!("Fin<U{}>", N::USIZE);
        f.debug_tuple(&name)
            .field(&self.val)
            .finish()
    }
}


impl<N: Unsigned> Fin<N> {
    /// Creates a new Fin from a usize. Fallible, checked at runtime.
    ///
    /// A BoundError is returned when the usize is larger than the maximum bound (`N`).
    #[inline(always)]
    pub const fn new(val: usize) -> Result<Self, BoundError<N>> {
        if val < N::USIZE {
            Ok(Fin{val, _phantom: PhantomData })
        } else {
            Err(BoundError::<N>::new())
        }
    }

    /// Creates a new Fin from a usize.
    ///
    /// # Safety
    /// The caller is responsible for making sure that `val` is smaller than `N::USIZE`.
    pub unsafe fn new_unchecked(val: usize) -> Self {
        Fin{val, _phantom: PhantomData}
    }

    /// Creates a new Fin from an unsigned typenum.
    ///
    /// Outcome is always valid as too large values result in a compile error.
    #[inline(always)]
    pub const fn tnew<Val>() -> Self
        where
        Val: Unsigned + IsLess<N>,
        typenum::Le<Val, N>: IsTrue,
    {
        Fin{val: Val::USIZE, _phantom: PhantomData}
    }

    /// Creates a new Fin from an compile-time constant usize.
    ///
    /// Outcome is always valid as too large values result in a compile error.
    #[inline(always)]
    pub const fn cnew<const VAL: usize>() -> Self
        where
        Const<VAL>: ToUInt,
        U<VAL>: Unsigned + IsLess<N>,
        typenum::Le<U<VAL>, N>: IsTrue,
    {
        Fin{val: VAL, _phantom: PhantomData}
    }

    #[inline(always)]
    pub const fn bound() -> usize {
        N::USIZE
    }
}

#[derive(Debug)]
pub struct BoundError<N: Unsigned> {
    _phantom: PhantomData<N>,
}

impl<N: Unsigned> BoundError<N> {
    pub const fn new() -> Self {
        Self {_phantom: PhantomData}
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
        Self: Unsigned + IsLess<N>,
        typenum::Le<Self, N>: IsTrue,
    {
        Fin::<N>::tnew::<Self>()
    }
}

impl<Index: Unsigned> UnsignedExt for Index {}

pub fn fin<const VAL: usize, N: Unsigned>() -> Fin<N>
where
    Const<VAL>: ToUInt,
U<VAL>: Unsigned + IsLess<N>,
    typenum::Le<U<VAL>, N>: IsTrue,
{
    Fin::<N>::cnew::<VAL>()
}

/// Extension trait for GenericArray
/// to look up one of its elements
/// using a typenum index constant.
/// This will reesult in a compile error for out-of-bounds access.
pub trait Lookup<T, N: Unsigned> {
    fn vlookup(&self, index: Fin<N>) -> &T;
}

impl<T, N: ArrayLength<T>> Lookup<T, N> for GenericArray<T, N>
{
    fn vlookup(&self, index: Fin<N>) -> &T {
        // TODO: Make sure a bounds check is *never* inserted here
        // as it is never needed
        &self[index.val]
    }
}

pub trait Iota<N> {
    fn iota() -> Self;
}

impl<N: ArrayLength<Fin<N>>> Iota<N> for GenericArray<Fin<N>, N> {
    fn iota() -> Self {
        GenericArray::from_iter((0..N::USIZE).map(|pos| unsafe { Fin::new_unchecked(pos)}))
        // let mut garr: GenericArray<Fin<N>, N> = Default::default();
        // for pos in 0..N::USIZE {
        //     // Safety: index is always is smaller than GenericArray's bounds
        //     garr[pos] = unsafe { Fin::new_unchecked(pos) };
        // }
        // garr
    }
}

impl<T, N: ArrayLength<T> + ArrayLength<Fin<N>>> Naperian<T> for GenericArray<T, N> {
    type Log = Fin<N>;
    // fn lookup(self) -> Box<dyn Fn(Self::Log) -> Self::T> {
    //     Box::new(|index: Fin<N>| { self.lookup(index) })
    // }
    fn lookup(&self, index: Fin<N>) -> &T {
        &self[index.val]
    }
    // fn positions<Nap: Naperian<T = Self::Log>>() -> Nap {
    //     <Self as Iota<N>>::iota()
    // }
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self {
        GenericArray::from_iter((0..N::USIZE).map(|pos| {
            let pos = unsafe { Fin::new_unchecked(pos)};
            fun(pos)
        }))

        // let positions: GenericArray<Fin<N>, N> = Iota::iota();
        // positions.map(|x| fun(x))
    }

    // fn positions<Nap: Naperian<T = Self::Log>>() -> Nap {
        
    // }
}


impl<N: ArrayLength<Fin<N>>> NaperianPos for GenericArray<Fin<N>, N> {
    fn positions() -> Self {
        Self::iota()
    }
}



#[cfg(test)]
mod tests2 {
    use super::*;
    use generic_array::arr;
    use typenum::U;
    #[test]
    fn compile_time_lookup_bounds() {
        let arr = arr![usize; 1, 2, 3];
        let val1 = arr.lookup(0.try_into().unwrap());
        println!("{:?}", val1);
        let val2 = arr.lookup(U::<0>::fin());
        println!("{:?}", val2);
        let idx = Fin::cnew::<0>();// fin::<2, _>();
        let val3 = arr.lookup(idx);
        println!("{:?}", val3);
        assert_eq!(val1, val2);
        assert_eq!(val1, val3);
    }

    #[test]
    fn iota() {
        let foo = GenericArray::<_, U3>::iota();
        println!("{:?}", foo);
        println!("{:?}", std::any::type_name_of_val(&foo[0]));
    }

    fn plus1(x: &usize) -> usize {
        x + 1
    }

    #[test]
    fn functor() {
        let arr = arr![usize; 1,2,3];
        let res = arr.map(|x| x * 2);
        assert_eq!(res, arr![usize; 2,4,6]);
    }

    // #[test]
    // fn applicative() {
    //     let arr = arr![usize; 1,2,3];
    //     let arr2 = arr![usize; 4,5,6];
    //     let funs = arr![fn(&usize) -> usize; plus1, plus1, plus1];
    //     let result = funs.ap(arr);
    //     // let result: GenericArray<usize, _> = GenericArray::pure(|| { |x: usize| {|y: usize| x + y}}).ap(arr).ap(arr2);
        
}
