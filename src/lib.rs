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
    type Log;
    fn lookup(self, log: Self::Log) -> T;
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self;
}

pub fn positions<T, Nap: Naperian<T, Log = T>>() -> Nap {
    Nap::tabulate(|x| x)
}


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


#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Extension trait for GenericArray
/// to look up one of its elements
/// using a typenum index constant.
/// This will reesult in a compile error for out-of-bounds access.
pub trait Lookup<T, N: Unsigned> {
    fn lookup(&self, index: Fin<N>) -> &T;
}

impl<T, N: ArrayLength<T>> Lookup<T, N> for GenericArray<T, N>
{
    fn lookup(&self, index: Fin<N>) -> &T {
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

pub fn fin<const VAL: usize, N: Unsigned>() -> Fin<N>
where
    Const<VAL>: ToUInt,
    U<VAL>: Unsigned + IsLess<N>,
    typenum::Le<U<VAL>, N>: IsTrue,
{
    Fin::<N>::cnew::<VAL>()
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
}
