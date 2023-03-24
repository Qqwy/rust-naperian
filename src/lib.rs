#![feature(type_name_of_val)]
use sealed::sealed;
use std::marker::PhantomData;

#[sealed]
pub trait TVec<T>: Sized {
    const LEN: usize;
    fn len(&self) -> usize {
        Self::LEN
    }

    fn prepend(self, h: T) -> TVecCons<T, Self> {
        TVecCons {
            head: h,
            tail: self
        }
    }
}


#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecNil;

#[sealed]
impl<T> TVec<T> for TVecNil {
    const LEN: usize = 0;
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecCons<T, Tail>{
    pub head: T,
    pub tail: Tail,
}

#[sealed]
impl<T, Tail: TVec<T>> TVec<T> for TVecCons<T, Tail> {
    const LEN : usize = 1 + Tail::LEN;
}

pub trait TVecReplicate<T>: TVec<T> {
    fn replicate(elem: T) -> Self;
}

impl<T> TVecReplicate<T> for TVecNil {
    fn replicate(_elem: T) -> Self {
        TVecNil
    }
}

impl<T: Clone, Tail: TVecReplicate<T>> TVecReplicate<T> for TVecCons<T, Tail> {
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
