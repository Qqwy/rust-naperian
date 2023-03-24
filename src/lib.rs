use sealed::sealed;
use std::marker::PhantomData;

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

    // fn vmap<U, Us: TVec<U>>(self, fun: impl Fn(T) -> U) -> Us;
}


#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecNil;

impl<T> TVec<T> for TVecNil {
    const LEN: usize = 0;

}


#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct TVecCons<T, Tail>{
    pub head: T,
    pub tail: Tail,
}

impl<T, Tail: TVec<T>> TVec<T> for TVecCons<T, Tail> {
    const LEN : usize = 1 + Tail::LEN;
}

pub trait TVecMappable<Mapper> {
    type Output;
    fn map(self, mapper: Mapper) -> Self::Output;
}

impl<F> TVecMappable<F> for TVecNil {
    type Output = TVecNil;

    fn map(self, _fun: F) -> Self::Output {
        TVecNil
    }
}

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








pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
