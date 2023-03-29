use super::{Container, Mappable, Mappable2, Mappable3, Apply, New, NewFrom};


// TODO implement IntoIterator for Pair
/// The simple example type from the Naperian paper.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair<T>(pub T, pub T);

unsafe impl<T> Container for Pair<T> {
    type Elem = T;
    type Containing<X> = Pair<X>;
}

impl<T, U> Mappable<U> for Pair<T> {
    fn map(&self, mut fun: impl FnMut(&Self::Elem) -> U) -> Self::Containing<U> {
        Pair(fun(&self.0), fun(&self.1))
    }

    fn map_by_value(self, mut fun: impl FnMut(Self::Elem) -> U) -> Self::Containing<U> {
        Pair(fun(self.0), fun(self.1))
    }
}

impl<T: Clone> New<T> for Pair<T> {
    fn new(elem_val: T) -> Self {
        Pair(elem_val.clone(), elem_val)
    }
}

impl<T> NewFrom<T> for Pair<T> {
    fn new_from(mut fun: impl FnMut() -> T) -> Self {
        Pair(fun(), fun())
    }
}

impl<A, B, F: Fn(&A) -> B> Apply<A, B, F> for Pair<F> {
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B> {
        let e0 = self.0(&vals.0);
        let e1 = self.1(&vals.1);
        Pair(e0, e1)
    }
}

impl<A, U> Mappable2<A, U> for Pair<A> {
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        mut fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U> {
        Pair(fun(&self.0, &rhs.0), fun(&self.1, &rhs.1))
    }

    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        mut fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U> {
        Pair(fun(self.0, rhs.0), fun(self.1, rhs.1))
    }
}

impl<A, U> Mappable3<A, U> for Pair<A> {
    fn map3<B, C>(
        &self,
        second: &Self::Containing<B>,
        third: &Self::Containing<C>,
        mut fun: impl FnMut(&A, &B, &C) -> U,
    ) -> Self::Containing<U> {
        Pair(
            fun(&self.0, &second.0, &third.0),
            fun(&self.1, &second.1, &third.1),
        )
    }
    fn map3_by_value<B, C>(
        self,
        second: Self::Containing<B>,
        third: Self::Containing<C>,
        mut fun: impl FnMut(A, B, C) -> U,
    ) -> Self::Containing<U> {
        Pair(
            fun(self.0, second.0, third.0),
            fun(self.1, second.1, third.1),
        )
    }
}
