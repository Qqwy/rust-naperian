use generic_array::sequence::GenericSequence;
use typenum::Unsigned;
use typenum::consts::*;
use generic_array::{GenericArray, ArrayLength};

/// Trait which makes higher-kindred types tick
///
/// Rust has no concept of 'higher kinds' but only of concrete types.
/// This means that if you want to talk about container-like types
/// whose element types might change in the process of performing a trait method,
/// there is no simple built-in way to do so.
///
/// Instead, it can be modeled using generic associated types (GATs).
/// There are however two drawbacks to this approach:
/// - It is possible to accidentally implement the trait incorrectly for your type. As such, this is an _unsafe_ trait; implementers are responsible for making sure their implementation is sensible.
/// - The compiler does not know that a `Vec<T>::Containing<X>` == `Vec<X>`. As such, you'll often have to re-state 'obvious' trait bounds.
pub unsafe trait Container {
    /// The element type of the container
    ///
    /// # Examples:
    /// For a Vec<T>, this is T.
    /// For an Option<A>, this is A.
    type Elem;

    /// The container type with its element type
    /// changed to X.
    ///
    /// # Examples:
    /// For a Vec<T>, this is Vec<X>
    /// For an Option<A>, this is Option<X>
    type Containing<X>;
}

unsafe impl<T> Container for Vec<T> {
    type Elem = T;
    type Containing<X> = Vec<X>;
}

unsafe impl<T> Container for Option<T> {
    type Elem = T;
    type Containing<X> = Option<X>;
}

unsafe impl<T> Container for Box<T> {
    type Elem = T;
    type Containing<X> = Box<X>;
}

/// The simple example type from the Naperian paper.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair<T>(T, T);

unsafe impl<T> Container for Pair<T> {
    type Elem = T;
    type Containing<X> = Pair<X>;
}

unsafe impl<T, N> Container for GenericArray<T, N>
    where
    N: ArrayLength,
{
    type Elem = T;
    type Containing<X> = GenericArray<X, N>;
}

/// Transform a container by running a unary function element-wise on its contents.
///
/// Also known as 'Functor'.
pub trait Mappable<U>: Container {
    fn map(&self, fun: impl Fn(&Self::Elem) -> U) -> Self::Containing<U>;
}

impl<T, U> Mappable<U> for Option<T> {
    fn map(&self, fun: impl Fn(&Self::Elem) -> U) -> Self::Containing<U> {
        match self {
            None => None,
            Some(val) => Some(fun(val))
        }
    }
}

impl<T, U> Mappable<U> for Pair<T> {
    fn map(&self, fun: impl Fn(&Self::Elem) -> U) -> Self::Containing<U> {
        Pair(fun(&self.0), fun(&self.1))
    }
}

impl<T, U, N> Mappable<U> for GenericArray<T, N>
where
    N: ArrayLength,
{
    fn map(&self, fun: impl Fn(&Self::Elem) -> U) -> Self::Containing<U> {
        GenericArray::generate(|pos| {
            let val = &self[pos];
            fun(val)
        })
    }
}

/// Create a new container from an initial value.
///
/// Also known as 'Pointed'
pub trait New<T>: Container {
    fn new(elem_val: T) -> Self;
}

/// Variant of the [`New`] trait which is implementable for more types
/// with relaxed bounds.
///
/// (Whereas for `[New]` you often would have to require [`Clone`]
/// for the types, this is not true for [`NewFrom`]).
pub trait NewFrom<T>: Container {
    fn new_from(fun: impl FnMut() -> T) -> Self;
}


impl<T> New<T> for Option<T> {
    fn new(elem_val: T) -> Self {
        Some(elem_val)
    }
}


impl<T> NewFrom<T> for Option<T> {
    fn new_from(mut fun: impl FnMut() -> T) -> Self {
        Some(fun())
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

impl<T: Clone, N: ArrayLength> New<T> for GenericArray<T, N>
    {
    fn new(elem_val: T) -> Self {
        GenericArray::generate(|_pos| {
            elem_val.clone()
        })
    }
}

impl<T, N: ArrayLength> NewFrom<T> for GenericArray<T, N>
{
    fn new_from(mut fun: impl FnMut() -> T) -> Self {
        GenericArray::generate(|_pos| {
            fun()
        })
    }
}


/// Combines a container containing functions and a container containing values
/// by elementwise running each function on the value at the same element position.
///
/// Should be implemented on `YourType<F>`, to allow chaining of `ap`.
/// (and so the compiler can automatically infer A and B from knowing F.)
pub trait Apply<A, B, F: Fn(&A) -> B>: Container
{
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B>;
}

impl<A, B, F: Fn(&A) -> B> Apply<A, B, F> for Option<F> {
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B> {
        match (self, vals) {
            (Some(f), Some(val)) => Some(f(val)),
            (_, _) => None,
        }
    }
}


impl<A, B, F: Fn(&A) -> B> Apply<A, B, F> for Pair<F> {
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B> {
        let e0 = self.0(&vals.0);
        let e1 = self.1(&vals.1);
        Pair(e0, e1)
    }
}


impl<A, B, F: Fn(&A) -> B, N: ArrayLength> Apply<A, B, F> for GenericArray<F, N> {
    fn ap(&self, vals: &Self::Containing<A>) -> Self::Containing<B> {
        GenericArray::generate(|pos| {
            let fun = &self[pos];
            let val = &vals[pos];
            fun(val)
        })
    }
}

/// Map a binary (two-parameter) function over two containers.
///
/// Implementing this is usually very straightforward.
/// Here is the implementation for Option:
/// ```rust
/// impl<A, U> Mappable2<A, U> for Option<A> {
///    fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
///        match (self, rhs) {
///            (Some(left), Some(right)) => Some(fun(left, right)),
///            (_, _) => None,
///        }
///    }
/// }
/// ```
///
/// # Rationale for type-happy folk
///
/// Indeed, [`Mappable2::map2`] is the same as the `liftA2` operation you might know from Haskell's Applicative typeclass.
///
/// In other words,
/// Mappable2 can always be implemented for containers implementing [`Apply`] and [`New`] (or [`NewFrom`]),
/// by currying `fun` and leveraging [`New::new`] ([`NewFrom::new_from`]) and [`Apply::ap`]:
///
/// ```rust
/// impl<A: Clone, U> Mappable2<A, U> for Option<A> {
///     fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
///         let curried_fun = |lhs: &A| {
///             let lhs = lhs.clone();
///             move |rhs: &B| fun(&lhs, rhs)
///         };
///         Option::new(curried_fun).ap(&self).ap(&rhs)
///     }
/// }
/// ```
///
/// But note that we need to clone `lhs` inside `curried_fun` to make the borrow checker happy.
/// Alternatively, we could use some unsafe code because we know that we are immediately running the full closure:
/// ```rust
/// impl<A, U> Mappable2<A, U> for Option<A> {
///     fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
///         let curried_fun = |lhs: &A| {
///             // SAFETY: We immediately run the full closure before returning from `map2` so the reference is in scope.
///             let lhs: &A = unsafe { core::mem::transmute(lhs)};
///             move |rhs: &B| fun(&lhs, rhs)
///         };
///         Option::new(curried_fun).ap(&self).ap(&rhs)
///     }
/// }
/// ```
///
/// Neither alternative is very appealing.
/// Also, the compiler must work very hard to see through the partial application trickery we are doing here
/// to be able to optimize the stuff away again.
///
/// As such, it is usually better idea to implement `map2` directly for your type (as seen in the example at the top).
pub trait Mappable2<A, U>: Container {
    fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U>;
}

impl<A, U> Mappable2<A, U> for Option<A> {
    fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
        match (self, rhs) {
            (Some(left), Some(right)) => Some(fun(left, right)),
            (_, _) => None,
        }
    }
}

impl<A, U> Mappable2<A, U> for Pair<A> {
    fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
        Pair(fun(&self.0, &rhs.0), fun(&self.1, &rhs.1))
    }
}

impl<A, U, N: ArrayLength> Mappable2<A, U> for GenericArray<A, N> {
    fn map2<B>(&self, rhs: &Self::Containing<B>, fun: &impl Fn(&A, &B) -> U) -> Self::Containing<U> {
        GenericArray::generate(|pos| {
            let left = &self[pos];
            let right = &rhs[pos];
            fun(left, right)
        })
    }
}

/// Map a ternary (three-parameter) function over a container.
///
/// This trait is very similar to [`Mappable`] and (especially) [`Mappable2`].
/// In particular, the same implementation trade-offs apply w.r.t. [`Mappable2`].
pub trait Mappable3<A, U>: Container {
    fn map3<B, C>(&self, second: &Self::Containing<B>, third: &Self::Containing<C>, fun: &impl Fn(&A, &B, &C) -> U) -> Self::Containing<U>;
}

impl<A, U> Mappable3<A, U> for Option<A> {
    fn map3<B, C>(&self, second: &Self::Containing<B>, third: &Self::Containing<C>, fun: &impl Fn(&A, &B, &C) -> U) -> Self::Containing<U> {
        match (self, second, third) {
            (Some(one), Some(two), Some(three)) => Some(fun(one, two, three)),
            (_, _, _) => None,
        }
    }
}

impl<A, U> Mappable3<A, U> for Pair<A> {
    fn map3<B, C>(&self, second: &Self::Containing<B>, third: &Self::Containing<C>, fun: &impl Fn(&A, &B, &C) -> U) -> Self::Containing<U> {
        Pair(fun(&self.0, &second.0, &third.0), fun(&self.1, &second.1, &third.1))

    }
}

impl<A, U, N: ArrayLength> Mappable3<A, U> for GenericArray<A, N> {
    fn map3<B, C>(&self, second: &Self::Containing<B>, third: &Self::Containing<C>, fun: &impl Fn(&A, &B, &C) -> U) -> Self::Containing<U> {
        GenericArray::generate(|pos| {
            let one = &self[pos];
            let two = &second[pos];
            let three = &third[pos];
            fun(one, two, three)
        })
    }
}

pub trait Naperian<T>: Mappable<T> {
    type Log: Copy;
    fn lookup(&self, index: Self::Log) -> &T;
    /// Follows the paper more closely.
    /// Unfortunately,
    /// currently requires boxing the returned function.
    ///
    /// In the future, this might be improved
    /// (Once the feature `#![feature(return_position_impl_trait_in_trait)]` is stabilized,
    /// we could write this as `fn lookup_<'a>(&'a self) -> impl Fn(Self::Log) -> &'a T;`.)
    ///
    /// The default implementation is probably suitable for all situations.
    fn lookup_<'a>(&'a self) -> Box<dyn Fn(Self::Log) -> &'a T + 'a> {
        Box::new(|index| {
            self.lookup(index)
        })
    }
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self;
    fn positions() -> Self::Containing<Self::Log>;
}


pub trait IsTrue {}
impl IsTrue for B1 {}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Fin<N: Unsigned> {
    val: usize,
    _phantom: core::marker::PhantomData<N>,
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
            Ok(Fin{val, _phantom: core::marker::PhantomData })
        } else {
            Err(BoundError::<N>::new())
        }
    }

    /// Creates a new Fin from a usize.
    ///
    /// # Safety
    /// The caller is responsible for making sure that `val` is smaller than `N::USIZE`.
    pub unsafe fn new_unchecked(val: usize) -> Self {
        Fin{val, _phantom: core::marker::PhantomData}
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
        Fin{val: Val::USIZE, _phantom: core::marker::PhantomData}
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
        Fin{val: VAL, _phantom: core::marker::PhantomData}
    }

    #[inline(always)]
    pub const fn bound() -> usize {
        N::USIZE
    }
}

#[derive(Debug)]
pub struct BoundError<N: Unsigned> {
    _phantom: core::marker::PhantomData<N>,
}

impl<N: Unsigned> BoundError<N> {
    pub const fn new() -> Self {
        Self {_phantom: core::marker::PhantomData}
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



impl<T> Naperian<T> for Pair<T> {
    type Log = bool;
    fn lookup(&self, index: Self::Log) -> &T {
        match index {
            false => &self.0,
            true => &self.1,
        }
    }
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self {
        Pair(fun(false), fun(true))
    }

    fn positions() -> Self::Containing<Self::Log> {
        Pair(false, true)
    }
}

impl<T, N: ArrayLength> Naperian<T> for GenericArray<T, N> {
    type Log = Fin<N>;
    fn lookup(&self, index: Self::Log) -> &T {
        &self[index.val]
    }
    fn positions() -> Self::Containing<Self::Log> {
        GenericArray::generate(|pos| {
            // SAFETY: pos is in range [0..N)
            unsafe { Fin::new_unchecked(pos) }
        })
    }
    fn tabulate(fun: impl Fn(Self::Log) -> T) -> Self {
        GenericArray::generate(|pos| {
            // SAFETY: pos is in range [0..N)
            let fin = unsafe { Fin::new_unchecked(pos) };
            fun(fin)
        })
    }
}

/// Transpose a F<G<A>> into G<F<A>> provided both F and G implement [`Naperian`].
/// (and A: [`Clone`] since we need to copy a bunch of A's around.)
///
/// There is no need to implement this trait manually since there is a blanket implementation
/// for all types implementing Naperian.
pub trait NaperianTranspose<G, A: Clone>
    where
    Self: Naperian<G>,
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{
    fn transpose(&self) -> G::Containing<Self::Containing<A>> {
        Naperian::tabulate(|x| {
            Naperian::tabulate(|y| {
                self.lookup(y).lookup(x).clone()
            })
        })
    }
}

impl<G, A: Clone, Nap: ?Sized + Naperian<G>> NaperianTranspose<G, A> for Nap
where
    G: Naperian<A>,
    Self::Containing<A>: Naperian<A, Log = Self::Log>,
    G::Containing<Self::Containing<A>>: Naperian<Self::Containing<A>, Log = G::Log>,
{}


pub trait Traversable<G, A, B>
    where
    Self: Mappable<A>,
    G: Mappable<Self::Containing<B>> + Container<Elem=B>,
{
    fn traverse(&self, fun: impl Fn(&A) -> G) -> G::Containing<Self::Containing<B>>;
}

impl<G, A, B: Clone> Traversable<G, A, B> for Option<A>
where
    Self: Mappable<A>,
    G: Mappable<Self::Containing<B>> + Container<Elem=B>,
    Self::Containing<B>: New<B>,
    <G as Container>::Containing<<Option<A> as Container>::Containing<B>>: New<Option<B>>,
{
    fn traverse(&self, fun: impl Fn(&A) -> G) -> <G>::Containing<Self::Containing<B>> {
        match self {
            None => {
                New::new(None)
            },
            Some(val) => {
                fun(val).map(|x: &B| {
                    New::new(x.clone())
                })
            }
        }
    }
}

trait ConcretePair<T> {
    fn reify(pair: Pair<T>) -> Self;
}

impl<T> ConcretePair<T> for Pair<T> {
    fn reify(pair: Pair<T>) -> Self {
        pair
    }
}

impl<G, A, B: Clone> Traversable<G, A, B> for Pair<A>
where
    Self: Mappable<A>,
    G: Mappable<Self::Containing<B>> + Mappable2<B, Self::Containing<B>>+ Container<Elem=B, Containing<B> = G>,
    Self::Containing<B>: ConcretePair<B>,
{
    fn traverse(&self, fun: impl Fn(&A) -> G) -> G::Containing<Self::Containing<B>> {
        fun(&self.0).map2(&fun(&self.1), &|one: &B, two: &B| {
            let pair = Pair(one.clone(), two.clone());
            ConcretePair::reify(pair)
        })
    }
}


// /// Example from the paper
// /// Also thanks to the 'monadic' crate for the translation of the State datatype to Rust
// pub struct State<'a, S, A> { 
//     pub run_state: Box<dyn 'a + Fn(S) -> (A, S)>,
// }

// impl<'a, S, A> State<'a, S, A> {
//     /// Given an initial_state,
//     /// runs the contained state-transition-function to completion.
//     /// returning the final outcome `A` and the end state `S`.
//     fn run(&self, initial_state: S) -> (A, S) {
//         (*self.run_state)(initial_state)
//     }

//     /// Variant of [`State::run`] that discards the resulting state,
//     /// only returning the final outcome `A`.
//     fn exec(&self, initial_state: S) -> A {
//         self.run(initial_state).0
//     }
// }

// unsafe impl<'a, S, A> Container for State<'a, S, A> {
//     type Elem = A;
//     type Containing<X> = State<'a, S, X>;
// }

// impl<'a, S: 'a, A: Clone + 'a> New<A> for State<'a, S, A> {
//     fn new(elem_val: A) -> Self {
//         State { run_state: Box::new( move |state: S| (elem_val.clone(), state))}
//     }
// }

// /// Behaves like Mappable but types that need restricted lifetime bounds cannot implement the normal Mappable
// pub trait MappableWithLifetime<'a, U>: Container {
//     fn map(&'a self, fun: impl Fn(&Self::Elem) -> U + 'a) -> Self::Containing<U>;
// }

// impl<'a, U, S, A> MappableWithLifetime<'a, U> for State<'a, S, A> {
//     fn map(&'a self, fun: impl Fn(&Self::Elem) -> U + 'a) -> Self::Containing<U> {
//         let boxed_fun = Box::new(fun);
//         State { run_state: Box::new( move |state: S| {
//             let (a, s) = self.run(state);
//             let b = boxed_fun(&a);
//             (b, s)
//         })}
//     }
// }


// /// Behaves like Apply but types that need restricted lifetime bounds cannot implement the normal Apply
// pub trait ApplyWithLifetime<'a, A, B, F: Fn(&A) -> B>: Container
// {
//     fn ap(&'a self, vals: &'a Self::Containing<A>) -> Self::Containing<B>;
// }

// impl<'a, A: 'a, B: 'a, F: 'a, S> ApplyWithLifetime<'a, A, B, F> for State<'a, S, F>
// where
//     F: Fn(&A) -> B,
// {
//     fn ap(&'a self, vals: &'a Self::Containing<A>) -> Self::Containing<B> {
//         State { run_state: Box::new(|s0| {
//             let (val, s1) = vals.run(s0);
//             let (fun, s2) = self.run(s1);
//             (fun(&val), s2)
//         })}
//     }
// }

/// Anything that is a Dimension can be used one one rank of a hypercuboid.
///
///
/// Conceptually Dimension is a supertrait of [`Mappable`], [`Apply`], [`New`], [`Mappable2`], [`Mappable3`], [`IntoIterator`] and  [`Traversable`].
/// But making it a 'true' supertrait would require a very large number of (usually unconstrained and therefore un-inferrable) type parameters.
/// Instead, this trait has been marked as unsafe: Only implement it for types for which above traits also have been implemented!
pub unsafe trait Dimension: Container
{
    fn size(&self) -> usize;
}

unsafe impl<T> Dimension for Pair<T> {
    fn size(&self) -> usize {
        2
    }
}

unsafe impl<T, N: ArrayLength> Dimension for GenericArray<T, N> {
    fn size(&self) -> usize {
        N::USIZE
    }
}

// pub trait InnerProduct {
//     fn innerp(&self, ) -> usize;
// }
/// Calculate the inner product of two containers of the same shape.
pub fn innerp<A, R>(a: &A, b: &A) -> R
    where
    A: Mappable2<R, R> + Container<Containing<R> = A> + IntoIterator,
    R: std::iter::Sum<<A as std::iter::IntoIterator>::Item>,
    R: core::ops::Mul<R, Output = R> + Copy,
{
    let products = a.map2(b, &|x: &R, y: &R| { (*x) * (*y) });
    let result = products.into_iter().sum();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use generic_array::arr;
    #[test]
    fn transpose() {
        let v123 = arr![usize; 1,2,3];
        let v456 = arr![usize; 4,5,6];
        let three_by_two: GenericArray<GenericArray<_, _>, U2> = arr![_;v123, v456];
        println!("{:?}", three_by_two);
        let two_by_three = three_by_two.transpose();
        println!("{:?}", two_by_three);
        assert_eq!(two_by_three, arr![_; arr![usize; 1, 4], arr![usize; 2, 5], arr![usize; 3, 6]]);

        let pair_of_vecs = Pair(arr![i32; 1,2,3], arr![i32; 10, 20, 30]);
        let vec_of_pairs = pair_of_vecs.transpose();
        println!("{:?}", vec_of_pairs);
        assert_eq!(vec_of_pairs, arr![_;Pair(1, 10), Pair(2, 20), Pair(3, 30)]);
    }

    // pub fn increase<'a>(m: &'a usize) -> State<'a, usize, usize>{
    //     State { run_state: Box::new(move |n| (m + n, m + n))}
    // }

    #[test]
    fn traversable() {
        let pair = Pair(10, 20);
        // let res = pair.traverse(increase);
        // println!("{:?}", pair);
        // let pair_of_vecs = Pair(arr![usize; 1,2,3], arr![usize; 10, 20, 30]);
        // let transposed = pair_of_vecs.transpose();
        // println!("{:?}", transposed);
        // increase m = State (λn → (m + n, m + n))
    }

    #[test]
    fn innerprod() {
        let v123 = arr![usize; 1,2,3];
        let v456 = arr![usize; 4,5,6];
        let res = innerp(&v123, &v456);
        println!("{:?}", res);
    }
}
