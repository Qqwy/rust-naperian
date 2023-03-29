mod array;
pub mod pair;

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
///   One solution here is to write bounds like `Pair<A>: Container<Containing<B> = Pair<B>>`.
///
/// # Safety
/// For instances to make sense, implementors need to follow the rules for the two associated types mentioned below to the letter.
pub unsafe trait Container {
    /// The element type of the container.
    /// For a type Foo<T> this has to be T.
    ///
    /// # Examples:
    /// For a Vec<T>, this is T.
    /// For an Option<A>, this is A.
    type Elem;

    /// The container type with its element type
    /// changed to X.
    /// For a type Foo<T> this has to be Foo<X>
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

/// Transform a container by running a unary function element-wise on its contents.
///
/// Also known as 'Functor'.
///
/// Note that different from the usual, functionally pure, definition, we allow you to pass a FnMut.
/// (a function or closure which may contain mutable state).
/// This is done to make it easier to use Mappable in the wider Rust ecosystem.
/// Using [`Mappable::map`] with (locally) mutable state is a much more
/// lightweight alternative (for both the compiler and the unsuspecting programmer)
/// than requiring Traversable.
///
///
/// # Examples
/// Transform each element using a unary function:
/// ```rust
/// use naperian::Mappable;
/// use generic_array::arr;
/// let v123 = arr![1,2,3];
/// assert_eq!(v123.map(|x| x + 1), arr![2, 3, 4]);
/// ```
///
/// Using mutable state as a 'lightweight traversable':
///```rust
/// use naperian::Mappable;
/// use generic_array::arr;
/// let v123 = arr![1,2,3];
/// let mut sum = 0;
/// let prefix_sums = v123.map(|val| {
///     sum += val;
///     sum
/// });
/// assert_eq!(prefix_sums, arr![1, 3, 6]);
/// assert_eq!(sum, 6);
///```
pub trait Mappable<U>: Container {
    fn map(&self, fun: impl FnMut(&Self::Elem) -> U) -> Self::Containing<U>;

    fn map_by_value(self, fun: impl FnMut(Self::Elem) -> U) -> Self::Containing<U>;
}

impl<T, U> Mappable<U> for Option<T> {
    fn map(&self, fun: impl FnMut(&Self::Elem) -> U) -> Self::Containing<U> {
        Option::map(self.as_ref(), fun)
    }

    fn map_by_value(self, fun: impl FnMut(Self::Elem) -> U) -> Self::Containing<U> {
        Option::map(self, fun)
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

/// Combines a container containing functions and a container containing values
/// by elementwise running each function on the value at the same element position.
///
/// Should be implemented on `YourType<F>`, to allow chaining of `ap`.
/// (and so the compiler can automatically infer A and B from knowing F.)
pub trait Apply<A, B, F: Fn(&A) -> B>: Container {
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

/// Map a binary (two-parameter) function over two containers.
///
/// Implementing this is usually very straightforward.
/// Here is the implementation for Option:
/// ```ignore
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
/// # On FnMut
/// Just like the unary [`Mappable`], this trait method takes a [`FnMut`] rather than a plain [`Fn`],
/// for the same reason as listed there.
///
/// # Rationale for type enthousiasts
///
/// Indeed, [`Mappable2::map2`] is the same as the `liftA2` operation you might know from Haskell's Applicative typeclass.
///
/// In other words,
/// Mappable2 can always be implemented for containers implementing [`Apply`] and [`New`] (or [`NewFrom`]),
/// by currying `fun` and leveraging [`New::new`] ([`NewFrom::new_from`]) and [`Apply::ap`]:
///
/// ```ignore
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
/// ```ignore
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
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U>;
    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U>;
}

impl<A, U> Mappable2<A, U> for Option<A> {
    fn map2<'b, B: 'b>(
        &self,
        rhs: &'b Self::Containing<B>,
        mut fun: impl FnMut(&A, &'b B) -> U,
    ) -> Self::Containing<U> {
        match (self, rhs) {
            (Some(left), Some(right)) => Some(fun(left, right)),
            (_, _) => None,
        }
    }

    fn map2_by_value<B>(
        self,
        rhs: Self::Containing<B>,
        mut fun: impl FnMut(A, B) -> U,
    ) -> Self::Containing<U> {
        match (self, rhs) {
            (Some(left), Some(right)) => Some(fun(left, right)),
            (_, _) => None,
        }
    }
}

/// Map a ternary (three-parameter) function over a container.
///
/// This trait is very similar to [`Mappable`] and (especially) [`Mappable2`].
/// In particular, the same implementation trade-offs apply w.r.t. [`Mappable2`].
pub trait Mappable3<A, U>: Container {
    fn map3<B, C>(
        &self,
        second: &Self::Containing<B>,
        third: &Self::Containing<C>,
        fun: impl FnMut(&A, &B, &C) -> U,
    ) -> Self::Containing<U>;
    fn map3_by_value<B, C>(
        self,
        second: Self::Containing<B>,
        third: Self::Containing<C>,
        fun: impl FnMut(A, B, C) -> U,
    ) -> Self::Containing<U>;
}

impl<A, U> Mappable3<A, U> for Option<A> {
    fn map3<B, C>(
        &self,
        second: &Self::Containing<B>,
        third: &Self::Containing<C>,
        mut fun: impl FnMut(&A, &B, &C) -> U,
    ) -> Self::Containing<U> {
        match (self, second, third) {
            (Some(one), Some(two), Some(three)) => Some(fun(one, two, three)),
            (_, _, _) => None,
        }
    }

    fn map3_by_value<B, C>(
        self,
        second: Self::Containing<B>,
        third: Self::Containing<C>,
        mut fun: impl FnMut(A, B, C) -> U,
    ) -> Self::Containing<U> {
        match (self, second, third) {
            (Some(one), Some(two), Some(three)) => Some(fun(one, two, three)),
            (_, _, _) => None,
        }
    }
}
