use generic_array::GenericArray;

/// A shorter alias for Array
/// since we use it so prevalently in this crate.
pub type Array<T, N> = GenericArray<T, N>;
