use core::marker::PhantomData;

pub trait HList {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HNil;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HCons<H, T>(PhantomData<(H, T)>);

impl HList for HNil {}
impl<H, T: HList> HList for HCons<H, T> {}
