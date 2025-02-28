# numerical-multiset: An ordered multiset of machine numbers

[![MPLv2 licensed](https://img.shields.io/badge/license-MPLv2-blue.svg)](./LICENSE)
[![On crates.io](https://img.shields.io/crates/v/numerical-multiset.svg)](https://crates.io/crates/numerical-multiset)
[![On docs.rs](https://docs.rs/numerical-multiset/badge.svg)](https://docs.rs/numerical-multiset/)
[![Continuous Integration](https://img.shields.io/github/actions/workflow/status/HadrienG2/numerical-multiset/ci.yml?branch=master)](https://github.com/HadrienG2/numerical-multiset/actions?query=workflow%3A%22Continuous+Integration%22)
![Requires rustc 1.85.0+](https://img.shields.io/badge/rustc-1.85.0+-lightgray.svg)


## What is this

Let's break down the title of this README into more digestible chunks:

- **Multiset:** The `NumericalMultiset` container provided by this crate
  implements a generalization of the mathematical notion of set, called a
  multiset. Unlike a set, a multiset can simultaneously hold multiple elements
  that compare equal to each other.
- **Ordered:** Most multiset implementations are based on associative
  containers. In a crates.io survey performed at the time of writing, the most
  popular choice was hash maps, which do not expose any meaningful element
  ordering:
  - Any insertion can change the order of elements that is exposed by iterators.
  - There is no efficient way to find what is e.g. the smallest element
  without iterating over all elements.
  In contrast, this multiset implementation is based on an ordered associative
  container (a
  [`BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html)
  at the time of writing). This allows it to efficiently answer order-related
  queries, like in-order iteration over elements or extraction of the
  minimum/maximum element. The price to pay for these fast ordering queries is
  that the algorithmic complexity of order-unrelated multiset operations like
  insertions are deletions will scale less well as the number of elements grows.
- **Numbers:** In Rust terms, a general-purpose multiset is typically based on a
  data structure of the form `Map<Element, Collection<OtherElements>>` where
  `Map` is an associative container and `Collection` groups together elements
  that share a common key using something like a `Vec`. However, the multiset
  provided by this crate is not general-purpose, but instead specialized for
  machine number types (`u32`, `f32`...) and newtypes thereof. These number
  types share a few properties that this crate leverages to improve ergonomics,
  memory usage and execution performance:
    * The equality operator of machine numbers is defined such that if two
      numbers compare equal, they are the same number, i.e. there is no hidden
      internal information that their `PartialEq` implementation does not look
      at. This means that storing a collection of identical values like a
      general-purpose multiset does is pointless in our case, and we can instead 
      simply count the number of occurences of each value, leading to a more
      efficient sparse histogram data structure of the form `Map<Value,
      NonZeroUsize>`.
    * Machine numbers all implement the `Copy` trait, which means that we do not
      need the complexity of the standard library's associative containers API
      (with references and nontrivial use of the `Borrow` trait) and can instead 
      provide conceptually simpler and slightly more efficient value-based APIs.

You may find this crate useful when implementing windowed signal processing
algorithms where you are receiving an stream of numbers, and for each new input
beyond the first few ones you want to answer a question about the last N numbers
that you received that would naively require partially or fully sorting them.
For example, a median filter can be efficiently implemented by maintaining two
`NumericalMultiset`, one representing numbers below the current median and one
representing numbers above the current median.


## License

This crate is distributed under the terms of the MPLv2 license. See the LICENSE
file for details.

More relaxed licensing (Apache, MIT, BSD...) may also be negociated, in exchange
of a financial contribution. Contact me for details at knights_of_ni AT gmx
DOTCOM.
