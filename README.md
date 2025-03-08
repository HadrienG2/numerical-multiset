# numerical-multiset: An ordered multiset of machine numbers

[![MPLv2 licensed](https://img.shields.io/badge/license-MPLv2-blue.svg)](./LICENSE)
[![On crates.io](https://img.shields.io/crates/v/numerical-multiset.svg)](https://crates.io/crates/numerical-multiset)
[![On docs.rs](https://docs.rs/numerical-multiset/badge.svg)](https://docs.rs/numerical-multiset/)
[![Continuous Integration](https://img.shields.io/github/actions/workflow/status/HadrienG2/numerical-multiset/ci.yml?branch=master)](https://github.com/HadrienG2/numerical-multiset/actions?query=workflow%3A%22Continuous+Integration%22)
![Requires rustc 1.85.0+](https://img.shields.io/badge/rustc-1.85.0+-lightgray.svg)


## What is this?

Let's break down the title of this README into more digestible chunks:

- **Multiset:** The [`NumericalMultiset`] container provided by this crate
  implements a generalization of the mathematical set, the multiset. Unlike a
  set, a multiset can conceptually hold multiple copies of a value. This is done
  by tracking how many copies of each value are present (element
  _multiplicities_).
- **Ordered:** Multiset implementations are usually based on associative
  containers, using distinct multiset elements as keys and integer occurence
  counts as values. A popular choice is hash maps, which do not expose any
  meaningful element ordering:

  - Any key insertion may change the order of keys that is exposed by
    iterators.
  - There is no way to find e.g. the smallest key without iterating over
    all key.

  In contrast, [`NumericalMultiset`] is based on an ordered associative
  container. This allows it to efficiently answer order-related queries, like
  in-order iteration over elements or extraction of the minimum/maximum element.
  The price to pay is that order-insenstive multiset operations, like item
  insertions and removals, will scale a little less well to larger sets than in
  a hash-based implementation.
- **Numbers:** The multiset provided by this crate is not general-purpose, but
  specialized for machine number types (`u32`, `f32`...) and newtypes thereof.
  These types are all `Copy`, which lets us provide a simplified value-based
  API, that may also result in slightly improved runtime //! performance in some
  scenarios.


## When should I use it?

This crate was initially written for the purpose of implementing order-based
windowed signal processing tasks. In windowed signal processing, the program is
receiving an stream of numerical inputs, and for each new input beyond the first
few ones, it needs to answer a question about the last N numbers that were
received. Furthermore, here we are talking about questions for which the naive
algorithm involves maintaining a sorted list of the previous N data points.

In optimized code, we do not want to implement such algorithms by sorting the
window of previous data point again and again on each new data point, because
that's somewhat expensive and involves lots of redundant work. And we would
rather not insert/remove points in the middle of a sorted
[`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html) in a loop either
because that involves lots of unnecessary memory movement, which will not scale
well to larger windows. This is where [`NumericalMultiset`] can help, for some
categories of order-based queries.

For example, a median filter can be efficiently implemented by maintaining two
[`NumericalMultiset`]s, one representing numbers below the current median and
one representing numbers above the current median. This is effectively just a
sparse variation of the dense histogramming algorithm that is commonly used when
processing 8-bit images, which is powerful but sadly does not scale to
higher-resolution input data (32-bit and more) due to excessive memory usage and
poor CPU cache locality.


## License

This crate is distributed under the terms of the MPLv2 license. See the LICENSE
file for details.

More relaxed licensing (Apache, MIT, BSD...) may also be negociated, in exchange
of a financial contribution. Contact me for details at knights_of_ni AT gmx
DOTCOM.

[`NumericalMultiset`]: https://docs.rs/numerical-multiset/NumericalMultiset.html
