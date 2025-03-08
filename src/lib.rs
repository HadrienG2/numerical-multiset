//! This crate implements an ordered multiset of machine
//! numbers.
//!
//! Well, that sentence is quite a mouthful. Let's break it down into
//! more digestible chunks:
//!
//! - **Multiset:** The [`NumericalMultiset`] container provided by this crate
//!   implements a generalization of the mathematical set, the multiset.
//!   Unlike a set, a multiset can conceptually hold multiple copies of a value.
//!   This is done by tracking how many occurences of each value are present.
//! - **Ordered:** Multiset implementations are usually based on associative
//!   containers, using distinct multiset elements as keys and integer occurence
//!   counts as values. A popular choice is hash maps, which do not provide any
//!   meaningful key ordering:
//!
//!   - Any key insertion may change the order of keys that is exposed by
//!     iterators.
//!   - There is no way to find e.g. the smallest key without iterating over
//!     all key.
//!
//!   In contrast, [`NumericalMultiset`] is based on an ordered associative
//!   container. This allows it to efficiently answer order-related queries,
//!   like in-order iteration over elements or extraction of the minimum/maximum
//!   element. The price to pay is that order-insenstive multiset operations,
//!   like item insertions and removals, will scale a little less well to
//!   larger sets of distinct values than in a hash-based implementation.
//! - **Numbers:** The multiset provided by this crate is not general-purpose,
//!   but specialized for machine number types (`u32`, `f32`...) and newtypes
//!   thereof. These types are all `Copy`, which lets us provide a simplified
//!   value-based API, that may also result in slightly improved runtime
//!   performance in some scenarios.

use std::{
    cmp::Ordering,
    collections::btree_map::{self, BTreeMap, Entry},
    hash::Hash,
    iter::FusedIterator,
    num::NonZeroUsize,
    ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub},
};

/// An ordered multiset of machine numbers.
///
/// You can learn more about the design rationale and overall capabilities of
/// this data structure in the [crate-level documentation](index.html).
///
/// At the time of writing, this data structure is based on the standard
/// library's [`BTreeMap`], and many points of the [`BTreeMap`] documentation
/// also apply to it. In particular, it is a logic error to modify the order of
/// values stored inside of the multiset using internal mutability tricks.
///
/// # Floating-point data
///
/// To build multisets of floating-point numbers, you will need to handle the
/// fact that NaN is unordered. This can be done using one of the [`Ord`] float
/// wrappers available on crates.io, which work by either [asserting absence of
/// NaNs](https://docs.rs/ordered-float/latest/ordered_float/struct.NotNan.html)
/// or [making NaNs
/// ordered](https://docs.rs/ordered-float/latest/ordered_float/struct.OrderedFloat.html).
///
/// For optimal `NumericalMultiset` performance, we advise...
///
/// - Preferring
///   [`NotNan`](https://docs.rs/ordered-float/latest/ordered_float/struct.NotNan.html)-like
///   wrappers, whose `Ord` implementation can leverage fast hardware
///   comparisons instead of implementing other ordering semantics in software.
/// - Using them right from the point where your application receives inputs, to
///   avoid repeatedly checking your inputs for NaNs by having to rebuild such
///   wrappers every time a number is inserted into a `NumericalMultiset`.
///
/// # Terminology
///
/// Because multisets can hold multiple occurences of a value, it is useful to
/// have concise wording to distinguish between unique values and (possibly
/// duplicate) occurences of these values.
///
/// Throughout this documentation, we will use the following terminology:
///
/// - "values" refers to distinct values of type `T` as defined by the [`Eq`]
///   implementation of `T`.
/// - "items" refers to possibly duplicate occurences of a value within the
///   multiset.
/// - "multiplicity" refers to the number of occurences of a value within the
///   multiset, i.e. the number of items that are equal to this value.
///
/// # Examples
///
/// ```
/// use numerical_multiset::NumericalMultiset;
/// use std::num::NonZeroUsize;
///
/// // Create a multiset
/// let mut mset = NumericalMultiset::new();
///
/// // Inserting items is handled much like a standard library set type,
/// // except we return an Option<NonZeroUsize> instead of a boolean.
/// assert!(mset.insert(123).is_none());
/// assert!(mset.insert(456).is_none());
///
/// // This allows us to report the number of pre-existing items
/// // that have the same value, if any.
/// assert_eq!(mset.insert(123), NonZeroUsize::new(1));
///
/// // It is possible to query the minimal and maximal values cheaply, along
/// // with their multiplicity within the multiset.
/// let nonzero = |x| NonZeroUsize::new(x).unwrap();
/// assert_eq!(mset.first(), Some((123, nonzero(2))));
/// assert_eq!(mset.last(), Some((456, nonzero(1))));
///
/// // ...and it is more generally possible to iterate over values and
/// // multiplicities in order, from the smallest value to the largest one:
/// for (elem, multiplicity) in &mset {
///     println!("{elem} with multiplicity {multiplicity}");
/// }
/// ```
#[derive(Clone, Debug, Default, Eq)]
pub struct NumericalMultiset<T> {
    /// Mapping from distinct values to their multiplicities
    value_to_multiplicity: BTreeMap<T, NonZeroUsize>,

    /// Number of items = sum of all multiplicities
    len: usize,
}
//
impl<T> NumericalMultiset<T> {
    /// Makes a new, empty `NumericalMultiset`.
    ///
    /// Does not allocate anything on its own.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mset = NumericalMultiset::<i32>::new();
    /// assert!(mset.is_empty());
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn new() -> Self {
        Self {
            value_to_multiplicity: BTreeMap::new(),
            len: 0,
        }
    }

    /// Clears the multiset, removing all items.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut v = NumericalMultiset::from_iter([1, 2, 3]);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.value_to_multiplicity.clear();
        self.len = 0;
    }

    /// Number of items currently present in the multiset, including
    /// duplicate occurences of the same value.
    ///
    /// See also [`num_values()`](Self::num_values) for a count of distinct
    /// values, ignoring duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut v = NumericalMultiset::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// v.insert(1);
    /// assert_eq!(v.len(), 2);
    /// v.insert(2);
    /// assert_eq!(v.len(), 3);
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of distinct values currently present in the multiset
    ///
    /// See also [`len()`](Self::len) for a count of multiset items,
    /// including duplicate occurences of the same value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut v = NumericalMultiset::new();
    /// assert_eq!(v.num_values(), 0);
    /// v.insert(1);
    /// assert_eq!(v.num_values(), 1);
    /// v.insert(1);
    /// assert_eq!(v.num_values(), 1);
    /// v.insert(2);
    /// assert_eq!(v.num_values(), 2);
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn num_values(&self) -> usize {
        self.value_to_multiplicity.len()
    }

    /// Truth that the multiset contains no items
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut v = NumericalMultiset::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates a consuming iterator visiting all distinct values in the
    /// multiset, i.e. the mathematical support of the multiset.
    ///
    /// Values are emitted in ascending order, and the multiset cannot be used
    /// after calling this method.
    ///
    /// Call `into_iter()` (from the [`IntoIterator`] trait) to get a variation
    /// of this iterator that additionally tells you how many occurences of each
    /// value were present in the multiset, in the usual `(value, multiplicity)`
    /// format.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mset = NumericalMultiset::from_iter([3, 1, 2, 2]);
    /// assert!(mset.into_values().eq([1, 2, 3]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn into_values(
        self,
    ) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator + FusedIterator {
        self.value_to_multiplicity.into_keys()
    }

    /// Update `self.len` to match `self.value_to_multiplicity`'s contents
    ///
    /// This expensive `O(N)` operation should only be performed after calling
    /// into `BTreeMap` operations that do not provide the right hooks to update
    /// the length field more efficiently.
    fn reset_len(&mut self) {
        self.len = self.value_to_multiplicity.values().map(|x| x.get()).sum();
    }
}

impl<T: Copy> NumericalMultiset<T> {
    /// Iterator over all distinct values in the multiset, along with their
    /// multiplicities.
    ///
    /// Values are emitted in ascending order.
    ///
    /// See also [`values()`](Self::values) for a more efficient alternative if
    /// you do not need to know how many occurences of each value are present.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mset = NumericalMultiset::from_iter([3, 1, 2, 2]);
    ///
    /// let mut iter = mset.iter();
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(iter.next(), Some((1, nonzero(1))));
    /// assert_eq!(iter.next(), Some((2, nonzero(2))));
    /// assert_eq!(iter.next(), Some((3, nonzero(1))));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn iter(&self) -> Iter<'_, T> {
        self.into_iter()
    }

    /// Iterator over all distinct values in the multiset, i.e. the mathematical
    /// support of the multiset.
    ///
    /// Values are emitted in ascending order.
    ///
    /// See also [`iter()`](Self::iter) if you need to know how many occurences
    /// of each value are present in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mset = NumericalMultiset::from_iter([3, 1, 2, 2]);
    ///
    /// let mut iter = mset.values();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn values(
        &self,
    ) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator + FusedIterator + Clone {
        self.value_to_multiplicity.keys().copied()
    }
}

impl<T: Ord> NumericalMultiset<T> {
    /// Returns `true` if the multiset contains at least one occurence of a
    /// value.
    ///
    /// See also [`multiplicity()`](Self::multiplicity) if you need to know how
    /// many occurences of a value are present inside of the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mset = NumericalMultiset::from_iter([1, 2, 2]);
    ///
    /// assert_eq!(mset.contains(1), true);
    /// assert_eq!(mset.contains(2), true);
    /// assert_eq!(mset.contains(3), false);
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn contains(&self, value: T) -> bool {
        self.value_to_multiplicity.contains_key(&value)
    }

    /// Returns the number of occurences of a value inside of the multiset, or
    /// `None` if this value is not present.
    ///
    /// See also [`contains()`](Self::contains) for a more efficient alternative
    /// if you only need to know whether at least one occurence of `value` is
    /// present inside of the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mset = NumericalMultiset::from_iter([1, 2, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.multiplicity(1), Some(nonzero(1)));
    /// assert_eq!(mset.multiplicity(2), Some(nonzero(2)));
    /// assert_eq!(mset.multiplicity(3), None);
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn multiplicity(&self, value: T) -> Option<NonZeroUsize> {
        self.value_to_multiplicity.get(&value).copied()
    }

    /// Returns `true` if `self` has no items in common with `other`. This is
    /// logically equivalent to checking for an empty intersection, but may be
    /// more efficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let a = NumericalMultiset::from_iter([1, 2, 2]);
    /// let mut b = NumericalMultiset::new();
    ///
    /// assert!(a.is_disjoint(&b));
    /// b.insert(3);
    /// assert!(a.is_disjoint(&b));
    /// b.insert(2);
    /// assert!(!a.is_disjoint(&b));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let mut iter1 = self.value_to_multiplicity.keys().peekable();
        let mut iter2 = other.value_to_multiplicity.keys().peekable();
        'joint_iter: loop {
            match (iter1.peek(), iter2.peek()) {
                // As long as both iterators yield values, must watch out for
                // common values through well-ordered joint iteration.
                (Some(value1), Some(value2)) => {
                    match value1.cmp(value2) {
                        // Advance the iterator which is behind, trying to make
                        // it reach the same value as the other iterator.
                        Ordering::Less => {
                            let _ = iter1.next();
                            continue 'joint_iter;
                        }
                        Ordering::Greater => {
                            let _ = iter2.next();
                            continue 'joint_iter;
                        }

                        // The same value was yielded by both iterators, which
                        // means that the multisets are not disjoint.
                        Ordering::Equal => return false,
                    }
                }

                // Once one iterator ends, we know there is no common value
                // left, so we can conclude that the multisets are disjoint.
                (Some(_), None) | (None, Some(_)) | (None, None) => return true,
            }
        }
    }

    /// Returns `true` if this multiset is a subset of another, i.e., `other`
    /// contains at least all the items in `self`.
    ///
    /// In a multiset context, this means that if `self` contains N occurences
    /// of a certain value, then `other` must contain at least N occurences of
    /// that value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let sup = NumericalMultiset::from_iter([1, 2, 2]);
    /// let mut mset = NumericalMultiset::new();
    ///
    /// assert!(mset.is_subset(&sup));
    /// mset.insert(2);
    /// assert!(mset.is_subset(&sup));
    /// mset.insert(2);
    /// assert!(mset.is_subset(&sup));
    /// mset.insert(2);
    /// assert!(!mset.is_subset(&sup));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn is_subset(&self, other: &Self) -> bool {
        let mut other_iter = other.value_to_multiplicity.iter().peekable();
        for (value, &multiplicity) in self.value_to_multiplicity.iter() {
            // Check if this value also exists in the other iterator
            'other_iter: loop {
                match other_iter.peek() {
                    Some((other_value, other_multiplicity)) => match value.cmp(other_value) {
                        // Other iterator is ahead, and because it emits values
                        // in sorted order, we know it's never going to get back
                        // to the current value.
                        //
                        // We can thus conclude that `other` does not contain
                        // `value` and thus `self` is not a subset of it.
                        Ordering::Less => return false,

                        // Other iterator is behind and may get to the current
                        // value later in its sorted sequence, so we must
                        // advance it and check again.
                        Ordering::Greater => {
                            let _ = other_iter.next();
                            continue 'other_iter;
                        }

                        // Current value exists in both iterators
                        Ordering::Equal => {
                            // For `self` to be a subset, `other` must also
                            // contain at least the same number of occurences of
                            // this common value. Check this.
                            if **other_multiplicity < multiplicity {
                                return false;
                            }

                            // We're done checking this common value, now we can
                            // advance the other iterator beyond it and move to
                            // the next value from `self`.
                            let _ = other_iter.next();
                            break 'other_iter;
                        }
                    },

                    // Other iterator has ended, it won't yield `value`. Thus
                    // `other` doesn't contain `value` and therefore `self` is
                    // not a subset of `other`.
                    None => return false,
                }
            }
        }
        true
    }

    /// Returns `true` if this multiset is a superset of another, i.e., `self`
    /// contains at least all the items in `other`.
    ///
    /// In a multiset context, this means that if `other` contains N occurences
    /// of a certain value, then `self` must contain at least N occurences of
    /// that value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let sub = NumericalMultiset::from_iter([1, 2, 2]);
    /// let mut mset = NumericalMultiset::new();
    ///
    /// assert!(!mset.is_superset(&sub));
    ///
    /// mset.insert(3);
    /// mset.insert(1);
    /// assert!(!mset.is_superset(&sub));
    ///
    /// mset.insert(2);
    /// assert!(!mset.is_superset(&sub));
    ///
    /// mset.insert(2);
    /// assert!(mset.is_superset(&sub));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Remove all occurences of the smallest value from the multiset, if any.
    ///
    /// Returns the former smallest value along with the number of occurences of
    /// this value that were previously present in the multiset.
    ///
    /// See also [`pop_first()`](Self::pop_first) if you only want to remove one
    /// occurence of the smallest value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 1, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.pop_all_first(), Some((1, nonzero(2))));
    /// assert_eq!(mset.pop_all_first(), Some((2, nonzero(1))));
    /// assert_eq!(mset.pop_all_first(), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_all_first(&mut self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .pop_first()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Remove all occurences of the largest value from the multiset, if any
    ///
    /// Returns the former largest value along with the number of occurences of
    /// this value that were previously present in the multiset.
    ///
    /// See also [`pop_last()`](Self::pop_last) if you only want to remove one
    /// occurence of the largest value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 1, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.pop_all_last(), Some((2, nonzero(1))));
    /// assert_eq!(mset.pop_all_last(), Some((1, nonzero(2))));
    /// assert_eq!(mset.pop_all_last(), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_all_last(&mut self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .pop_last()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Insert an item into the multiset, tell how many identical items were
    /// already present in the multiset before insertion.
    ///
    /// See also [`insert_multiple()`](Self::insert_multiple) for a more
    /// efficient alternative if you need to insert multiple copies of a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::new();
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.insert(1), None);
    /// assert_eq!(mset.insert(1), Some(nonzero(1)));
    /// assert_eq!(mset.insert(1), Some(nonzero(2)));
    /// assert_eq!(mset.insert(2), None);
    ///
    /// assert_eq!(mset.len(), 4);
    /// assert_eq!(mset.num_values(), 2);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> Option<NonZeroUsize> {
        self.insert_multiple(value, NonZeroUsize::new(1).unwrap())
    }

    /// Insert multiple copies of an item, tell how many identical items were
    /// already present in the multiset.
    ///
    /// This method is typically used for the purpose of efficiently
    /// transferring all copies of a value from one multiset to another.
    ///
    /// See also [`insert()`](Self::insert) for a convenience shortcut in cases
    /// where you only need to insert one copy of a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::new();
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.insert_multiple(1, nonzero(2)), None);
    /// assert_eq!(mset.insert_multiple(1, nonzero(3)), Some(nonzero(2)));
    /// assert_eq!(mset.insert_multiple(2, nonzero(2)), None);
    ///
    /// assert_eq!(mset.len(), 7);
    /// assert_eq!(mset.num_values(), 2);
    /// ```
    #[inline]
    pub fn insert_multiple(&mut self, value: T, count: NonZeroUsize) -> Option<NonZeroUsize> {
        let result = match self.value_to_multiplicity.entry(value) {
            Entry::Vacant(v) => {
                v.insert(count);
                None
            }
            Entry::Occupied(mut o) => {
                let old_count = *o.get();
                *o.get_mut() = old_count
                    .checked_add(count.get())
                    .expect("Multiplicity counter has overflown");
                Some(old_count)
            }
        };
        self.len += count.get();
        result
    }

    /// Insert multiple copies of a value, replacing all occurences of this
    /// value that were previously present in the multiset. Tell how many
    /// occurences of the value were previously present in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::new();
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.replace_all(1, nonzero(2)), None);
    /// assert_eq!(mset.replace_all(1, nonzero(3)), Some(nonzero(2)));
    /// assert_eq!(mset.replace_all(2, nonzero(2)), None);
    ///
    /// assert_eq!(mset.len(), 5);
    /// assert_eq!(mset.num_values(), 2);
    /// ```
    #[inline]
    pub fn replace_all(&mut self, value: T, count: NonZeroUsize) -> Option<NonZeroUsize> {
        let result = match self.value_to_multiplicity.entry(value) {
            Entry::Vacant(v) => {
                v.insert(count);
                None
            }
            Entry::Occupied(mut o) => {
                let old_count = *o.get();
                *o.get_mut() = count;
                self.len -= old_count.get();
                Some(old_count)
            }
        };
        self.len += count.get();
        result
    }

    /// Attempt to remove one item from the multiset, on success tell how many
    /// identical items were previously present in the multiset (including the
    /// one that was just removed).
    ///
    /// See also [`remove_all()`](Self::remove_all) if you want to remove all
    /// occurences of a value from the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 1, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.remove(1), Some(nonzero(2)));
    /// assert_eq!(mset.remove(1), Some(nonzero(1)));
    /// assert_eq!(mset.remove(1), None);
    /// assert_eq!(mset.remove(2), Some(nonzero(1)));
    /// assert_eq!(mset.remove(2), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn remove(&mut self, value: T) -> Option<NonZeroUsize> {
        match self.value_to_multiplicity.entry(value) {
            Entry::Vacant(_) => None,
            Entry::Occupied(mut o) => {
                let old_multiplicity = *o.get();
                self.len -= 1;
                match NonZeroUsize::new(old_multiplicity.get() - 1) {
                    Some(new_multiplicity) => {
                        *o.get_mut() = new_multiplicity;
                    }
                    None => {
                        o.remove_entry();
                    }
                }
                Some(old_multiplicity)
            }
        }
    }

    /// Attempt to remove all occurences of a value from the multiset, on
    /// success tell how many items were removed from the multiset.
    ///
    /// See also [`remove()`](Self::remove) if you only want to remove one
    /// occurence of a value from the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 1, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.remove_all(1), Some(nonzero(2)));
    /// assert_eq!(mset.remove_all(1), None);
    /// assert_eq!(mset.remove_all(2), Some(nonzero(1)));
    /// assert_eq!(mset.remove_all(2), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn remove_all(&mut self, value: T) -> Option<NonZeroUsize> {
        let result = self.value_to_multiplicity.remove(&value);
        self.len -= result.map_or(0, |nz| nz.get());
        result
    }

    /// Splits the collection into two at the specified `value`.
    ///
    /// This returns a new multiset containing all items greater than or equal
    /// to `value`. The multiset on which this method was called will retain all
    /// items strictly smaller than `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut a = NumericalMultiset::from_iter([1, 2, 2, 3, 3, 3, 4]);
    /// let b = a.split_off(3);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.iter().eq([
    ///     (1, nonzero(1)),
    ///     (2, nonzero(2)),
    /// ]));
    /// assert!(b.iter().eq([
    ///     (3, nonzero(3)),
    ///     (4, nonzero(1)),
    /// ]));
    /// ```
    pub fn split_off(&mut self, value: T) -> Self {
        let mut result = Self {
            value_to_multiplicity: self.value_to_multiplicity.split_off(&value),
            len: 0,
        };
        self.reset_len();
        result.reset_len();
        result
    }
}

impl<T: Copy + Ord> NumericalMultiset<T> {
    /// Double-ended iterator over a sub-range of values and their
    /// multiplicities
    ///
    /// The simplest way is to use the range syntax `min..max`, thus
    /// `range(min..max)` will yield values from `min` (inclusive) to `max`
    /// (exclusive).
    ///
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so
    /// for example `range((Excluded(4), Included(10)))` will yield a
    /// left-exclusive, right-inclusive value range from 4 to 10.
    ///
    /// # Panics
    ///
    /// May panic if range `start > end`, or if range `start == end` and both
    /// bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mset = NumericalMultiset::from_iter([3, 3, 5, 8, 8]);
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(mset.range(4..).eq([
    ///     (5, nonzero(1)),
    ///     (8, nonzero(2)),
    /// ]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn range<R>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = (T, NonZeroUsize)> + FusedIterator
    where
        R: RangeBounds<T>,
    {
        self.value_to_multiplicity
            .range(range)
            .map(|(&k, &v)| (k, v))
    }

    /// Visits the items representing the difference, i.e., those that are in
    /// `self` but not in `other`. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The difference is computed item-wise, not value-wise, so if both
    /// `self` and `other` contain occurences of a certain value `v` with
    /// respective multiplicities `s` and `o`, then...
    ///
    /// - If `self` contains more occurences of `v` than `other` (i.e. `s > o`),
    ///   then the difference will contain `s - o` occurences of `v`.
    /// - Otherwise (if `s <= o`) the difference will not contain any occurence
    ///   of `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.difference(&b).eq([
    ///     (1, nonzero(2)),
    ///     (2, nonzero(1)),
    /// ]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter = self.iter();
        let mut other_iter = other.iter().peekable();
        std::iter::from_fn(move || {
            // Advance self iterator normally
            let (mut value, mut multiplicity) = iter.next()?;

            // Check if this value also exists in the other iterator
            'other_iter: loop {
                match other_iter.peek() {
                    Some((other_value, other_multiplicity)) => match value.cmp(other_value) {
                        // Other iterator is ahead, and because it emits values
                        // in sorted order, we know it's never going to get back
                        // to the current value. So we can yield it.
                        Ordering::Less => return Some((value, multiplicity)),

                        // Other iterator is behind and may get to the current
                        // value later in its sorted sequence, so we must
                        // advance it and check again.
                        Ordering::Greater => {
                            let _ = other_iter.next();
                            continue 'other_iter;
                        }

                        // Current value exists in both iterators
                        Ordering::Equal => {
                            // If `self` contains more occurences of the common
                            // value than `other`, then we must still yield
                            // those occurences.
                            if multiplicity > *other_multiplicity {
                                let difference_multiplicity = NonZeroUsize::new(
                                    multiplicity.get() - other_multiplicity.get(),
                                )
                                .expect("Checked above that this is fine");
                                let _ = other_iter.next();
                                return Some((value, difference_multiplicity));
                            } else {
                                // Otherwise, discard this entry on both sides
                                // and move on to the next iterator items.
                                let _ = other_iter.next();
                                (value, multiplicity) = iter.next()?;
                                continue 'other_iter;
                            }
                        }
                    },

                    // Other iterator has ended, can yield all remaining items
                    None => return Some((value, multiplicity)),
                }
            }
        })
    }

    /// Visits the items representing the symmetric difference, i.e., those
    /// that are in `self` or in `other` but not in both. They are sorted in
    /// ascending value order and emitted in the usual deduplicated `(value,
    /// multiplicity)` format.
    ///
    /// The symmetric difference is computed item-wise, not value-wise, so if
    /// both `self` and `other` contain occurences of a certain value `v` with
    /// respective multiplicities `s` and `o`, then...
    ///
    /// - If `self` contains as many occurences of `v` as `other` (i.e. `s ==
    ///   o`), then the symmetric difference will not contain any occurence of
    ///   `v`.
    /// - Otherwise (if `s != o`) the symmetric difference will contain
    ///   `s.abs_diff(o)` occurences of `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.symmetric_difference(&b).eq([
    ///     (1, nonzero(2)),
    ///     (2, nonzero(1)),
    ///     (4, nonzero(1)),
    /// ]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || {
            'joint_iter: loop {
                match (iter1.peek(), iter2.peek()) {
                    // As long as both iterators yield values, must be careful to
                    // yield values from both iterators, in the right order, and to
                    // skip common values.
                    (Some((value1, multiplicity1)), Some((value2, multiplicity2))) => {
                        match value1.cmp(value2) {
                            // Yield the smallest value, if any, advancing the
                            // corresponding iterator along the way
                            Ordering::Less => return iter1.next(),
                            Ordering::Greater => return iter2.next(),

                            // Same value was yielded by both iterators
                            Ordering::Equal => {
                                // If the value was yielded with different
                                // multiplicities, then we must still yield an
                                // entry with a multiplicity that is the
                                // absolute difference of these multiplicities.
                                if multiplicity1 != multiplicity2 {
                                    let value12 = *value1;
                                    let difference_multiplicity = NonZeroUsize::new(
                                        multiplicity1.get().abs_diff(multiplicity2.get()),
                                    )
                                    .expect("Checked above that this is fine");
                                    let _ = (iter1.next(), iter2.next());
                                    return Some((value12, difference_multiplicity));
                                } else {
                                    // Otherwise ignore the common value,
                                    // advance both iterators and try again
                                    let _ = (iter1.next(), iter2.next());
                                    continue 'joint_iter;
                                }
                            }
                        }
                    }

                    // One one iterator ends, we know there's no common value
                    // left and there is no sorted sequence merging business to
                    // care about, so we can just yield the remainder as-is.
                    (Some(_), None) => return iter1.next(),
                    (None, Some(_)) => return iter2.next(),
                    (None, None) => return None,
                }
            }
        })
    }

    /// Visits the items representing the intersection, i.e., those that are
    /// both in `self` and `other`. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The intersection is computed item-wise, not value-wise, so if both
    /// `self` and `other` contain occurences of a certain value `v` with
    /// respective multiplicities `s` and `o`, then the intersection will
    /// contain `s.min(o)` occurences of `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.intersection(&b).eq([
    ///     (2, nonzero(1)),
    ///     (3, nonzero(1)),
    /// ]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn intersection<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || {
            'joint_iter: loop {
                match (iter1.peek(), iter2.peek()) {
                    // As long as both iterators yield values, must be careful
                    // to yield common values with merged multiplicities
                    (Some((value1, multiplicity1)), Some((value2, multiplicity2))) => {
                        match value1.cmp(value2) {
                            // Advance the iterator which is behind, trying to make
                            // it reach the same value as the other iterator.
                            Ordering::Less => {
                                let _ = iter1.next();
                                continue 'joint_iter;
                            }
                            Ordering::Greater => {
                                let _ = iter2.next();
                                continue 'joint_iter;
                            }

                            // Merge items associated with a common value
                            Ordering::Equal => {
                                let value12 = *value1;
                                let multiplicity12 = *multiplicity1.min(multiplicity2);
                                let _ = (iter1.next(), iter2.next());
                                return Some((value12, multiplicity12));
                            }
                        }
                    }

                    // One one iterator ends, we know there's no common value
                    // left, so we can just yield nothing.
                    (Some(_), None) | (None, Some(_)) | (None, None) => return None,
                }
            }
        })
    }

    /// Visits the items representing the union, i.e., those that are in
    /// either `self` or `other`, without counting values that are present in
    /// both multisets twice. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The union is computed item-wise, not value-wise, so if both
    /// `self` and `other` contain occurences of a certain value `v` with
    /// respective multiplicities `s` and `o`, then the union will contain
    /// `s.max(o)` occurences of `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.union(&b).eq([
    ///     (1, nonzero(2)),
    ///     (2, nonzero(2)),
    ///     (3, nonzero(1)),
    ///     (4, nonzero(1)),
    /// ]));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn union<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || match (iter1.peek(), iter2.peek()) {
            // As long as both iterators yield values, must be careful to
            // yield values in the right order and merge common multiplicities
            (Some((value1, multiplicity1)), Some((value2, multiplicity2))) => {
                match value1.cmp(value2) {
                    // Yield non-common values in the right order
                    Ordering::Less => iter1.next(),
                    Ordering::Greater => iter2.next(),

                    // Merge items associated with a common value
                    Ordering::Equal => {
                        let value12 = *value1;
                        let multiplicity12 = *multiplicity1.max(multiplicity2);
                        let _ = (iter1.next(), iter2.next());
                        Some((value12, multiplicity12))
                    }
                }
            }

            // Once one iterator ends, we can just yield the rest as-is
            (Some(_), None) => iter1.next(),
            (None, Some(_)) => iter2.next(),
            (None, None) => None,
        })
    }

    /// Minimal value present in the multiset, if any, along with its
    /// multiplicity.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::new();
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.first(), None);
    /// mset.insert(2);
    /// assert_eq!(mset.first(), Some((2, nonzero(1))));
    /// mset.insert(2);
    /// assert_eq!(mset.first(), Some((2, nonzero(2))));
    /// mset.insert(1);
    /// assert_eq!(mset.first(), Some((1, nonzero(1))));
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn first(&self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .first_key_value()
            .map(|(&k, &v)| (k, v))
    }

    /// Maximal value present in the multiset, if any, along with its
    /// multiplicity.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::new();
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(mset.last(), None);
    /// mset.insert(1);
    /// assert_eq!(mset.last(), Some((1, nonzero(1))));
    /// mset.insert(1);
    /// assert_eq!(mset.last(), Some((1, nonzero(2))));
    /// mset.insert(2);
    /// assert_eq!(mset.last(), Some((2, nonzero(1))));
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn last(&self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .last_key_value()
            .map(|(&k, &v)| (k, v))
    }

    /// Remove the smallest item from the multiset.
    ///
    /// See also [`pop_all_first()`](Self::pop_all_first) if you want to remove
    /// all occurences of the smallest value from the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut mset = NumericalMultiset::new();
    /// mset.insert(1);
    /// mset.insert(1);
    /// mset.insert(2);
    ///
    /// assert_eq!(mset.pop_first(), Some(1));
    /// assert_eq!(mset.pop_first(), Some(1));
    /// assert_eq!(mset.pop_first(), Some(2));
    /// assert_eq!(mset.pop_first(), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_first(&mut self) -> Option<T> {
        let mut occupied = self.value_to_multiplicity.first_entry()?;
        let old_multiplicity = *occupied.get();
        let value = *occupied.key();
        match NonZeroUsize::new(old_multiplicity.get() - 1) {
            Some(new_multiplicity) => {
                *occupied.get_mut() = new_multiplicity;
            }
            None => {
                occupied.remove_entry();
            }
        }
        self.len -= 1;
        Some(value)
    }

    /// Remove the largest item from the multiset.
    ///
    /// See also [`pop_all_last()`](Self::pop_all_last) if you want to remove
    /// all occurences of the smallest value from the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut mset = NumericalMultiset::new();
    /// mset.insert(1);
    /// mset.insert(1);
    /// mset.insert(2);
    ///
    /// assert_eq!(mset.pop_last(), Some(2));
    /// assert_eq!(mset.pop_last(), Some(1));
    /// assert_eq!(mset.pop_last(), Some(1));
    /// assert_eq!(mset.pop_last(), None);
    /// ```
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_last(&mut self) -> Option<T> {
        let mut occupied = self.value_to_multiplicity.last_entry()?;
        let old_multiplicity = *occupied.get();
        let value = *occupied.key();
        match NonZeroUsize::new(old_multiplicity.get() - 1) {
            Some(new_multiplicity) => {
                *occupied.get_mut() = new_multiplicity;
            }
            None => {
                occupied.remove_entry();
            }
        }
        self.len -= 1;
        Some(value)
    }

    /// Retains only the items specified by the predicate.
    ///
    /// For efficiency reasons, the filtering callback `f` is not run once per
    /// item, but once per distinct value present inside of the multiset.
    /// However, it is also provided with the multiplicity of that value within
    /// the multiset, which can be used as a filtering criterion.
    ///
    /// Furthermore, you get read/write access to the multiplicity, which allows
    /// you to change it if you desire to do so.
    ///
    /// In other words, this method removes all values `v` with multiplicity `m`
    /// for which `f(v, m)` returns `false`, and allows changing the
    /// multiplicity for all values where `f` returns `true`.
    ///
    /// Values are visited in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 1, 2, 3, 4, 4, 5, 5, 5]);
    /// // Keep even values with an even multiplicity
    /// // and odd values with an odd multiplicity.
    /// mset.retain(|value, multiplicity| value % 2 == multiplicity.get() % 2);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(mset.iter().eq([
    ///     (3, nonzero(1)),
    ///     (4, nonzero(2)),
    ///     (5, nonzero(3)),
    /// ]));
    /// ```
    pub fn retain(&mut self, mut f: impl FnMut(T, &mut NonZeroUsize) -> bool) {
        self.value_to_multiplicity.retain(|&k, v| f(k, v));
        self.reset_len();
    }

    /// Moves all items from `other` into `self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut a = NumericalMultiset::from_iter([1, 1, 2, 3]);
    /// let mut b = NumericalMultiset::from_iter([3, 3, 4, 5]);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 8);
    /// assert!(b.is_empty());
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(a.iter().eq([
    ///     (1, nonzero(2)),
    ///     (2, nonzero(1)),
    ///     (3, nonzero(3)),
    ///     (4, nonzero(1)),
    ///     (5, nonzero(1)),
    /// ]));
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        // Fast path when self is empty
        if self.is_empty() {
            std::mem::swap(self, other);
            return;
        }

        // Otherwise just insert everything into self. This is the fastest
        // available approach because...
        //
        // - BTreeMap::append() does not have the right semantics, if both self
        //   and other contain entries associated with a certain value it will
        //   discard the entries from self instead of adding those from others.
        // - BTreeMap does not externally expose a mutable iterator that allows
        //   for both modification of existing entries and insertions of new
        //   entries, which is what we would need in order to implement this
        //   loop more efficiently.
        for (value, multiplicity) in other.iter() {
            self.insert_multiple(value, multiplicity);
        }
        other.clear();
    }
}

impl<T: Copy + Ord> BitAnd<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the intersection of `self` and `rhs` as a new `NumericalMultiset<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    /// assert_eq!(
    ///     &a & &b,
    ///     NumericalMultiset::from_iter([2, 3])
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    fn bitand(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.intersection(rhs).collect()
    }
}

impl<T: Copy + Ord> BitOr<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the union of `self` and `rhs` as a new `NumericalMultiset<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    /// assert_eq!(
    ///     &a | &b,
    ///     NumericalMultiset::from_iter([1, 1, 2, 2, 3, 4])
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    fn bitor(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.union(rhs).collect()
    }
}

impl<T: Copy + Ord> BitXor<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `NumericalMultiset<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    /// assert_eq!(
    ///     &a ^ &b,
    ///     NumericalMultiset::from_iter([1, 1, 2, 4])
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    fn bitxor(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.symmetric_difference(rhs).collect()
    }
}

impl<T: Ord> Extend<T> for NumericalMultiset<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for element in iter {
            self.insert(element);
        }
    }
}

impl<T: Ord> Extend<(T, NonZeroUsize)> for NumericalMultiset<T> {
    /// More efficient alternative to [`Extend<T>`] for cases where you know in
    /// advance that you are going to insert several copies of a value
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mset = NumericalMultiset::from_iter([1, 2, 3]);
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// mset.extend([(3, nonzero(3)), (4, nonzero(2))]);
    /// assert_eq!(mset, NumericalMultiset::from_iter([1, 2, 3, 3, 3, 3, 4, 4]));
    /// ```
    fn extend<I: IntoIterator<Item = (T, NonZeroUsize)>>(&mut self, iter: I) {
        for (value, count) in iter {
            self.insert_multiple(value, count);
        }
    }
}

impl<T: Ord> FromIterator<T> for NumericalMultiset<T> {
    #[must_use = "Only effect is to produce a result"]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

impl<T: Ord> FromIterator<(T, NonZeroUsize)> for NumericalMultiset<T> {
    /// More efficient alternative to [`FromIterator<T>`] for cases where you
    /// know in advance that you are going to insert several copies of a value
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     NumericalMultiset::from_iter([1, 2, 2, 2, 3, 3]),
    ///     NumericalMultiset::from_iter([
    ///         (1, nonzero(1)),
    ///         (2, nonzero(3)),
    ///         (3, nonzero(2)),
    ///     ])
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    fn from_iter<I: IntoIterator<Item = (T, NonZeroUsize)>>(iter: I) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

impl<T: Hash> Hash for NumericalMultiset<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value_to_multiplicity.hash(state)
    }
}

impl<'a, T: Copy> IntoIterator for &'a NumericalMultiset<T> {
    type Item = (T, NonZeroUsize);
    type IntoIter = Iter<'a, T>;

    #[must_use = "Only effect is to produce a result"]
    fn into_iter(self) -> Self::IntoIter {
        Iter(self.value_to_multiplicity.iter())
    }
}
//
/// An iterator over the contents of an [`NumericalMultiset`], sorted by value.
///
/// This `struct` is created by the [`iter()`](NumericalMultiset::iter) method on
/// [`NumericalMultiset`]. See its documentation for more.
#[derive(Clone, Debug, Default)]
pub struct Iter<'a, T: Copy>(btree_map::Iter<'a, T, NonZeroUsize>);
//
impl<T: Copy> DoubleEndedIterator for Iter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(&k, &v)| (k, v))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n).map(|(&k, &v)| (k, v))
    }
}
//
impl<T: Copy> ExactSizeIterator for Iter<'_, T> {
    #[must_use = "Only effect is to produce a result"]
    fn len(&self) -> usize {
        self.0.len()
    }
}
//
impl<T: Copy> FusedIterator for Iter<'_, T> {}
//
impl<T: Copy> Iterator for Iter<'_, T> {
    type Item = (T, NonZeroUsize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(&k, &v)| (k, v))
    }

    #[must_use = "Only effect is to produce a result"]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.0.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.0.last().map(|(&k, &v)| (k, v))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n).map(|(&k, &v)| (k, v))
    }

    fn is_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        true
    }
}

impl<T> IntoIterator for NumericalMultiset<T> {
    type Item = (T, NonZeroUsize);
    type IntoIter = IntoIter<T>;

    /// Gets an iterator for moving out the `NumericalMultiset`’s contents.
    ///
    /// Items are grouped by value and emitted in `(value, multiplicity)`
    /// format, in ascending value order.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mset = NumericalMultiset::from_iter([3, 1, 2, 2]);
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert!(mset.into_iter().eq([
    ///     (1, nonzero(1)),
    ///     (2, nonzero(2)),
    ///     (3, nonzero(1))
    /// ]));
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.value_to_multiplicity.into_iter())
    }
}
//
/// An owning iterator over the contents of an [`NumericalMultiset`], sorted by
/// value.
///
/// This struct is created by the `into_iter()` method on [`NumericalMultiset`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
#[derive(Debug, Default)]
pub struct IntoIter<T>(btree_map::IntoIter<T, NonZeroUsize>);
//
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n)
    }
}
//
impl<T> ExactSizeIterator for IntoIter<T> {
    #[must_use = "Only effect is to produce a result"]
    fn len(&self) -> usize {
        self.0.len()
    }
}
//
impl<T> FusedIterator for IntoIter<T> {}
//
impl<T> Iterator for IntoIter<T> {
    type Item = (T, NonZeroUsize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[must_use = "Only effect is to produce a result"]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.0.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }

    fn is_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        true
    }
}

impl<T: Ord> Ord for NumericalMultiset<T> {
    #[must_use = "Only effect is to produce a result"]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value_to_multiplicity.cmp(&other.value_to_multiplicity)
    }
}

impl<T: PartialEq> PartialEq for NumericalMultiset<T> {
    #[must_use = "Only effect is to produce a result"]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.value_to_multiplicity == other.value_to_multiplicity
    }
}

impl<T: PartialOrd> PartialOrd for NumericalMultiset<T> {
    #[must_use = "Only effect is to produce a result"]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value_to_multiplicity
            .partial_cmp(&other.value_to_multiplicity)
    }
}

impl<T: Copy + Ord> Sub<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the difference of `self` and `rhs` as a new `NumericalMultiset<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let a = NumericalMultiset::from_iter([1, 1, 2, 2, 3]);
    /// let b = NumericalMultiset::from_iter([2, 3, 4]);
    /// assert_eq!(
    ///     &a - &b,
    ///     NumericalMultiset::from_iter([1, 1, 2])
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    fn sub(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.difference(rhs).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::{prelude::*, sample::SizeRange};
    use std::{
        cmp::Ordering,
        collections::HashSet,
        fmt::Debug,
        hash::{BuildHasher, RandomState},
        ops::Range,
    };

    /// Clearer name for the constant 1 in NonZeroUsize format
    const ONE: NonZeroUsize = NonZeroUsize::MIN;

    /// Alternative to Iterator::eq that prints a clearer message on failure
    fn check_equal_iterable<V, It1, It2>(it1: It1, it2: It2)
    where
        It1: IntoIterator<Item = V>,
        It2: IntoIterator<Item = V>,
        V: Debug + PartialEq,
    {
        assert_eq!(
            it1.into_iter().collect::<Vec<_>>(),
            it2.into_iter().collect::<Vec<_>>(),
        );
    }

    /// Alternative to it.count() == 0 that prints a clearer message on failure
    fn check_empty_iterable<It>(it: It)
    where
        It: IntoIterator,
        It::Item: Debug + PartialEq,
    {
        check_equal_iterable(it, std::iter::empty());
    }

    /// Check properties that should be true of any pair of multisets
    fn check_any_mset_pair(mset1: &NumericalMultiset<i32>, mset2: &NumericalMultiset<i32>) {
        let intersection = mset1 & mset2;
        for (val, mul) in &intersection {
            assert_eq!(
                mul,
                mset1
                    .multiplicity(val)
                    .unwrap()
                    .min(mset2.multiplicity(val).unwrap()),
            );
        }
        for val1 in mset1.values() {
            assert!(intersection.contains(val1) || !mset2.contains(val1));
        }
        for val2 in mset2.values() {
            assert!(intersection.contains(val2) || !mset1.contains(val2));
        }
        check_equal_iterable(mset1.intersection(mset2), &intersection);

        let union = mset1 | mset2;
        for (val, mul) in &union {
            assert_eq!(
                mul.get(),
                mset1
                    .multiplicity(val)
                    .map_or(0, |nz| nz.get())
                    .max(mset2.multiplicity(val).map_or(0, |nz| nz.get()))
            );
        }
        for val in mset1.values().chain(mset2.values()) {
            assert!(union.contains(val));
        }
        check_equal_iterable(mset1.union(mset2), &union);

        let difference = mset1 - mset2;
        for (val, mul) in &difference {
            assert_eq!(
                mul.get(),
                mset1
                    .multiplicity(val)
                    .unwrap()
                    .get()
                    .checked_sub(mset2.multiplicity(val).map_or(0, |nz| nz.get()))
                    .unwrap()
            );
        }
        for (val, mul1) in mset1 {
            assert!(difference.contains(val) || mset2.multiplicity(val).unwrap() >= mul1);
        }
        check_equal_iterable(mset1.difference(mset2), difference);

        let symmetric_difference = mset1 ^ mset2;
        for (val, mul) in &symmetric_difference {
            assert_eq!(
                mul.get(),
                mset1
                    .multiplicity(val)
                    .map_or(0, |nz| nz.get())
                    .abs_diff(mset2.multiplicity(val).map_or(0, |nz| nz.get()))
            );
        }
        for (val1, mul1) in mset1 {
            assert!(
                symmetric_difference.contains(val1) || mset2.multiplicity(val1).unwrap() >= mul1
            );
        }
        for (val2, mul2) in mset2 {
            assert!(
                symmetric_difference.contains(val2) || mset1.multiplicity(val2).unwrap() >= mul2
            );
        }
        check_equal_iterable(mset1.symmetric_difference(mset2), symmetric_difference);

        assert_eq!(mset1.is_disjoint(mset2), intersection.is_empty(),);

        if mset1.is_subset(mset2) {
            for (val, mul1) in mset1 {
                assert!(mset2.multiplicity(val).unwrap() >= mul1);
            }
        } else {
            assert!(
                mset1.iter().any(|(val1, mul1)| {
                    mset2.multiplicity(val1).is_none_or(|mul2| mul2 < mul1)
                })
            )
        }
        assert_eq!(mset2.is_superset(mset1), mset1.is_subset(mset2));

        let mut combined = mset1.clone();
        let mut appended = mset2.clone();
        combined.append(&mut appended);
        assert_eq!(
            combined,
            mset1
                .iter()
                .chain(mset2.iter())
                .collect::<NumericalMultiset<_>>()
        );
        assert!(appended.is_empty());

        let mut extended_by_tuples = mset1.clone();
        extended_by_tuples.extend(mset2.iter());
        assert_eq!(extended_by_tuples, combined);

        let mut extended_by_values = mset1.clone();
        extended_by_values.extend(
            mset2
                .iter()
                .flat_map(|(val, mul)| std::iter::repeat_n(val, mul.get())),
        );
        assert_eq!(
            extended_by_values, combined,
            "{mset1:?} + {mset2:?} != {extended_by_values:?}"
        );

        assert_eq!(
            mset1.cmp(mset2),
            mset1
                .value_to_multiplicity
                .cmp(&mset2.value_to_multiplicity)
        );
        assert_eq!(mset1.partial_cmp(mset2), Some(mset1.cmp(mset2)));
    }

    /// Check properties that should be true of any multiset, knowing its contents
    fn check_any_mset(mset: &NumericalMultiset<i32>, contents: &[(i32, NonZeroUsize)]) {
        let sorted_contents = contents
            .iter()
            .map(|(v, m)| (*v, m.get()))
            .collect::<BTreeMap<i32, usize>>();

        check_equal_iterable(
            mset.iter().map(|(val, mul)| (val, mul.get())),
            sorted_contents.iter().map(|(&k, &v)| (k, v)),
        );
        check_equal_iterable(mset, mset.iter());
        check_equal_iterable(mset.clone(), mset.iter());
        check_equal_iterable(mset.range(..), mset.iter());
        check_equal_iterable(mset.values(), sorted_contents.keys().copied());
        check_equal_iterable(mset.clone().into_values(), mset.values());

        assert_eq!(mset.len(), sorted_contents.values().sum());
        assert_eq!(mset.num_values(), contents.len());
        assert_eq!(mset.is_empty(), contents.is_empty());

        for (&val, &mul) in &sorted_contents {
            assert!(mset.contains(val));
            assert_eq!(mset.multiplicity(val).unwrap().get(), mul);
        }

        assert_eq!(
            mset.first().map(|(val, mul)| (val, mul.get())),
            sorted_contents.first_key_value().map(|(&k, &v)| (k, v)),
        );
        assert_eq!(
            mset.last().map(|(val, mul)| (val, mul.get())),
            sorted_contents.last_key_value().map(|(&k, &v)| (k, v)),
        );

        #[allow(clippy::eq_op)]
        {
            assert_eq!(mset, mset);
        }
        assert_eq!(*mset, mset.clone());
        assert_eq!(mset.cmp(mset), Ordering::Equal);
        assert_eq!(mset.partial_cmp(mset), Some(mset.cmp(mset)));

        let state = RandomState::new();
        assert_eq!(
            state.hash_one(mset),
            state.hash_one(&mset.value_to_multiplicity),
        );

        let mut mutable = mset.clone();
        if let Some((first, first_mul)) = mset.first() {
            // Pop the smallest items...
            assert_eq!(mutable.pop_all_first(), Some((first, first_mul)));
            assert_eq!(mutable.len(), mset.len() - first_mul.get());
            assert_eq!(mutable.num_values(), mset.num_values() - 1);
            assert!(!mutable.contains(first));
            assert_eq!(mutable.multiplicity(first), None);
            assert_ne!(mutable, *mset);

            // ...then insert them back
            assert_eq!(mutable.insert_multiple(first, first_mul), None);
            assert_eq!(mutable, *mset);

            // Same with a single item
            assert_eq!(mutable.pop_first(), Some(first));
            assert_eq!(mutable.len(), mset.len() - 1);
            let new_first_mul = NonZeroUsize::new(first_mul.get() - 1);
            let first_is_single = new_first_mul.is_none();
            assert_eq!(
                mutable.num_values(),
                mset.num_values() - first_is_single as usize
            );
            assert_eq!(mutable.contains(first), !first_is_single);
            assert_eq!(mutable.multiplicity(first), new_first_mul);
            assert_ne!(mutable, *mset);
            assert_eq!(mutable.insert(first), new_first_mul);
            assert_eq!(mutable, *mset);

            // If there is a first item, there is a last item
            let (last, last_mul) = mset.last().unwrap();

            // And everything we checked for the smallest items should also
            // applies to the largest ones
            assert_eq!(mutable.pop_all_last(), Some((last, last_mul)));
            assert_eq!(mutable.len(), mset.len() - last_mul.get());
            assert_eq!(mutable.num_values(), mset.num_values() - 1);
            assert!(!mutable.contains(last));
            assert_eq!(mutable.multiplicity(last), None);
            assert_ne!(mutable, *mset);
            //
            assert_eq!(mutable.insert_multiple(last, last_mul), None);
            assert_eq!(mutable, *mset);
            //
            assert_eq!(mutable.pop_last(), Some(last));
            assert_eq!(mutable.len(), mset.len() - 1);
            let new_last_mul = NonZeroUsize::new(last_mul.get() - 1);
            let last_is_single = new_last_mul.is_none();
            assert_eq!(
                mutable.num_values(),
                mset.num_values() - last_is_single as usize
            );
            assert_eq!(mutable.contains(last), !last_is_single);
            assert_eq!(mutable.multiplicity(last), new_last_mul);
            assert_ne!(mutable, *mset);
            assert_eq!(mutable.insert(last), new_last_mul);
            assert_eq!(mutable, *mset);
        } else {
            assert!(mset.is_empty());
            assert_eq!(mutable.pop_first(), None);
            assert!(mutable.is_empty());
            assert_eq!(mutable.pop_all_first(), None);
            assert!(mutable.is_empty());
            assert_eq!(mutable.pop_last(), None);
            assert!(mutable.is_empty());
            assert_eq!(mutable.pop_all_last(), None);
            assert!(mutable.is_empty());
        }

        let mut retain_all = mset.clone();
        retain_all.retain(|_, _| true);
        assert_eq!(retain_all, *mset);

        let mut retain_nothing = mset.clone();
        retain_nothing.retain(|_, _| false);
        assert!(retain_nothing.is_empty());
    }

    /// Check properties that should be true of an empty multiset
    fn check_empty_mset(empty: &NumericalMultiset<i32>) {
        check_any_mset(empty, &[]);

        assert_eq!(empty.len(), 0);
        assert_eq!(empty.num_values(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);

        check_empty_iterable(empty.iter());
        check_empty_iterable(empty.values());
        check_empty_iterable(empty.clone());
        check_empty_iterable(empty.clone().into_values());

        let mut mutable = empty.clone();
        assert_eq!(mutable.pop_first(), None);
        assert_eq!(mutable.pop_last(), None);
        assert_eq!(mutable.pop_all_first(), None);
        assert_eq!(mutable.pop_all_last(), None);
    }

    /// Check that clear() makes a multiset empty
    fn check_clear_outcome(mut mset: NumericalMultiset<i32>) {
        mset.clear();
        check_empty_mset(&mset);
    }

    /// Check the various ways to build an empty multiset
    #[test]
    fn empty() {
        check_empty_mset(&NumericalMultiset::default());
        let mset = NumericalMultiset::<i32>::new();
        check_empty_mset(&mset);
        check_clear_outcome(mset);
    }

    /// Maximal acceptable multiplicity value
    fn max_multiplicity() -> usize {
        SizeRange::default().end_excl()
    }

    /// Generate a reasonably low multiplicity value
    fn multiplicity() -> impl Strategy<Value = NonZeroUsize> {
        prop_oneof![Just(1), Just(2), 3..max_multiplicity()]
            .prop_map(|m| NonZeroUsize::new(m).unwrap())
    }

    /// Build an arbitrary multiset
    fn mset_contents() -> impl Strategy<Value = Vec<(i32, NonZeroUsize)>> {
        any::<HashSet<i32>>().prop_flat_map(|values| {
            prop::collection::vec(multiplicity(), values.len()).prop_map(move |multiplicities| {
                values.iter().copied().zip(multiplicities).collect()
            })
        })
    }

    proptest! {
        /// Check properties of arbitrary multisets
        #[test]
        fn single(contents in mset_contents()) {
            for mset in [
                contents.iter().copied().collect(),
                contents.iter().flat_map(|(v, m)| {
                    std::iter::repeat_n(*v, m.get())
                }).collect(),
            ] {
                check_any_mset(&mset, &contents);
                check_any_mset_pair(&mset, &mset);
                let empty = NumericalMultiset::default();
                check_any_mset_pair(&mset, &empty);
                check_any_mset_pair(&empty, &mset);
                check_clear_outcome(mset);
            }
        }
    }

    /// Build an arbitrary multiset
    fn mset() -> impl Strategy<Value = NumericalMultiset<i32>> {
        mset_contents().prop_map(NumericalMultiset::from_iter)
    }

    /// Build a multiset and pick a value that has a high chance of being from
    /// the multiset if it is not empty.
    fn mset_and_value() -> impl Strategy<Value = (NumericalMultiset<i32>, i32)> {
        mset().prop_flat_map(|mset| {
            if mset.is_empty() {
                (Just(mset), any::<i32>()).boxed()
            } else {
                let inner_value = prop::sample::select(mset.values().collect::<Vec<_>>());
                let value = prop_oneof![inner_value, any::<i32>()];
                (Just(mset), value).boxed()
            }
        })
    }

    proptest! {
        /// Test operations that combine a multiset with a value
        #[test]
        fn with_value((initial, value) in mset_and_value()) {
            // Most operations depend on whether the value was present...
            if let Some(&mul) = initial.value_to_multiplicity.get(&value) {
                assert!(initial.contains(value));
                assert_eq!(initial.multiplicity(value), Some(mul));
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.insert(value), Some(mul));
                    let mut expected = initial.clone();
                    let Entry::Occupied(mut entry) = expected.value_to_multiplicity.entry(value) else {
                        unreachable!();
                    };
                    *entry.get_mut() = mul.checked_add(1).unwrap();
                    expected.len += 1;
                    assert_eq!(mset, expected);
                }
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.remove(value), Some(mul));
                    let mut expected = initial.clone();
                    if mul == ONE {
                        expected.value_to_multiplicity.remove(&value);
                    } else {
                        let Entry::Occupied(mut entry) = expected.value_to_multiplicity.entry(value) else {
                            unreachable!();
                        };
                        *entry.get_mut() = NonZeroUsize::new(mul.get() - 1).unwrap();
                    }
                    expected.len -= 1;
                    assert_eq!(mset, expected);
                }
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.remove_all(value), Some(mul));
                    let mut expected = initial.clone();
                    expected.value_to_multiplicity.remove(&value);
                    expected.len -= mul.get();
                    assert_eq!(mset, expected);
                }
            } else {
                assert!(!initial.contains(value));
                assert_eq!(initial.multiplicity(value), None);
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.insert(value), None);
                    let mut expected = initial.clone();
                    expected.value_to_multiplicity.insert(value, ONE);
                    expected.len += 1;
                    assert_eq!(mset, expected);
                }
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.remove(value), None);
                    assert_eq!(mset, initial);
                    assert_eq!(mset.remove_all(value), None);
                    assert_eq!(mset, initial);
                }
            }

            // ...except for split_off, which doesn't really care
            {
                let mut mset = initial.clone();
                let ge = mset.split_off(value);
                let lt = mset;
                let mut expected_lt = initial.clone();
                let ge_value_to_multiplicity = expected_lt.value_to_multiplicity.split_off(&value);
                expected_lt.reset_len();
                assert_eq!(lt, expected_lt);
                let expected_ge = NumericalMultiset::from_iter(ge_value_to_multiplicity);
                assert_eq!(ge, expected_ge);
            }
        }

        /// Check operations that require a value and a multiplicity
        #[test]
        fn with_value_and_multiplicity((initial, value) in mset_and_value(),
                                       new_mul in multiplicity()) {
            // Most operations depend on whether the value was present...
            if let Some(&initial_mul) = initial.value_to_multiplicity.get(&value) {
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.insert_multiple(value, new_mul), Some(initial_mul));
                    let mut expected = initial.clone();
                    let Entry::Occupied(mut entry) = expected.value_to_multiplicity.entry(value) else {
                        unreachable!();
                    };
                    *entry.get_mut() = initial_mul.checked_add(new_mul.get()).unwrap();
                    expected.len += new_mul.get();
                    assert_eq!(mset, expected);
                }
                {
                    let mut mset = initial.clone();
                    assert_eq!(mset.replace_all(value, new_mul), Some(initial_mul));
                    let mut expected = initial.clone();
                        let Entry::Occupied(mut entry) = expected.value_to_multiplicity.entry(value) else {
                            unreachable!();
                        };
                        *entry.get_mut() = new_mul;
                    expected.len = expected.len - initial_mul.get() + new_mul.get();
                    assert_eq!(mset, expected);
                }
            } else {
                let mut inserted = initial.clone();
                assert_eq!(inserted.insert_multiple(value, new_mul), None);
                let mut expected = initial.clone();
                expected.value_to_multiplicity.insert(value, new_mul);
                expected.len += new_mul.get();
                assert_eq!(inserted, expected);

                let mut replaced = initial.clone();
                assert_eq!(replaced.replace_all(value, new_mul), None);
                assert_eq!(replaced, expected);
            }

            // ...but retain doesn't care much
            {
                let f = |v, m: &mut NonZeroUsize| {
                    if v <= value && *m <= new_mul {
                        *m = m.checked_add(42).unwrap();
                        true
                    } else {
                        false
                    }
                };
                let mut retained = initial.clone();
                retained.retain(f);
                let mut expected = initial.clone();
                expected.value_to_multiplicity.retain(|&v, m| f(v, m));
                expected.reset_len();
                assert_eq!(retained, expected);
            }
        }
    }

    /// Build a multiset and pick a range of values that have a high chance of
    /// being from the multiset if it is not empty, and of being in sorted order
    fn mset_and_value_range() -> impl Strategy<Value = (NumericalMultiset<i32>, Range<i32>)> {
        let pair_to_range = |values: [i32; 2]| values[0]..values[1];
        mset().prop_flat_map(move |mset| {
            if mset.is_empty() {
                (Just(mset), any::<[i32; 2]>().prop_map(pair_to_range)).boxed()
            } else {
                let inner_value = || prop::sample::select(mset.values().collect::<Vec<_>>());
                let value = || prop_oneof![3 => inner_value(), 2 => any::<i32>()];
                let range = [value(), value()].prop_map(pair_to_range);
                (Just(mset), range).boxed()
            }
        })
    }

    proptest! {
        #[test]
        fn range((mset, range) in mset_and_value_range()) {
            match std::panic::catch_unwind(|| {
                mset.range(range.clone()).collect::<Vec<_>>()
            }) {
                Ok(output) => check_equal_iterable(output, mset.value_to_multiplicity.range(range).map(|(&v, &m)| (v, m))),
                Err(_panicked) => assert!(range.start > range.end),
            }
        }
    }

    /// Build a pair of multisets that have reasonable odds of having some
    /// simple set relationship with each other.
    fn mset_pair() -> impl Strategy<Value = (NumericalMultiset<i32>, NumericalMultiset<i32>)> {
        mset().prop_flat_map(|mset1| {
            if mset1.is_empty() {
                (Just(mset1), mset()).boxed()
            } else {
                // For related sets, we first extract a subsequence of the
                // (value, multiplicity) pairs contained inside mset1...
                let related = prop::sample::subsequence(
                    mset1.iter().collect::<Vec<_>>(),
                    0..mset1.num_values(),
                )
                .prop_flat_map(move |subseq| {
                    // ...then, for each retained (value, multiplicity) pairs...
                    subseq
                        .into_iter()
                        .map(|(v, m)| {
                            let m = m.get();
                            // ...we pick a multiplicity that has equal chance of being...
                            // - 1: Common gotcha in tests
                            // - 2..M: Less than in mset1
                            // - M: As many as in mset1
                            // - (M+1)..: More than in mset1
                            let multiplicity = match m {
                                1 => prop_oneof![Just(1), 2..max_multiplicity()].boxed(),
                                2 => prop_oneof![Just(1), Just(2), 3..max_multiplicity()].boxed(),
                                _ if m + 1 < max_multiplicity() => {
                                    prop_oneof![Just(1), 2..m, Just(m), (m + 1)..max_multiplicity()]
                                        .boxed()
                                }
                                _ => prop_oneof![Just(1), 2..max_multiplicity()].boxed(),
                            }
                            .prop_map(|m| NonZeroUsize::new(m).unwrap());
                            (Just(v), multiplicity)
                        })
                        .collect::<Vec<_>>()
                })
                .prop_map(|elems| elems.into_iter().collect());

                // As a result, mset2 convers less values than mset1, so their
                // roles are asymmetrical. To ensure this bias isn't exposed to
                // tests, we should randomly flip them.
                let related_pair = (Just(mset1.clone()), related, any::<bool>()).prop_map(
                    |(mset1, mset2, flip)| {
                        if flip { (mset2, mset1) } else { (mset1, mset2) }
                    },
                );

                // Finally, we can and should also sometimes pick unrelated sets
                // like we do when mset1 is empty
                prop_oneof![
                    1 => (Just(mset1), mset()),
                    4 => related_pair,
                ]
                .boxed()
            }
        })
    }

    proptest! {
        /// Check properties of arbitrary pairs of multisets
        #[test]
        fn pair((mset1, mset2) in mset_pair()) {
            check_any_mset_pair(&mset1, &mset2);
        }
    }
}
