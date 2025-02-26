//! An ordered multiset implementation for primitive number types based on
//! sparse histograms.
//!
//! This crate implements a kind of multiset, which is a generalization of the
//! notion of mathematical set where multiple values that are equal to each
//! other can be present simulatneously.
//!
//! Our multiset implementation differs from the most popular multiset
//! implementations of crates.io at the time of its release in the following
//! respects:
//!
//! - It is ordered, which means that order-based queries such as getting a
//!   sorted list of elements or finding the minimum and maximum elements are
//!   relatively cheap. For example, the complexity of a min/max query grows
//!   logarithmically with the number of distinct values in the multiset.
//!     * The price to pay for this feature is that classic set operations like
//!       inserting/removing elements or querying whether an element is present
//!       will scale less well to larger datasets than in the more common
//!       hash-based multiset implementations: element-wise operations also have
//!       logarithmic complexity, whereas a hash-based multiset can instead
//!       achieve a constant-time operation complexity that's independent of the
//!       collection size.
//! - It is specialized for primitive number types and `repr(transparent)`
//!   wrappers thereof, which means that it can leverage the property of these
//!   numbers to improve ergonomics and efficiency:
//!     * Since all primitive number types are Copy, we do not need to bother
//!       with references and [`Borrow`](std::borrow::Borrow) trait complexity
//!       like general-purpose map and set implementations do, and can instead
//!       provide a simpler value-based API.
//!     * Since all primitive number types have well-behaved [`Eq`]
//!       implementations where numbers that compare equal are identical, we do
//!       not need to track lists of equal entries like many multiset
//!       implementations do, and can instead use a more efficient sparse
//!       histogramming approach where we simply count the number of equal
//!       entries.
//!
//! One example application of such a multiset implementation is median
//! filtering of streaming numerical data for which the classic dense histogram
//! approach is not applicable, such as floats and integers of width >= 32 bits.
//!
//! To use this crate with floating-point data, you will need to use one of the
//! available [`Ord`] float wrapper that assert absence of NaNs, such as the
//! [`NotNan`](https://docs.rs/ordered-float/latest/ordered_float/struct.NotNan.html)
//! type from the [`ordered_float`](https://docs.rs/ordered-float) crate. We do
//! not handle this concern for you because initially checking for NaN has a
//! cost and we believe this cost is best paid once on your side and hopefully
//! amortized across many reuses of the resulting dataset, rather than
//! repeatedly paid every time an element is inserted into a
//! [`NumericalMultiset`].

use std::{
    cmp::Ordering,
    collections::btree_map::{self, BTreeMap, Entry},
    hash::Hash,
    iter::FusedIterator,
    num::NonZeroUsize,
    ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub},
};

/// An ordered multiset implementation for primitive number types based on
/// sparse histograms.
///
/// You can learn more about the design rationale and overall capabilities of
/// this data structure in the [crate-level documentation](index.html).
///
/// At the time of writing, this data structure is based on the standard
/// library's [`BTreeMap`], and many points of the [`BTreeMap`] documentation
/// also apply to it. In particular, it is a logic error to modify the order of
/// values stored inside of the multiset using internal mutability tricks.
///
/// In all the following documentation, we will use the following terminology:
///
/// - "values" refers to a unique value as defined by equality of the
///   [`Eq`] implementation of type `T`
/// - "elements" refers to possibly duplicate occurences of a value within the
///   multiset.
/// - "multiplicity" refers to the number of occurences of a value within the
///   multiset, i.e. the number of elements that are equal to this value.
///
/// # Examples
///
/// ```
/// use numerical_multiset::NumericalMultiset;
/// use std::num::NonZeroUsize;
///
/// // Create a multiset
/// let mut set = NumericalMultiset::new();
///
/// // Inserting elements that do not exist yet is handled much like a standard
/// // library set type, except we return an Option instead of a boolean...
/// assert!(set.insert(123).is_none());
/// assert!(set.insert(456).is_none());
///
/// // ...which allows us to report the number of pre-existing elements, if any
/// assert_eq!(set.insert(123), NonZeroUsize::new(1));
///
/// // It is possible to query the minimal and maximal elements cheaply, along
/// // with their multiplicity within the multiset.
/// let nonzero = |x| NonZeroUsize::new(x).unwrap();
/// assert_eq!(set.first(), Some((123, nonzero(2))));
/// assert_eq!(set.last(), Some((456, nonzero(1))));
///
/// // ...and it is more generally possible to iterate over elements in order,
/// // from the smallest to the largest:
/// for (elem, multiplicity) in &set {
///     println!("{elem} with multiplicity {multiplicity}");
/// }
/// ```
#[derive(Clone, Debug, Default, Eq)]
pub struct NumericalMultiset<T> {
    value_to_multiplicity: BTreeMap<T, NonZeroUsize>,
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
    /// let mut set: NumericalMultiset<i32> = NumericalMultiset::new();
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn new() -> Self {
        Self {
            value_to_multiplicity: BTreeMap::new(),
            len: 0,
        }
    }

    /// Clears the multiset, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let mut v = NumericalMultiset::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.value_to_multiplicity.clear();
        self.len = 0;
    }

    /// Truth that the multiset contains no elements
    #[must_use = "Only effect is to produce a result"]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of elements currently present in the multiset, including
    /// duplicate occurences of a value.
    ///
    /// See also [`num_values()`](Self::num_values) for a count of unique
    /// values, ignoring duplicate elements.
    #[must_use = "Only effect is to produce a result"]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of distinct values currently present in the multiset
    ///
    /// See also [`len()`](Self::len) if you want to know the total number of
    /// multiset elements, including duplicates of the same value.
    #[must_use = "Only effect is to produce a result"]
    pub fn num_values(&self) -> usize {
        self.value_to_multiplicity.len()
    }

    /// Creates a consuming iterator visiting all the distinct values, in sorted
    /// order. The multiset cannot be used after calling this method.
    #[must_use = "Only effect is to produce a result"]
    pub fn into_values(
        self,
    ) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator + FusedIterator {
        self.value_to_multiplicity.into_keys()
    }

    /// Update `self.len` to match `self.value_to_multiplicity` contents
    ///
    /// This expensive `O(N)` operation should only be performed after calling
    /// `BTreeMap` operations that do not provide the right hooks to update the
    /// length field more efficiently.
    fn reset_len(&mut self) {
        self.len = self.value_to_multiplicity.values().map(|x| x.get()).sum();
    }
}

impl<T: Copy> NumericalMultiset<T> {
    /// Iterator over all distinct values in the multiset, without their
    /// multiplicities. Output is sorted by ascending value.
    // TODO: Advertise iter()
    #[must_use = "Only effect is to produce a result"]
    pub fn values(
        &self,
    ) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator + FusedIterator + Clone {
        self.value_to_multiplicity.keys().copied()
    }

    /// Iterator over all distinct values in the multiset, along with their
    /// multiplicities. Output is sorted by ascending value.
    // TODO: Advertise values()
    #[must_use = "Only effect is to produce a result"]
    pub fn iter(&self) -> Iter<'_, T> {
        self.into_iter()
    }
}

impl<T: Ord> NumericalMultiset<T> {
    /// Returns `true` if the multiset contains at least one occurence of a
    /// value.
    ///
    /// See also [`multiplicity()`](Self::multiplicity) if you need to know
    /// precisely how many occurences of a value are present inside of the
    /// multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let set = NumericalMultiset::from_iter([1, 2, 2]);
    ///
    /// assert_eq!(set.contains(1), true);
    /// assert_eq!(set.contains(2), true);
    /// assert_eq!(set.contains(3), false);
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn contains(&self, value: T) -> bool {
        self.value_to_multiplicity.contains_key(&value)
    }

    /// Returns the number of occurences of a value inside of the multiset, or
    /// `None` if it is not present.
    ///
    /// See also [`contains()`](Self::contains) if you only need to know whether
    /// at least one occurence of value is present inside of the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let set = NumericalMultiset::from_iter([1, 2, 2]);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(set.multiplicity(1), Some(nonzero(1)));
    /// assert_eq!(set.multiplicity(2), Some(nonzero(2)));
    /// assert_eq!(set.multiplicity(3), None);
    /// ```
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn multiplicity(&self, value: T) -> Option<NonZeroUsize> {
        self.value_to_multiplicity.get(&value).copied()
    }

    /// Returns `true` if `self` has no elements in common with `other`. This is
    /// logically equivalent to checking for an empty intersection, but more
    /// efficient.
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

                // Once one iterator ends, we know there's no common element
                // left, so we can conclude that the multisets are disjoint.
                (Some(_), None) | (None, Some(_)) | (None, None) => return true,
            }
        }
    }

    /// Returns `true` if the set is a subset of another, i.e., `other` contains
    /// at least all the elements in `self`.
    ///
    /// In a multiset context, this means that if `self` contains N occurences
    /// of a certain value, then `other` must contain N or more occurences of
    /// that value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let sup = NumericalMultiset::from_iter([1, 2, 2]);
    /// let mut set = NumericalMultiset::new();
    ///
    /// assert!(set.is_subset(&sup));
    /// set.insert(2);
    /// assert!(set.is_subset(&sup));
    /// set.insert(2);
    /// assert!(set.is_subset(&sup));
    /// set.insert(2);
    /// assert!(!set.is_subset(&sup));
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
                            // contain at least the same amount of copies of
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

    /// Returns `true` if the set is a superset of another, i.e., `self`
    /// contains at least all the elements in `other`.
    ///
    /// In a multiset context, this means that if `other` contains N copies of a
    /// certain value, then `self` must contain N or more copies of that value.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    ///
    /// let sub = NumericalMultiset::from_iter([1, 2, 2]);
    /// let mut set = NumericalMultiset::new();
    ///
    /// assert!(!set.is_superset(&sub));
    ///
    /// set.insert(3);
    /// set.insert(1);
    /// assert!(!set.is_superset(&sub));
    ///
    /// set.insert(2);
    /// assert!(!set.is_superset(&sub));
    ///
    /// set.insert(2);
    /// assert!(set.is_superset(&sub));
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    // TODO: Add pop_first() based on BTreeMap::first_entry() + advertise pop_all_first()

    /// Remove all copies of the smallest value from the multiset, if any
    // TODO: Advertise pop_first()
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_all_first(&mut self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .pop_first()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    // TODO: Add pop_last() based on BTreeMap::last_entry() + advertise pop_all_last()

    /// Remove all copies of the largest value from the multiset, if any
    // TODO: Advertise pop_last()
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn pop_all_last(&mut self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .pop_last()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Insert a copy of a value, tell how many identical elements were already
    /// present in the multiset before insertion.
    // TODO: Advertise insert_multiple()
    #[inline]
    pub fn insert(&mut self, value: T) -> Option<NonZeroUsize> {
        self.insert_multiple(value, NonZeroUsize::new(1).unwrap())
    }

    /// Insert multiple copies of a value, tell how many identical elements were
    /// already present in the multiset before insertion.
    // TODO: Advertise insert() as a convenience shortcut
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

    /// Add multiple copies of a value, replacing all copies of this value that
    /// were previously present in the multiset. Tell how many copies of `value`
    /// were previously present in the multiset.
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

    /// Attempt to remove one occurence of a value from the multiset, on success
    /// tell how many identical elements were previously present in the multiset.
    // TODO: Add remove_multiple() and advertise that + remove_all()
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn remove(&mut self, value: T) -> Option<NonZeroUsize> {
        let result = match self.value_to_multiplicity.entry(value) {
            Entry::Vacant(_) => None,
            Entry::Occupied(mut o) => {
                let old_multiplicity = *o.get();
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
        };
        self.len -= 1;
        result
    }

    /// Attempt to remove all occurences of a value from the multiset, on
    /// success tell how many identical elements were removed from the multiset.
    // TODO: Advertise remove()
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn remove_all(&mut self, value: T) -> Option<NonZeroUsize> {
        let result = self.value_to_multiplicity.remove(&value);
        self.len -= result.map_or(0, |nz| nz.get());
        result
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        // FIXME: This is wrong, must sum identical entries
        self.value_to_multiplicity
            .append(&mut other.value_to_multiplicity);
        self.reset_len();
        other.len = 0;
    }

    /// Splits the collection into two at the value. Returns a new collection
    /// with all elements greater than or equal to the value.
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
    /// Minimal value present in the multiset, if any, along with its element
    /// multiplicity.
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn first(&self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .first_key_value()
            .map(|(&k, &v)| (k, v))
    }

    /// Maximal value present in the multiset, if any, along with its element
    /// multiplicity.
    #[inline]
    #[must_use = "Only effect is to produce a result"]
    pub fn last(&self) -> Option<(T, NonZeroUsize)> {
        self.value_to_multiplicity
            .last_key_value()
            .map(|(&k, &v)| (k, v))
    }

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
    /// Panics if range `start > end`. Panics if range `start == end` and both
    /// bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut set = NumericalMultiset::new();
    /// set.insert(3);
    /// set.insert(3);
    /// set.insert(5);
    /// set.insert(8);
    /// set.insert(8);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     set.range(4..).collect::<Vec<_>>(),
    ///     [(5, nonzero(1)), (8, nonzero(2))]
    /// );
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

    /// Visits the elements representing the difference, i.e., those that are in
    /// `self` but not in `other`. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The difference is computed element-wise, not value-wise, so if both
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
    /// let mut a = NumericalMultiset::new();
    /// a.insert(1);
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = NumericalMultiset::new();
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     a.difference(&b).collect::<Vec<_>>(),
    ///     [(1, nonzero(2)), (2, nonzero(1))]
    /// );
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

                    // Other iterator has ended, can yield all remaining values
                    None => return Some((value, multiplicity)),
                }
            }
        })
    }

    /// Visits the elements representing the symmetric difference, i.e., those
    /// that are in `self` or in `other` but not in both. They are sorted in
    /// ascending value order and emitted in the usual deduplicated `(value,
    /// multiplicity)` format.
    ///
    /// The symmetric difference is computed element-wise, not value-wise, so if
    /// both `self` and `other` contain occurences of a certain value `v` with
    /// respective multiplicities `s` and `o`, then...
    ///
    /// - If `self` contains as many occurences of `v` as `other` (i.e. `s ==
    ///   o`), then the symmetric difference will not contain any occurence of
    ///   `v`.
    /// - Otherwise (if `s != o` the symmetric difference will contain
    ///   `s.abs_diff(o)` occurences of `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numerical_multiset::NumericalMultiset;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut a = NumericalMultiset::new();
    /// a.insert(1);
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = NumericalMultiset::new();
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     a.symmetric_difference(&b).collect::<Vec<_>>(),
    ///     [(1, nonzero(2)), (2, nonzero(1)), (4, nonzero(1))]
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || 'joint_iter: loop {
            match (iter1.peek(), iter2.peek()) {
                // As long as both iterators yield values, must be careful to
                // yield values from both iterators, in the right order, and to
                // skip common values.
                (Some((value1, multiplicity1)), Some((value2, multiplicity2))) => {
                    match value1.cmp(value2) {
                        // Return the smallest element, if any, advancing the
                        // corresponding iterator along the way
                        Ordering::Less => return iter1.next(),
                        Ordering::Greater => return iter2.next(),

                        // Same value was yielded from both iterators
                        Ordering::Equal => {
                            // If the value was yielded with different
                            // multiplicities, then we must still yield an entry
                            // with a multiplicity that is the absolute
                            // difference of these multiplicities.
                            if multiplicity1 != multiplicity2 {
                                let value12 = *value1;
                                let difference_multiplicity = NonZeroUsize::new(
                                    multiplicity1.get().abs_diff(multiplicity2.get()),
                                )
                                .expect("Checked above that this is fine");
                                let _ = (iter1.next(), iter2.next());
                                return Some((value12, difference_multiplicity));
                            } else {
                                // Otherwise ignore the common value, advance
                                // both iterators and try again
                                let _ = (iter1.next(), iter2.next());
                                continue 'joint_iter;
                            }
                        }
                    }
                }

                // One one iterator ends, we know there's no common value left
                // and there is no sorted sequence merging business to care
                // about, so we can just yield the remainder as-is.
                (Some(_), None) => return iter1.next(),
                (None, Some(_)) => return iter2.next(),
                (None, None) => return None,
            }
        })
    }

    /// Visits the elements representing the intersection, i.e., those that are
    /// both in `self` and `other`. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The intersection is computed element-wise, not value-wise, so if both
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
    /// let mut a = NumericalMultiset::new();
    /// a.insert(1);
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = NumericalMultiset::new();
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     a.intersection(&b).collect::<Vec<_>>(),
    ///     [(2, nonzero(1)), (3, nonzero(1))]
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn intersection<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || 'joint_iter: loop {
            match (iter1.peek(), iter2.peek()) {
                // As long as both iterators yield elements, must be careful to
                // yield common elements with merged multiplicities
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

                        // Merge entries associated with a common value
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
        })
    }

    /// Visits the elements representing the union, i.e., those that are in
    /// either `self` or `other`, without counting values that are present in
    /// both multisets twice. They are sorted in ascending value order and
    /// emitted in the usual deduplicated `(value, multiplicity)` format.
    ///
    /// The union is computed element-wise, not value-wise, so if both
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
    /// let mut a = NumericalMultiset::new();
    /// a.insert(1);
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = NumericalMultiset::new();
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let nonzero = |x| NonZeroUsize::new(x).unwrap();
    /// assert_eq!(
    ///     a.union(&b).collect::<Vec<_>>(),
    ///     [(1, nonzero(2)), (2, nonzero(2)), (3, nonzero(1)), (4, nonzero(1))]
    /// );
    /// ```
    #[must_use = "Only effect is to produce a result"]
    pub fn union<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (T, NonZeroUsize)> + Clone + 'a {
        let mut iter1 = self.iter().peekable();
        let mut iter2 = other.iter().peekable();
        std::iter::from_fn(move || match (iter1.peek(), iter2.peek()) {
            // As long as both iterators yield elements, must be careful to
            // yield elements in the right order and merge common multiplicities
            (Some((value1, multiplicity1)), Some((value2, multiplicity2))) => {
                match value1.cmp(value2) {
                    // Yield non-common elements in the right order
                    Ordering::Less => iter1.next(),
                    Ordering::Greater => iter2.next(),

                    // Merge entries associated with a common value
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

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all values `v` with multiplicity `m` for which
    /// `f(v, m)` returns `false`. The values are visited in ascending order.
    pub fn retain(&mut self, mut f: impl FnMut(T, NonZeroUsize) -> bool) {
        self.value_to_multiplicity.retain(|&k, &mut v| f(k, v));
        self.reset_len();
    }
}

impl<T: Copy + Ord> BitAnd<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the intersection of `self` and `rhs` as a new `NumericalMultiset<T>`.
    #[must_use = "Only effect is to produce a result"]
    fn bitand(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.intersection(rhs).collect()
    }
}

impl<T: Copy + Ord> BitOr<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the union of `self` and `rhs` as a new `NumericalMultiset<T>`.
    #[must_use = "Only effect is to produce a result"]
    fn bitor(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.union(rhs).collect()
    }
}

impl<T: Copy + Ord> BitXor<&NumericalMultiset<T>> for &NumericalMultiset<T> {
    type Output = NumericalMultiset<T>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `NumericalMultiset<T>`.
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
/// An iterator over the entries of an [`NumericalMultiset`], sorted by value.
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

    /// Gets an iterator for moving out the `NumericalMultiset`’s contents in
    /// ascending order.
    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.value_to_multiplicity.into_iter())
    }
}
//
/// An owning iterator over the entries of an [`NumericalMultiset`], sorted by
/// value.
///
/// This struct is created by the [`into_iter`](IntoIterator::into_iter) method
/// on [`NumericalMultiset`] (provided by the [`IntoIterator`] trait). See its
/// documentation for more.
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
    #[must_use = "Only effect is to produce a result"]
    fn sub(self, rhs: &NumericalMultiset<T>) -> Self::Output {
        self.difference(rhs).collect()
    }
}
