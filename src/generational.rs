use super::bit_vector::BlockedBitVec;
use super::generational_builder::{
    GenerationalBuilderWithBits, GenerationalBuilderWithFalsePositiveRate,
};
use super::hasher::DefaultHasher;
use super::sparse_hash::{self, SparseHash};
use crate::{block_index, get_orginal_hashes};
use std::hash::{BuildHasher, Hash, Hasher};
use wide::{u64x2, u64x4};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenerationalBloomFilter<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    bits: BlockedBitVec<BLOCK_SIZE_BITS>,
    /// the total number of generations to track
    generations: u64,
    /// number of entries per generation
    generation_size: u64,
    /// stored size
    total_stored: u64,
    /// The total target hashes per item that is specified by user or optimized to maximize accuracy
    target_hashes: u64,
    /// The target number of bits to set/check per u64 per item when inserting/checking an item.
    num_rounds: Option<u64>,
    /// The number of hashes per item in addition to `num_rounds`. These hashes can be applied across many `u64`s in a block.
    /// These hashes are in addition to `num_rounds` to make up for rounding errors.
    num_hashes: u64,
    hasher: S,
}

impl GenerationalBloomFilter {
    fn new_builder<const BLOCK_SIZE_BITS: usize>(
        num_bits: usize,
    ) -> GenerationalBuilderWithBits<BLOCK_SIZE_BITS> {
        assert!(num_bits > 0);
        // Only available in rust 1.73+
        // let num_u64s = num_bits.div_ceil(64);
        let num_u64s = (num_bits + 64 - 1) / 64;
        GenerationalBuilderWithBits::<BLOCK_SIZE_BITS> {
            data: vec![0; num_u64s],
            hasher: Default::default(),
        }
    }

    fn new_from_vec<const BLOCK_SIZE_BITS: usize>(
        vec: Vec<u64>,
    ) -> GenerationalBuilderWithBits<BLOCK_SIZE_BITS> {
        assert!(!vec.is_empty());
        GenerationalBuilderWithBits::<BLOCK_SIZE_BITS> {
            data: vec,
            hasher: Default::default(),
        }
    }

    fn new_with_false_pos<const BLOCK_SIZE_BITS: usize>(
        fp: f64,
    ) -> GenerationalBuilderWithFalsePositiveRate<BLOCK_SIZE_BITS> {
        assert!(fp > 0.0);
        GenerationalBuilderWithFalsePositiveRate::<BLOCK_SIZE_BITS> {
            desired_fp_rate: fp,
            hasher: Default::default(),
        }
    }

    /// Creates a new instance of [`BuilderWithFalsePositiveRate`] to construct a `GenerationalBloomFilter` with a target false positive rate of `fp`.
    /// The memory size of the underlying bit vector is dependent on the false positive rate and the expected number of items.
    /// # Panics
    /// Panics if the false positive rate, `fp`, is 0.
    ///
    /// # Examples
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    /// let bloom = GenerationalBloomFilter::with_false_pos(0.001).expected_items(1000);
    /// ```
    pub fn with_false_pos(fp: f64) -> GenerationalBuilderWithFalsePositiveRate<512> {
        GenerationalBloomFilter::new_with_false_pos::<512>(fp)
    }

    /// Creates a new instance of [`BuilderWithBits`] to construct a `GenerationalBloomFilter` with `num_bits` number of bits for tracking item membership.
    /// # Panics
    /// Panics if the number of bits, `num_bits`, is 0.
    ///
    /// # Examples
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    /// let bloom = GenerationalBloomFilter::with_num_bits(1024).hashes(4);
    /// ```
    pub fn with_num_bits(num_bits: usize) -> GenerationalBuilderWithBits<512> {
        GenerationalBloomFilter::new_builder::<512>(num_bits)
    }

    /// Creates a new instance of [`BuilderWithBits`] to construct a `GenerationalBloomFilter` initialized with bit vector `bit_vec`.
    ///
    /// To fit the bit block size, `bit_vec` will be padded with `0u64`s and the end.
    ///
    /// # Panics
    /// Panics if the bit vector, `bit_vec`, is empty.
    /// # Examples
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    ///
    /// let orig = GenerationalBloomFilter::with_false_pos(0.001).seed(&42).items([1, 2]);
    /// let num_hashes = orig.num_hashes();
    /// let new = GenerationalBloomFilter::from_vec(orig.as_slice().to_vec()).seed(&42).hashes(num_hashes);
    ///
    /// assert!(new.contains(&1));
    /// assert!(new.contains(&2));
    /// ```
    pub fn from_vec(bit_vec: Vec<u64>) -> GenerationalBuilderWithBits<512> {
        GenerationalBloomFilter::new_from_vec::<512>(bit_vec)
    }
}

const fn validate_block_size(size: usize) -> usize {
    match size {
        64 | 128 | 256 | 512 => size,
        _ => panic!("The only BLOCK_SIZE's allowed are 64, 128, 256, and 512."),
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> GenerationalBloomFilter<BLOCK_SIZE_BITS, S> {
    /// Used to grab the last N bits from a hash.
    const BIT_INDEX_MASK: u64 = (validate_block_size(BLOCK_SIZE_BITS) - 1) as u64;

    /// The optimal number of hashes to perform for an item given the expected number of items to be contained in one block.
    /// Proof under "False Positives Analysis": <https://brilliant.org/wiki/bloom-filter/>
    #[inline]
    fn optimal_hashes_f(items_per_block: f64) -> f64 {
        let block_size = BLOCK_SIZE_BITS as f64;

        // `items_per_block` is an average. When block sizes decrease
        // the variance in the actual item per block increase,
        // meaning we are more likely to have a "crowded" block, with
        // way too many bits set. So we decrease the max hashes
        // to decrease this "crowding" effect.
        let min_hashes_mult = (BLOCK_SIZE_BITS as f64) / (512f64);

        let max_hashes = block_size / 64.0f64 * sparse_hash::hashes_for_bits(32) * min_hashes_mult;
        let hashes_per_block = block_size / items_per_block * f64::ln(2.0f64);
        if hashes_per_block > max_hashes {
            max_hashes
        } else if hashes_per_block < 1.0 {
            1.0
        } else {
            hashes_per_block
        }
    }

    #[inline]
    fn bit_index(hash1: &mut u64, hash2: u64) -> usize {
        let h = u64::next_hash(hash1, hash2);
        (h & Self::BIT_INDEX_MASK) as usize
    }

    /// Inserts an element into the Bloom filter.
    ///
    /// # Returns
    ///
    /// `true` if the item may have been previously in the Bloom filter (indicating a potential false positive),
    /// `false` otherwise.
    ///
    /// # Examples
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    ///
    /// let mut bloom = GenerationalBloomFilter::with_num_bits(1024).hashes(4);
    /// bloom.insert(&2);
    /// assert!(bloom.contains(&2));
    /// ```
    #[inline]
    pub fn insert(&mut self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, h2] = get_orginal_hashes(&self.hasher, val);
        let mut previously_contained = true;
        for _ in 0..self.num_hashes {
            // Set bits the traditional way--1 bit per composed hash
            let index = block_index(self.num_blocks(), h1);
            let block = &mut self.bits.get_block_mut(index);
            previously_contained &= BlockedBitVec::<BLOCK_SIZE_BITS>::set_for_block(
                block,
                Self::bit_index(&mut h1, h2),
            );
        }
        if let Some(num_rounds) = self.num_rounds {
            // Set many bits in parallel using a sparse hash
            let index = block_index(self.num_blocks(), h1);
            match BLOCK_SIZE_BITS {
                128 => {
                    let mut hashes_1 = u64x2::h1(&mut h1, h2);
                    let hashes_2 = u64x2::h2(h2);
                    let data = u64x2::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                    previously_contained &= u64x2::matches(self.bits.get_block(index), data);
                    u64x2::set(self.bits.get_block_mut(index), data);
                }
                256 => {
                    let mut hashes_1 = u64x4::h1(&mut h1, h2);
                    let hashes_2 = u64x4::h2(h2);
                    let data = u64x4::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                    previously_contained &= u64x4::matches(self.bits.get_block(index), data);
                    u64x4::set(self.bits.get_block_mut(index), data);
                }
                512 => {
                    let hashes_2 = u64x4::h2(h2);
                    let mut hashes_1 = u64x4::h1(&mut h1, h2);
                    for i in 0..2 {
                        let data = u64x4::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                        previously_contained &=
                            u64x4::matches(&self.bits.get_block(index)[4 * i..], data);
                        u64x4::set(&mut self.bits.get_block_mut(index)[4 * i..], data);
                    }
                }
                _ => {
                    for i in 0..self.bits.get_block(index).len() {
                        let data = u64::sparse_hash(&mut h1, h2, num_rounds);
                        let block = &mut self.bits.get_block_mut(index);
                        previously_contained &= (block[i] & data) == data;
                        block[i] |= data;
                    }
                }
            }
        }
        previously_contained
    }

    /// Checks if an element is possibly in the Bloom filter.
    ///
    /// # Returns
    ///
    /// `true` if the item is possibly in the Bloom filter, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    ///
    /// let bloom = GenerationalBloomFilter::with_num_bits(1024).items([1, 2, 3]);
    /// assert!(bloom.contains(&1));
    /// ```
    #[inline]
    pub fn contains(&self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, h2] = get_orginal_hashes(&self.hasher, val);
        (0..self.num_hashes).all(|_| {
            // Set bits the traditional way--1 bit per composed hash
            let index = block_index(self.num_blocks(), h1);
            let block = &self.bits.get_block(index);
            BlockedBitVec::<BLOCK_SIZE_BITS>::check_for_block(block, Self::bit_index(&mut h1, h2))
        }) && (if let Some(num_rounds) = self.num_rounds {
            // Set many bits in parallel using a sparse hash
            let index = block_index(self.num_blocks(), h1);
            let block = &self.bits.get_block(index);
            match BLOCK_SIZE_BITS {
                128 => {
                    let mut hashes_1 = u64x2::h1(&mut h1, h2);
                    let hashes_2 = u64x2::h2(h2);
                    let data = u64x2::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                    u64x2::matches(block, data)
                }
                256 => {
                    let mut hashes_1 = u64x4::h1(&mut h1, h2);
                    let hashes_2 = u64x4::h2(h2);
                    let data = u64x4::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                    u64x4::matches(block, data)
                }
                512 => {
                    let mut hashes_1 = u64x4::h1(&mut h1, h2);
                    let hashes_2 = u64x4::h2(h2);
                    (0..2).all(|i| {
                        let data = u64x4::sparse_hash(&mut hashes_1, hashes_2, num_rounds);
                        u64x4::matches(&block[4 * i..], data)
                    })
                }
                _ => (0..block.len()).all(|i| {
                    let data = u64::sparse_hash(&mut h1, h2, num_rounds);
                    (block[i] & data) == data
                }),
            }
        } else {
            true
        })
    }

    /// Returns the number of hashes per item.
    #[inline]
    pub fn num_hashes(&self) -> u32 {
        self.target_hashes as u32
    }

    /// Returns the total number of in-memory bits supporting the Bloom filter.
    pub fn generation_bits(&self) -> usize {
        self.generation_blocks() * BLOCK_SIZE_BITS
    }

    /// Returns the total number of in-memory blocks supporting the Bloom filter.
    /// Each block is `BLOCK_SIZE_BITS` bits.
    pub fn generation_blocks(&self) -> usize {
        self.bits.num_blocks() / self.generations
    }

    /// Returns a `u64` slice of this `GenerationalBloom`’s contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use generational_bloom::GenerationalBloomFilter;
    ///
    /// let data = vec![0x517cc1b727220a95; 8];
    /// let bloom = GenerationalBloomFilter::<512>::from_vec(data.clone()).hashes(4);
    /// assert_eq!(bloom.as_slice().to_vec(), data);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        self.bits.as_slice()
    }

    /// Clear all of the bits in the Bloom filter, removing all items.
    #[inline]
    pub fn clear(&mut self) {
        self.bits.clear();
    }
}

impl<T, const BLOCK_SIZE_BITS: usize, S: BuildHasher> Extend<T>
    for GenerationalBloomFilter<BLOCK_SIZE_BITS, S>
where
    T: Hash,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for val in iter {
            self.insert(&val);
        }
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> PartialEq
    for GenerationalBloomFilter<BLOCK_SIZE_BITS, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
            && self.num_hashes == other.num_hashes
            && self.num_rounds == other.num_rounds
    }
}
impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> Eq
    for GenerationalBloomFilter<BLOCK_SIZE_BITS, S>
{
}
