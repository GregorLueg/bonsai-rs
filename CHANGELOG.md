# News

## 0.0.1

**Features**

- Flat tree arena with leaves first and internal nodes in level order, so
  ascending index order is a post-order and each level is a contiguous range of
  rows.
- Pruning recursion and tree loglikelihood, in a row-major layout for reference
  and a feature-blocked layout for production. The blocked layout parallelises
  over features rather than over tree levels, which makes it indifferent to
  tree shape.
- Branch-length optimisation by safeguarded Newton on the closed-form
  stationarity condition, bracketed from above during the same pass that
  prepares the edge constants.
- SIMD tier for `f32` storage via `wide`. `f64` stays scalar on measurement:
  see the table in `src/utils/simd.rs`.
