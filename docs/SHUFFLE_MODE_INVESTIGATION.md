# Shuffle Mode Investigation: `none` vs `timeshift_fold`

## Summary

**Bug found and fixed:** The outer-fold evaluation in `run_nested_kernelwidth_cv` was *not* applying `shuffle_mode` to the training labels. The inner CV (kernel-width search) did shuffle, but `_evaluate_outer_fold` → `_train_test_single_fold` always trained on real labels. So the final reported scores were always real decoding performance regardless of shuffle_mode. **Fix:** Pass `shuffle_mode` and `groups_train` into `_evaluate_outer_fold` and apply the same shuffle logic in `_train_test_single_fold` before fitting.

---

## Pipeline Trace

### 1. Script → Runner

**`decode_stops_script.py`** (and decode_pn, decode_vis):
```python
results_df_shuffled = runner.run(shuffle_mode='timeshift_fold', ...)
results_df = runner.run(shuffle_mode='none', ...)
```

### 2. Runner → CV

**`base_decoding_runner.py`**:
- `run()` → `run_cv_decoding(shuffle_mode=...)`
- `shuffle_mode != "none"` → `save_dir` becomes `save_dir / f"shuffle_{shuffle_mode}"` (separate cache paths)
- `_run_single_model_cv(shuffle_mode=...)` → `run_nested_kernelwidth_cv(shuffle_mode=...)`
- `_run_inner_kernelwidth_search(shuffle_mode=...)` → `cv_decoding.run_cv_decoding(shuffle_mode=...)`

### 3. CV → Training

**`cv_decoding.py`**:
- `run_regression_cv()` / `run_classification_cv()` per fold:
  - `none`: `y_tr = y[tr]`
  - `timeshift_fold`: `y_tr = _timeshift_1d(y[tr], rng, min_shift=min_shift)`
- `_timeshift_1d`: circular shift by random offset in `[min_shift, n - min_shift]`
- `min_shift = max(1, config.buffer_samples + 1)` = 21 (default `buffer_samples=20`)

### 4. Config

- **DecodingRunConfig**: `buffer_samples=20`, `cv_mode='blocked_time_buffered'`
- **No cache collision**: `none` saves to `save_dir/...`, `timeshift_fold` to `save_dir/shuffle_timeshift_fold/...`

---

## Verification

### Check that timeshift runs

When `timeshift_fold` is used, `cv_decoding._timeshift_1d` prints:
```
Doing a circle time shift with shift: <N>
```
You should see many of these (one per fold × feature). If they never appear, something is wrong.

### Stricter null: `foldwise`

To test a more destructive null:
```python
runner.run(shuffle_mode='timeshift_fold', ...)
```
This permutes labels within each fold (full permutation) and should reduce performance more than `timeshift_fold`.

---

## Why `none` and `timeshift_fold` can look similar

1. **Circular shift keeps autocorrelation** – Mean, variance, and autocorrelation of `y` are preserved, so smooth trends can remain partially decodable.
2. **Shared slow trends** – If decodability comes from session-level drift rather than trial-level alignment, timeshift may not fully remove signal.
3. **Weak real signal** – If true decoding is near chance, both conditions give similar scores.
4. **Blocked time splits** – Train folds are non-contiguous time blocks; circular shift can still leave structure that correlates with test data.

---

## Files checked

- `jobs/decoding/decode_stops/scripts/decode_stops_script.py`
- `multiff_code/.../base_decoding_runner.py`
- `multiff_code/.../cv_decoding.py` (`run_regression_cv`, `run_classification_cv`, `_timeshift_1d`)
- `multiff_code/.../decode_stops_pipeline.py`
