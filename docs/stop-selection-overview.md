# Selection Criteria — `one_stop_df` vs `rsw` vs `rcap`
---

# Old method (but some parts are used in the new method)

## Common initial steps (applies to all)
1. Detect stops from `monkey_speeddummy` transitions  
2. Find distinct stops, separated by a rise in speed **> 1 cm/s**  
3. Identify stop clusters: split when **cumulative distance traveled between two stops > 50 cm**. Clusters spanning multiple trials are broken at capture boundaries.

---

## `rcap` (Try A Few Times)
- **Multiple stops per trial**
  - Require **≥ 2** stops in a cluster  
- **Target proximity**
  - At least one stop (in the stop cluster ) within **50 cm** of the current target  
  - Keep the **latest cluster** per trial if there are multiple (which is unlikely)  
- **Purpose**: identify repeated attempts

---

## `rsw` (Give Up After Trying)
- **Base DataFrames**
  - `rsw_trials_df`: base trials without firefly context  
  - `rsw_expanded_trials_df`: base trials + firefly proximity annotations  
  - `rsw_w_ff_df`: only trials where there is at least one firefly near the stop  
    - If two stop clusters map to the same firefly around the same time, keep the cluster whose stop is closest to the trajectory point nearest the stop (measured in number of `point_indices`).  
    - *This rule is implemented in `deal_with_duplicated_stop_point_index`, and may change. Such cases are very rare.*  
- **Multiple stops per trial**
  - Require **≥ 2** stops in a cluster  
- **Temporal constraints**
  - Firefly visible **≤ 3 s** ago  
  - Stop within **50 cm** of the firefly position  
  - The temtative current firefly target (that the monkey misses) cannot be the current or the previous captured target
- **Purpose**: identify persistence behavior

---

## `one_stop_df` (One-Stop Misses; no evidence of retrying)
#### Note, the criteria had been updated alongside the function streamline_getting_one_stop_df (criteria were made to be less stringent in order to include all misses)

- **Base DataFrames**
  - `one_stop_df`: Long format dataframe with one row per (stop × nearby firefly)  
  - `one_stop_w_ff_df`: Aggregates one_stop_df by grouping stops and selecting the most recently visible firefly. 
- **Distance filter**
  - Not in a cluster of stops
  - Keep stops in the **25–50 cm** band of firefly (that has been visible within the last 3 s)
  - The temtative current firefly target (that the monkey misses) cannot be the current or the previous captured target
- **Spatial relationship**
  - Focus on stops **near but not at** fireflies  


---


# New method

## Initial assignment (precedence order)

1. **Tag rcap and captures first**  
   Use the rcap/capture methods to label stops that clearly belong to those categories.

2. **Fallback-to-own-target (≤ 50 cm)**  
   For any remaining stop, if it is within **50 cm** of its trial’s intended target, set `candidate_target = target_index`.  
   *This pulls the stop into the same `candidate_target` as a nearby rcap or capture when appropriate.*

3. **rsw from leftovers**  
   From the remaining unlabeled stops, select **rsw** using the rsw method.

4. **One-stop miss from the rest**  
   From what’s still left, select **one-stop misses**.

> After steps 1–4, every labeled stop has an **`candidate_target`** (rcap/capture assignments take precedence over rsw/miss).

5. **Merge consecutive runs by `candidate_target`**  
   Build **consecutive clusters** (ordered by time): each time `candidate_target` changes (including to/from NaN), start a new cluster.

6. **Reassign labels per cluster** (rules below)

---

## Rules for cluster-level re-assignment

Within each **consecutive `candidate_target` cluster**:

1. **If any stop = rcap → entire cluster = rcap.**  
2. **Else, if any stop = capture**:  
   - Cluster size **> 1** → **rcap** (captures embedded in persistence are treated as rcap)  
   - Cluster size **= 1** → **capture**  
3. **Else (no rcap, no capture)**:  
   - Cluster size **> 1** → **rsw**  
   - Cluster size **= 1** → **miss** (one-stop)

Stops with **no associated firefly** (`candidate_target` is NaN) are labeled **`unclassified`**.

---

## Output
- `attempt_type ∈ {capture, rcap, rsw, miss, unclassified}`  
- All stops within the same consecutive `candidate_target` cluster share the **same** final label.  

---

## Summary table

| Cluster condition                     | Final label     |
|--------------------------------------|-----------------|
| Any stop = rcap                      | **rcap**        |
| Contains capture, size > 1           | **rcap**        |
| Contains capture, size = 1           | **capture**     |
| No rcap/capture, size > 1            | **rsw**        |
| No rcap/capture, size = 1            | **miss**        |
| `candidate_target` is NaN               | **unclassified** |
