# Selection Criteria — `one_stop_df` vs `GUAT` vs `TAFT`
---

# Old method (but some parts are used in the new method)

## Common initial steps (applies to all)
1. Detect stops from `monkey_speeddummy` transitions  
2. Find distinct stops, separated by a rise in speed **> 1 cm/s**  
3. Identify stop clusters: split when **cumulative distance traveled between two stops > 50 cm**. Clusters spanning multiple trials are broken at capture boundaries.

---

## `TAFT` (Try A Few Times)
- **Multiple stops per trial**
  - Require **≥ 2** stops in a cluster  
- **Target proximity**
  - At least one stop (in the stop cluster ) within **50 cm** of the current target  
  - Keep the **latest cluster** per trial if there are multiple (which is unlikely)  
- **Purpose**: identify repeated attempts

---

## `GUAT` (Give Up After Trying)
- **Base DataFrames**
  - `GUAT_trials_df`: base trials without firefly context  
  - `GUAT_expanded_trials_df`: base trials + firefly proximity annotations  
  - `GUAT_w_ff_df`: only trials where there is at least one firefly near the stop  
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

1. **Tag TAFT and captures first**  
   Use the TAFT/capture methods to label stops that clearly belong to those categories.

2. **Fallback-to-own-target (≤ 50 cm)**  
   For any remaining stop, if it is within **50 cm** of its trial’s intended target, set `associated_ff = target_index`.  
   *This pulls the stop into the same `associated_ff` as a nearby TAFT or capture when appropriate.*

3. **GUAT from leftovers**  
   From the remaining unlabeled stops, select **GUAT** using the GUAT method.

4. **One-stop miss from the rest**  
   From what’s still left, select **one-stop misses**.

> After steps 1–4, every labeled stop has an **`associated_ff`** (TAFT/capture assignments take precedence over GUAT/miss).

5. **Merge consecutive runs by `associated_ff`**  
   Build **consecutive clusters** (ordered by time): each time `associated_ff` changes (including to/from NaN), start a new cluster.

6. **Reassign labels per cluster** (rules below)

---

## Rules for cluster-level re-assignment

Within each **consecutive `associated_ff` cluster**:

1. **If any stop = TAFT → entire cluster = TAFT.**  
2. **Else, if any stop = capture**:  
   - Cluster size **> 1** → **TAFT** (captures embedded in persistence are treated as TAFT)  
   - Cluster size **= 1** → **capture**  
3. **Else (no TAFT, no capture)**:  
   - Cluster size **> 1** → **GUAT**  
   - Cluster size **= 1** → **miss** (one-stop)

Stops with **no associated firefly** (`associated_ff` is NaN) are labeled **`unclassified`**.

---

## Output
- `attempt_type ∈ {capture, TAFT, GUAT, miss, unclassified}`  
- All stops within the same consecutive `associated_ff` cluster share the **same** final label.  

---

## Summary table

| Cluster condition                     | Final label     |
|--------------------------------------|-----------------|
| Any stop = TAFT                      | **TAFT**        |
| Contains capture, size > 1           | **TAFT**        |
| Contains capture, size = 1           | **capture**     |
| No TAFT/capture, size > 1            | **GUAT**        |
| No TAFT/capture, size = 1            | **miss**        |
| `associated_ff` is NaN               | **unclassified** |
