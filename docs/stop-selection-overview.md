# Selection Criteria — `one_stop_df` vs `GUAT` vs `TAFT`
---

## Common initial steps (applies to all)
1. Detect stops from `monkey_speeddummy` transitions  
2. Find distinct stops, separated by a rise in speed **> 1 cm/s**  
3. Identify stop clusters: split when **cumulative distance > 50 cm**. Clusters spanning multiple trials are broken at capture boundaries.

---

## `one_stop_df` (One-Stop Misses; no evidence of retrying)
#### Note, the criteria had been updated alongside the function streamline_getting_one_stop_df (criteria were made to be less stringent in order to include all misses)

- **Base DataFrames**
  - `one_stop_df`: Long format dataframe with one row per (stop × nearby firefly)  
  - `one_stop_w_ff_df`: Aggregates one_stop_df by grouping stops and selecting the most recently visible firefly. Contains nearby_alive_ff_indices (list of all nearby fireflies) and latest_visible_ff (primary firefly)
- **Distance filter**
  - (currently not used) At least 25 cm in cum distance from the point when previous or next ff was caught
  - (currently not used) At least 50 cm in absolute distance (which means it has also to be > 50 cm in cum_distance) from the stop before or after
  - (has added) Not in a cluster of stops
  - Keep stops in the **25–50 cm** band of firefly (that has been visible within the last 2.5 s)
- **Spatial relationship**
  - Focus on stops **near but not at** fireflies  
  - Output in long format: **one row per (stop × nearby firefly)**  
- **Purpose**: identify potential decision points  

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
  - Firefly visible **≤ 2.5 s** ago  
  - Stop within **50 cm** of the firefly position  
- **Purpose**: identify persistence behavior

---

## `TAFT` (Try A Few Times)
- **Multiple stops per trial**
  - Require **≥ 2** stops in a cluster  
- **Target proximity**
  - Stop within **50 cm** of the current target  
  - Keep the **latest cluster** per trial if there are multiple (which is unlikely)  
- **Purpose**: identify repeated attempts

---
