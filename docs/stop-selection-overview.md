# Selection Criteria — `one_stop_df` vs `GUAT` vs `TAFT`
---

## Common initial steps (applies to all)
1. Detect stops from `monkey_speeddummy` transitions  
2. Find distinct stops, separated by a rise in speed **> 1 cm/s**  
3. Identify stop clusters: split when **cumulative distance > 50 cm**

---

## `one_stop_df`
- **Distance filter**
  - Exclude stops **< 25 cm** from any firefly  
  - Keep stops in the **25–50 cm** band  
- **Spatial relationship**
  - Focus on stops **near but not at** fireflies  
  - Output in long format: **one row per (stop × nearby firefly)**  
- **Purpose**: identify potential decision points  

*Potential additional filters (to confirm)*  
- Minimum stop duration: **0.02 s**  
- Stop debouncing: **0.15 s**

---

## `GUAT` (Going Up Against Target)
- **Base DataFrames**
  - `GUAT_trials_df`: base trials without firefly context  
  - `GUAT_expanded_trials_df`: base trials + firefly proximity annotations  
  - `GUAT_w_ff_df`: only trials where there is at least one firefly near the stop  
    - If two stop clusters map to the same firefly around the same time, keep the cluster whose stop is closest to the trajectory point nearest the stop (measured in number of `point_indices`). This is implemented in deal_with_duplicated_stop_point_index, and is subject to change. At the same time, such a situation is also very rare. 
- **Multiple stops per trial**
  - Require **≥ 2** stops in a cluster  
- **Separated by captures**
  - Clusters spanning multiple trials are broken by capture boundaries  
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
  - Keep **latest cluster** per trial  
- **Purpose**: identify repeated attempts

---
