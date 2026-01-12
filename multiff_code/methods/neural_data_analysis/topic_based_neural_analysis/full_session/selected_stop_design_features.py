# =========================================
# MultiFF Event / Retry GLM predictors
# =========================================

# -------------------------
# 1. Event-aligned temporal basis (raised cosine kernels)
# -------------------------
event_time_basis = [
    'rcos_-0.24s',
    'rcos_-0.16s',
    'rcos_-0.08s',
    'rcos_+0.00s',
    'rcos_+0.08s',
    'rcos_+0.16s',
    'rcos_+0.24s',
]

# -------------------------
# 2. Event boundary indicators
# -------------------------
event_boundary = [
    'prepost',
]

event_kinematic_interactions = [
    'prepost*speed',
]

# -------------------------
# 3. Capture-related signals
# -------------------------
capture_state = [
    'captured',
]

capture_time_basis = [
    'rcos_-0.24s*captured',
    'rcos_-0.16s*captured',
    'rcos_-0.08s*captured',
    'rcos_+0.00s*captured',
    'rcos_+0.08s*captured',
    'rcos_+0.16s*captured',
    'rcos_+0.24s*captured',
]

# -------------------------
# 4. Local event timing / history
# -------------------------
event_history = [
    'time_since_prev_event',
    'time_to_next_event',
    'event_is_first_in_cluster',
]

# -------------------------
# 5. Cluster / bout structure (slow timescale)
# -------------------------
cluster_structure = [
    'prev_gap_s_z',
    'next_gap_s_z',
    'cluster_duration_s_z',
    'cluster_progress_c',
    'cluster_progress_c2',
    'cluster_rel_time_s_z',
    'time_rel_to_event_start',
]

# -------------------------
# 6. Event position within sequence
# -------------------------
sequence_position = [
    'rsw_first',
    'rcap_first',
    'rsw_middle',
    'rcap_middle',
    'rsw_last',
    'rcap_last',
]

# -------------------------
# 7. Miss / retry state
# -------------------------
retry_state = [
    'miss',
    'one_stop_miss',
    'whether_in_retry_series',
]

# -------------------------
# 8. Retry-modulated kinematics
# -------------------------
retry_kinematics = [
    'accel*retry',
    'speed*retry',
    'ang_speed*retry',
]

# -------------------------
# 9. Retry-modulated event timing
# -------------------------
retry_event_timing = [
    'prepost*retry',
    'prepost*speed*retry',
]

# -------------------------
# 10. Retry-modulated event kernels
# -------------------------
retry_event_time_basis = [
    'rcos_-0.24s*retry',
    'rcos_-0.16s*retry',
    'rcos_-0.08s*retry',
    'rcos_+0.00s*retry',
    'rcos_+0.08s*retry',
    'rcos_+0.16s*retry',
    'rcos_+0.24s*retry',
]

# -------------------------
# 11. Retry-modulated history & structure
# -------------------------
retry_history_structure = [
    'time_since_prev_event*retry',
    'time_to_next_event*retry',
    'event_is_first_in_cluster*retry',
    'prev_gap_s_z*retry',
    'cluster_duration_s_z*retry',
    'cluster_progress_c*retry',
    'cluster_progress_c2*retry',
    'cluster_rel_time_s_z*retry',
    'time_rel_to_event_start*retry',
]

# -------------------------
# Full ordered predictor list
# -------------------------
stop_design_predictors = (
    event_time_basis
    + event_boundary
    + event_kinematic_interactions
    + capture_state
    + capture_time_basis
    + event_history
    + cluster_structure
    + sequence_position
    + retry_state
    + retry_kinematics
    + retry_event_timing
    + retry_event_time_basis
    + retry_history_structure
)
