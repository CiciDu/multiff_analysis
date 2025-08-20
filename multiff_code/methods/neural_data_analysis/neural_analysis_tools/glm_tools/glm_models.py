def onset_from_mask(mask):
    m = (mask > 0).astype(int)
    return np.maximum(0, np.diff(m, prepend=0))  # 1 on rising edge

def gate(x, mask):
    x = np.asarray(x)
    m = (mask > 0).astype(float)
    x = np.where(np.isfinite(x), x, 0.0)  # replace NaN with 0 before gating
    return x * m


# -- events (onsets) --
cur_on = onset_from_mask(trial['cur_vis'])
nxt_on = onset_from_mask(trial['nxt_vis'])
# convolve cur_on / nxt_on with your event basis as usual
for k in range(B_evt.shape[1]):
    X_cols.append(convolve_causal(cur_on, B_evt[:,k])); names.append(f"cur_on_rc{k+1}")
    X_cols.append(convolve_causal(nxt_on, B_evt[:,k])); names.append(f"nxt_on_rc{k+1}")

# -- gated continuous/value features --
cur_dist_g = gate(trial['cur_dist'], trial['cur_vis'])
nxt_dist_g = gate(trial['nxt_dist'], trial['nxt_vis'])
cur_sin_g  = gate(np.sin(trial['cur_ang']), trial['cur_vis'])
cur_cos_g  = gate(np.cos(trial['cur_ang']), trial['cur_vis'])
nxt_sin_g  = gate(np.sin(trial['nxt_ang']), trial['nxt_vis'])
nxt_cos_g  = gate(np.cos(trial['nxt_ang']), trial['nxt_vis'])

# If you want **lagged** effects, project onto a 0–300 ms basis:
lags, B_val = raised_cosine_basis(K, t_max=0.30, dt=dt, log_spaced=True)
for k in range(B_val.shape[1]):
    X_cols.append(convolve_causal(cur_dist_g, B_val[:,k])); names.append(f"cur_dist_rc{k+1}")
    X_cols.append(convolve_causal(cur_sin_g,  B_val[:,k])); names.append(f"cur_sin_rc{k+1}")
    X_cols.append(convolve_causal(cur_cos_g,  B_val[:,k])); names.append(f"cur_cos_rc{k+1}")
    X_cols.append(convolve_causal(nxt_dist_g, B_val[:,k])); names.append(f"nxt_dist_rc{k+1}")
    X_cols.append(convolve_causal(nxt_sin_g,  B_val[:,k])); names.append(f"nxt_sin_rc{k+1}")
    X_cols.append(convolve_causal(nxt_cos_g,  B_val[:,k])); names.append(f"nxt_cos_rc{k+1}")

# If you prefer **instantaneous** effects (no lags), skip the basis and add directly:
# X_cols += [cur_dist_g, cur_sin_g, cur_cos_g, nxt_dist_g, nxt_sin_g, nxt_cos_g]
# names  += ["cur_dist", "cur_sin", "cur_cos", "nxt_dist", "nxt_sin", "nxt_cos"]
