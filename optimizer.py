"""
optimizer.py  —  FAST Inverse Optimizer
----------------------------------------
Strategy: Pure TensorFlow gradient descent on batched candidates.
NO scipy. NO per-sample DNN calls. Everything runs as a tensor batch.

Objective : Minimise H2 (VARY 2) subject to PURITY and MASSFLOW targets.
Typical runtime : 3-8 seconds on CPU.
"""

import numpy as np
import tensorflow as tf

# ── Physical bounds ───────────────────────────────────────────────────────────
BOUNDS = {
    "VARY 1 FEED": (0.000,  0.050),
    "VARY 2 H2":   (0.0018, 0.004),
    "VARY 3 F1":   (8.0,    12.0),
    "VARY 4 F2":   (10.0,   14.0),
    "VARY 5 R2":   (43.0,   50.0),
    "VARY 6 R3":   (43.0,   50.0),
}
FEATURE_NAMES = list(BOUNDS.keys())

IDX_H2       = 1
IDX_PURITY   = 0
IDX_MASSFLOW = 1


def _build_tensors(scaler_X):
    """Pre-convert StandardScaler to TF constants so scaling stays in-graph."""
    mean_x = tf.constant(scaler_X.mean_,  dtype=tf.float32)
    std_x  = tf.constant(scaler_X.scale_, dtype=tf.float32)
    return mean_x, std_x


def _engineer_batch(x_batch: tf.Tensor) -> tf.Tensor:
    """
    x_batch : (N, 6)  raw inputs
    returns  : (N, 8) with h2_feed_ratio and flash_p_delta appended
    """
    feed = x_batch[:, 0:1]
    h2   = x_batch[:, 1:2]
    f1   = x_batch[:, 2:3]
    f2   = x_batch[:, 3:4]
    h2_feed_ratio = h2 / (feed + 1e-9)
    flash_p_delta  = f2 - f1
    return tf.concat([x_batch, h2_feed_ratio, flash_p_delta], axis=1)


def _forward_batch(x_raw_batch, model, mean_x, std_x):
    """
    Full forward pass for a batch of raw 6-dim inputs.
    Returns model output (N, 4) in PowerTransformer-scaled space.
    """
    x8   = _engineer_batch(x_raw_batch)   # (N, 8)
    x_sc = (x8 - mean_x) / std_x         # StandardScaler in-graph
    y_sc = model(x_sc, training=False)    # (N, 4)
    return y_sc


def run_inverse_optimization(
    target_purity:   float,
    target_massflow: float,
    model,
    scaler_X,
    pt_y,
    purity_tol:   float = 0.005,
    massflow_tol: float = 0.002,
    n_restarts:   int   = 200,
    n_steps:      int   = 400,
    lr:           float = 0.002,
):
    """
    Batched TF gradient-descent inverse optimizer.

    All n_restarts candidates are optimised simultaneously in a single
    batched forward/backward pass — completes in ~5 seconds on CPU.

    Returns dict: success, inputs, predicted, h2_minimized, message
    """

    lo = np.array([b[0] for b in [BOUNDS[k] for k in FEATURE_NAMES]], dtype=np.float32)
    hi = np.array([b[1] for b in [BOUNDS[k] for k in FEATURE_NAMES]], dtype=np.float32)

    mean_x, std_x = _build_tensors(scaler_X)

    # ── Convert targets to PowerTransformer-scaled space ─────────────────────
    # Use a representative row to avoid transform edge cases
    dummy = np.array([[target_purity, target_massflow, 0.088, 0.181]], dtype=np.float64)
    scaled_targets = pt_y.transform(dummy)[0]
    t_purity_sc   = tf.constant(float(scaled_targets[IDX_PURITY]),   dtype=tf.float32)
    t_massflow_sc = tf.constant(float(scaled_targets[IDX_MASSFLOW]), dtype=tf.float32)

    # Scale tolerances to match PowerTransformer space via finite differences
    eps = 1e-4
    def _scaled_tol(idx, raw_tol):
        d_hi = dummy.copy(); d_hi[0, idx] += eps
        d_lo = dummy.copy(); d_lo[0, idx] -= eps
        scale = abs(pt_y.transform(d_hi)[0, idx] - pt_y.transform(d_lo)[0, idx]) / (2*eps)
        return float(raw_tol * scale)

    tol_p_sc = tf.constant(_scaled_tol(IDX_PURITY,   purity_tol),   dtype=tf.float32)
    tol_m_sc = tf.constant(_scaled_tol(IDX_MASSFLOW, massflow_tol), dtype=tf.float32)

    PENALTY = tf.constant(1e4, dtype=tf.float32)

    # ── Initialise candidates ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    corner_starts = np.array([
        [0.5, 0.0, 0.5, 0.5, 0.5, 0.5],   # min H2
        [0.5, 1.0, 0.5, 0.5, 0.5, 0.5],   # max H2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.2, 0.1, 0.3, 0.4, 0.5, 0.5],
        [0.8, 0.1, 0.7, 0.6, 0.5, 0.5],
        [0.5, 0.1, 0.5, 0.5, 0.3, 0.7],
        [0.5, 0.1, 0.5, 0.5, 0.7, 0.3],
    ], dtype=np.float32)

    rand_starts   = rng.random((n_restarts - len(corner_starts), 6)).astype(np.float32)
    x_norm_init   = np.vstack([corner_starts, rand_starts])   # (N, 6) in [0,1]

    # Reparametrise: x_raw = lo + sigmoid(w) * (hi - lo)  → always in bounds
    x_norm_clipped = np.clip(x_norm_init, 0.01, 0.99)
    w_init = np.log(x_norm_clipped / (1.0 - x_norm_clipped)).astype(np.float32)

    w      = tf.Variable(w_init, trainable=True, dtype=tf.float32)
    lo_tf  = tf.constant(lo, dtype=tf.float32)
    hi_tf  = tf.constant(hi, dtype=tf.float32)
    opt_tf = tf.keras.optimizers.Adam(learning_rate=lr)

    # ── Batched gradient descent ──────────────────────────────────────────────
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            x_raw = lo_tf + tf.sigmoid(w) * (hi_tf - lo_tf)     # (N, 6)
            y_sc  = _forward_batch(x_raw, model, mean_x, std_x)  # (N, 4)

            purity_pred   = y_sc[:, IDX_PURITY]
            massflow_pred = y_sc[:, IDX_MASSFLOW]
            h2_raw        = x_raw[:, IDX_H2]

            p_viol = tf.nn.relu(tf.abs(purity_pred   - t_purity_sc)   - tol_p_sc)
            m_viol = tf.nn.relu(tf.abs(massflow_pred - t_massflow_sc) - tol_m_sc)

            h2_norm  = (h2_raw - lo_tf[IDX_H2]) / (hi_tf[IDX_H2] - lo_tf[IDX_H2])
            loss_per = h2_norm + PENALTY * (p_viol**2 + m_viol**2)
            loss     = tf.reduce_mean(loss_per)

        grads = tape.gradient(loss, [w])
        opt_tf.apply_gradients(zip(grads, [w]))
        return loss

    for _ in range(n_steps):
        step()

    # ── Evaluate all candidates in real units ─────────────────────────────────
    x_final = (lo_tf + tf.sigmoid(w) * (hi_tf - lo_tf)).numpy()   # (N, 6)

    x8_all = np.hstack([
        x_final,
        (x_final[:, 1] / (x_final[:, 0] + 1e-9)).reshape(-1, 1),
        (x_final[:, 3] - x_final[:, 2]).reshape(-1, 1),
    ])
    x_sc_all   = scaler_X.transform(x8_all)
    y_sc_all   = model.predict(x_sc_all, verbose=0, batch_size=256)
    y_real_all = pt_y.inverse_transform(y_sc_all)   # (N, 4)

    # ── Select best feasible candidate (lowest H2) ────────────────────────────
    best_idx      = None
    best_h2       = np.inf
    best_feasible = False

    for i in range(len(x_final)):
        p_ok     = abs(y_real_all[i, IDX_PURITY]   - target_purity)   <= purity_tol
        m_ok     = abs(y_real_all[i, IDX_MASSFLOW] - target_massflow) <= massflow_tol
        feasible = p_ok and m_ok
        h2_val   = x_final[i, IDX_H2]

        if feasible and (not best_feasible or h2_val < best_h2):
            best_feasible = True
            best_h2  = h2_val
            best_idx = i
        elif not best_feasible and h2_val < best_h2:
            best_h2  = h2_val
            best_idx = i

    best_x6  = x_final[best_idx]
    best_out = y_real_all[best_idx]

    result_inputs = {name: float(best_x6[i]) for i, name in enumerate(FEATURE_NAMES)}
    result_preds  = {
        "PURITY":   float(best_out[0]),
        "MASSFLOW": float(best_out[1]),
        "CO2OUT":   float(best_out[2]),
        "H2OUT":    float(best_out[3]),
    }

    if best_feasible:
        msg = (
            f"✅ Feasible solution found. "
            f"H₂ minimised to {best_x6[IDX_H2]:.6f} kmol/s "
            f"(PURITY ±{purity_tol}, MASSFLOW ±{massflow_tol} satisfied)."
        )
    else:
        p_err = abs(best_out[0] - target_purity)
        m_err = abs(best_out[1] - target_massflow)
        msg = (
            f"⚠️ Best approximation found — no fully feasible point. "
            f"PURITY error: {p_err:.4f}, MASSFLOW error: {m_err:.5f}. "
            f"Try relaxing tolerances or picking targets within the training range "
            f"(PURITY 0.62–0.99, MASSFLOW 0.086–0.095)."
        )

    return {
        "success":      best_feasible,
        "inputs":       result_inputs,
        "predicted":    result_preds,
        "h2_minimized": float(best_x6[IDX_H2]),
        "message":      msg,
    }