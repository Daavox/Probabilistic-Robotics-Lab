#!/usr/bin/env python3
"""
Probabilistic Robotics Lab — Plot Results
==========================================
Reads CSV files from the evaluation_node and generates all plots
required by the project specification.

Usage:
    python3 plot_results.py

Output: ~/ros2_ws/src/prob_lab/results/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────
RESULTS_DIR = os.path.expanduser("~/ros2_ws/src/prob_lab/results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Colors
C_GT  = "#2ecc71"   # green  — Ground Truth
C_KF  = "#e74c3c"   # red    — KF
C_EKF = "#3498db"   # blue   — EKF
C_PF  = "#f39c12"   # orange — PF

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})

# ── Load Data ─────────────────────────────────────────────────
def load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[OK] Loaded {filename}: {len(df)} rows")
    return df

poses  = load("poses.csv")
rmse   = load("rmse.csv")
cov    = load("covariance.csv")
lm     = load("landmark.csv")
drop   = load("interruption.csv")

# ═══════════════════════════════════════════════════════════════
# PLOT 1 — Trajectories (Ground Truth vs KF vs EKF vs PF)
# ═══════════════════════════════════════════════════════════════
if poses is not None:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(poses["gt_x"],  poses["gt_y"],  color=C_GT,  label="Ground Truth (/odom)", zorder=4)
    ax.plot(poses["kf_x"],  poses["kf_y"],  color=C_KF,  label="Kalman Filter (KF)",   zorder=3)
    ax.plot(poses["ekf_x"], poses["ekf_y"], color=C_EKF, label="Extended KF (EKF)",    zorder=2)
    if poses["pf_x"].abs().sum() > 0:
        ax.plot(poses["pf_x"], poses["pf_y"], color=C_PF, label="Particle Filter (PF)", zorder=1)

    # Start / End markers
    ax.scatter(poses["gt_x"].iloc[0],  poses["gt_y"].iloc[0],
               s=120, color="black", marker="o", zorder=5, label="Start")
    ax.scatter(poses["gt_x"].iloc[-1], poses["gt_y"].iloc[-1],
               s=120, color="black", marker="X", zorder=5, label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory Comparison: Ground Truth vs Filters")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_trajectories.png"))
    plt.close()
    print("[OK] Plot 1: Trajectories")

# ═══════════════════════════════════════════════════════════════
# PLOT 2 — RMSE over Time
# ═══════════════════════════════════════════════════════════════
if rmse is not None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rmse["time"], rmse["rmse_kf"],  color=C_KF,  label="KF RMSE")
    ax.plot(rmse["time"], rmse["rmse_ekf"], color=C_EKF, label="EKF RMSE")
    if rmse["rmse_pf"].abs().sum() > 0:
        ax.plot(rmse["time"], rmse["rmse_pf"],  color=C_PF,  label="PF RMSE")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RMSE [m]")
    ax.set_title("Cumulative RMSE over Time")
    ax.legend()

    # Print final values
    final_kf  = rmse["rmse_kf"].iloc[-1]
    final_ekf = rmse["rmse_ekf"].iloc[-1]
    final_pf  = rmse["rmse_pf"].iloc[-1]
    ax.text(0.02, 0.95,
        f"Final RMSE — KF: {final_kf:.4f}m  EKF: {final_ekf:.4f}m  PF: {final_pf:.4f}m",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_rmse.png"))
    plt.close()
    print(f"[OK] Plot 2: RMSE  KF={final_kf:.4f}m  EKF={final_ekf:.4f}m  PF={final_pf:.4f}m")

# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Position Error X and Y separately
# ═══════════════════════════════════════════════════════════════
if rmse is not None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(rmse["time"], rmse["error_x_kf"],  color=C_KF,  label="KF")
    ax1.plot(rmse["time"], rmse["error_x_ekf"], color=C_EKF, label="EKF")
    if "error_x_pf" in rmse.columns:
        ax1.plot(rmse["time"], rmse["error_x_pf"],  color=C_PF,  label="PF")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Error X [m]")
    ax1.set_title("Position Error over Time")
    ax1.legend()

    ax2.plot(rmse["time"], rmse["error_y_kf"],  color=C_KF,  label="KF")
    ax2.plot(rmse["time"], rmse["error_y_ekf"], color=C_EKF, label="EKF")
    if "error_y_pf" in rmse.columns:
        ax2.plot(rmse["time"], rmse["error_y_pf"],  color=C_PF,  label="PF")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error Y [m]")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_position_error.png"))
    plt.close()
    print("[OK] Plot 3: Position Error X/Y")

# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Covariance over Time
# ═══════════════════════════════════════════════════════════════
if cov is not None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["Covariance X [m²]", "Covariance Y [m²]", "Covariance θ [rad²]"]
    cols_kf  = ["kf_cov_x",  "kf_cov_y",  "kf_cov_theta"]
    cols_ekf = ["ekf_cov_x", "ekf_cov_y", "ekf_cov_theta"]
    cols_pf  = ["pf_cov_x",  "pf_cov_y",  "pf_cov_theta"]

    for i, ax in enumerate(axes):
        ax.plot(cov["time"], cov[cols_kf[i]],  color=C_KF,  label="KF")
        ax.plot(cov["time"], cov[cols_ekf[i]], color=C_EKF, label="EKF")
        if cov[cols_pf[i]].abs().sum() > 0:
            ax.plot(cov["time"], cov[cols_pf[i]],  color=C_PF,  label="PF")
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right")

    axes[0].set_title("Covariance (Uncertainty) over Time")
    axes[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_covariance.png"))
    plt.close()
    print("[OK] Plot 4: Covariance")

# ═══════════════════════════════════════════════════════════════
# PLOT 5 — Landmark Detection
# ═══════════════════════════════════════════════════════════════
if lm is not None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(lm["time"], lm["true_dist"], color=C_GT,  label="True Distance", linestyle="--")
    ax1.plot(lm["time"], lm["kf_dist"],   color=C_KF,  label="KF estimated")
    ax1.plot(lm["time"], lm["ekf_dist"],  color=C_EKF, label="EKF estimated")
    if lm["pf_dist"].abs().sum() > 0:
        ax1.plot(lm["time"], lm["pf_dist"],  color=C_PF,  label="PF estimated")
    ax1.set_ylabel("Distance to Landmark [m]")
    ax1.set_title("Landmark Detection: Distance Estimation")
    ax1.legend()

    ax2.plot(lm["time"], lm["kf_err"],  color=C_KF,  label="KF error")
    ax2.plot(lm["time"], lm["ekf_err"], color=C_EKF, label="EKF error")
    if lm["pf_err"].abs().sum() > 0:
        ax2.plot(lm["time"], lm["pf_err"], color=C_PF,  label="PF error")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Landmark Distance Error [m]")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_landmark.png"))
    plt.close()
    print("[OK] Plot 5: Landmark Detection")

# ═══════════════════════════════════════════════════════════════
# PLOT 6 — Measurement Interruption (Dropout)
# ═══════════════════════════════════════════════════════════════
if drop is not None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Shade each dropout region individually (with gaps between them)
    in_dropout = drop["in_dropout"].values
    times = drop["time"].values
    i = 0
    first_label = True
    while i < len(in_dropout):
        if in_dropout[i] == 1:
            t_start = times[i]
            while i < len(in_dropout) and in_dropout[i] == 1:
                i += 1
            t_end = times[i-1]
            label = "Sensor Dropout" if first_label else ""
            first_label = False
            for ax in axes:
                ax.axvspan(t_start, t_end, alpha=0.3, color="red", label=label)
        else:
            i += 1

    # Trajectories during dropout
    ax1 = axes[0]
    ax1.plot(drop["time"], drop["gt_x"],  color=C_GT,  label="GT x",  linestyle="--")
    ax1.plot(drop["time"], drop["kf_x"],  color=C_KF,  label="KF x")
    ax1.plot(drop["time"], drop["ekf_x"], color=C_EKF, label="EKF x")
    if drop["pf_x"].abs().sum() > 0:
        ax1.plot(drop["time"], drop["pf_x"],  color=C_PF,  label="PF x")
    ax1.set_ylabel("X Position [m]")
    ax1.set_title("Behavior During Sensor Dropout (red = no measurements)")
    ax1.legend(loc="upper left", fontsize=8)

    # Error during dropout
    ax2 = axes[1]
    ax2.plot(drop["time"], drop["kf_err"],  color=C_KF,  label="KF error")
    ax2.plot(drop["time"], drop["ekf_err"], color=C_EKF, label="EKF error")
    if drop["pf_err"].abs().sum() > 0:
        ax2.plot(drop["time"], drop["pf_err"],  color=C_PF,  label="PF error")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position Error [m]")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_dropout.png"))
    plt.close()
    print("[OK] Plot 6: Measurement Interruption / Dropout")

# ═══════════════════════════════════════════════════════════════
# PLOT 7 — Final RMSE Bar Chart (Summary)
# ═══════════════════════════════════════════════════════════════
if rmse is not None:
    fig, ax = plt.subplots(figsize=(7, 5))

    filters = ["KF", "EKF", "PF"]
    values  = [
        rmse["rmse_kf"].iloc[-1],
        rmse["rmse_ekf"].iloc[-1],
        rmse["rmse_pf"].iloc[-1],
    ]
    colors = [C_KF, C_EKF, C_PF]
    bars = ax.bar(filters, values, color=colors, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f} m", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Final RMSE [m]")
    ax.set_title("Filter Comparison: Final RMSE")
    ax.set_ylim(0, max(values) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_rmse_bar.png"))
    plt.close()
    print("[OK] Plot 7: RMSE Bar Chart")

# ═══════════════════════════════════════════════════════════════
print(f"\n✓ All plots saved to: {PLOTS_DIR}")
print("  01_trajectories.png")
print("  02_rmse.png")
print("  03_position_error.png")
print("  04_covariance.png")
print("  05_landmark.png")
print("  06_dropout.png")
print("  07_rmse_bar.png")
