"""
Evolution Curve Visualization
==============================
Plots Round 1 (Naive) vs Round 2 (Improved) evolutionary learning curves.
Generates publication-quality figures.
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import EVOLUTION_LOG, EVOLUTION_LOG_V2, SEG_FIGURE_DIR, ensure_dirs  # noqa: E402

OUT_DIR = SEG_FIGURE_DIR
ensure_dirs(OUT_DIR)

# Load logs
with open(EVOLUTION_LOG) as f:
    v1_log = json.load(f)

with open(EVOLUTION_LOG_V2) as f:
    v2_log = json.load(f)

# Extract data
v1_gens = [e["generation"] for e in v1_log]
v1_dice = [e["dice"] for e in v1_log]
v1_iou = [e["iou"] for e in v1_log]
v1_actions = [e.get("action", "initial") for e in v1_log]

v2_gens = [e["generation"] for e in v2_log]
v2_dice = [e["dice"] for e in v2_log]
v2_iou = [e["iou"] for e in v2_log]
v2_actions = [e.get("action", "initial") for e in v2_log]

# Also extract the actual DSC before rollback for v1
v1_dice_actual = []
for e in v1_log:
    if e.get("action") == "rollback":
        # Find the actual dice from the output (stored in log as best_dice after rollback)
        # We need the raw score -- check if there's iou mismatch
        v1_dice_actual.append(e["dice"])
    else:
        v1_dice_actual.append(e["dice"])

# ============================================================
# Figure 1: DSC Evolution Curve (main figure)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

# --- Left: DSC ---
ax1.set_title("Dice Similarity Coefficient (DSC)", fontsize=13, fontweight="bold", pad=10)

# Baseline: generation 0 of the log, not a hard-coded value
BASE_DICE = v1_dice[0]
BASE_IOU = v1_iou[0]
ax1.axhline(y=BASE_DICE, color="#888888", linestyle="--", linewidth=1, alpha=0.7, label="Initial (Gen 0)")

# Round 1
for i in range(len(v1_gens)):
    color = "#e74c3c" if v1_actions[i] == "rollback" else (
        "#2ecc71" if v1_actions[i] == "improved" else "#3498db")
    marker = "x" if v1_actions[i] == "rollback" else (
        "^" if v1_actions[i] == "improved" else "o")
    ax1.scatter(v1_gens[i], v1_dice[i], color=color, marker=marker, s=60, zorder=5,
                edgecolors="white", linewidth=0.5)
ax1.plot(v1_gens, v1_dice, color="#3498db", linewidth=1.5, alpha=0.6, label="Round 1 (Naive)")

# Round 2
for i in range(len(v2_gens)):
    color = "#e74c3c" if v2_actions[i] == "rollback" else (
        "#2ecc71" if v2_actions[i] == "improved" else "#e67e22")
    marker = "x" if v2_actions[i] == "rollback" else (
        "^" if v2_actions[i] == "improved" else "o")
    ax1.scatter(v2_gens[i], v2_dice[i], color=color, marker=marker, s=60, zorder=5,
                edgecolors="white", linewidth=0.5)
ax1.plot(v2_gens, v2_dice, color="#e67e22", linewidth=1.5, alpha=0.6, label="Round 2 (Improved)")

ax1.set_xlabel("Generation", fontsize=11)
ax1.set_ylabel("DSC", fontsize=11)
ax1.set_xlim(-0.5, max(v1_gens + v2_gens) + 0.5)
ax1.margins(y=0.15)          # autoscale; do not fix to one dataset's range
ax1.set_xticks(range(0, 11))
ax1.grid(True, alpha=0.3)

legend_elements = [
    plt.Line2D([0], [0], color="#3498db", linewidth=2, label="Round 1 (Naive)"),
    plt.Line2D([0], [0], color="#e67e22", linewidth=2, label="Round 2 (Improved)"),
    plt.Line2D([0], [0], color="#888888", linestyle="--", linewidth=1, label="Initial (Gen 0)"),
    plt.scatter([], [], color="#2ecc71", marker="^", s=60, label="Improved"),
    plt.scatter([], [], color="#3498db", marker="o", s=60, label="Maintained"),
    plt.scatter([], [], color="#e74c3c", marker="x", s=60, label="Rollback"),
]
ax1.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

# --- Right: IoU ---
ax2.set_title("Intersection over Union (IoU)", fontsize=13, fontweight="bold", pad=10)

ax2.axhline(y=BASE_IOU, color="#888888", linestyle="--", linewidth=1, alpha=0.7)

for i in range(len(v1_gens)):
    color = "#e74c3c" if v1_actions[i] == "rollback" else "#3498db"
    marker = "x" if v1_actions[i] == "rollback" else "o"
    ax2.scatter(v1_gens[i], v1_iou[i], color=color, marker=marker, s=60, zorder=5,
                edgecolors="white", linewidth=0.5)
ax2.plot(v1_gens, v1_iou, color="#3498db", linewidth=1.5, alpha=0.6, label="Round 1")

for i in range(len(v2_gens)):
    color = "#e74c3c" if v2_actions[i] == "rollback" else (
        "#2ecc71" if v2_actions[i] == "improved" else "#e67e22")
    marker = "x" if v2_actions[i] == "rollback" else (
        "^" if v2_actions[i] == "improved" else "o")
    ax2.scatter(v2_gens[i], v2_iou[i], color=color, marker=marker, s=60, zorder=5,
                edgecolors="white", linewidth=0.5)
ax2.plot(v2_gens, v2_iou, color="#e67e22", linewidth=1.5, alpha=0.6, label="Round 2")

ax2.set_xlabel("Generation", fontsize=11)
ax2.set_ylabel("IoU", fontsize=11)
ax2.set_xlim(-0.5, max(v1_gens + v2_gens) + 0.5)
ax2.margins(y=0.15)
ax2.set_xticks(range(0, 11))
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)

plt.tight_layout()
fig.savefig(OUT_DIR / "evolution_curve_dsc_iou.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'evolution_curve_dsc_iou.png'}")


# ============================================================
# Figure 2: Action Distribution (bar chart)
# ============================================================
fig2, ax3 = plt.subplots(figsize=(8, 5), dpi=150)

categories = ["Improved", "Maintained", "Rollback"]
v1_counts = [
    sum(1 for a in v1_actions[1:] if a == "improved"),
    sum(1 for a in v1_actions[1:] if a == "maintained"),
    sum(1 for a in v1_actions[1:] if a == "rollback"),
]
v2_counts = [
    sum(1 for a in v2_actions[1:] if a == "improved"),
    sum(1 for a in v2_actions[1:] if a == "maintained"),
    sum(1 for a in v2_actions[1:] if a == "rollback"),
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, v1_counts, width, label="Round 1 (Naive)",
                color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.5, edgecolor="gray")
bars2 = ax3.bar(x + width/2, v2_counts, width, label="Round 2 (Improved)",
                color=["#2ecc71", "#e67e22", "#e74c3c"], alpha=0.9, edgecolor="gray")

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax3.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12,
                 fontweight='bold', color='gray')
for bar in bars2:
    h = bar.get_height()
    ax3.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12,
                 fontweight='bold')

ax3.set_title("Evolutionary Selection Actions: Round 1 vs Round 2",
              fontsize=13, fontweight="bold", pad=10)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=11)
ax3.set_ylim(0, max(len(v1_gens), len(v2_gens)))
ax3.legend(fontsize=10)
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig(OUT_DIR / "evolution_action_distribution.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'evolution_action_distribution.png'}")


# ============================================================
# Figure 3: Confidence Statistics
# ============================================================
fig3, ax4 = plt.subplots(figsize=(10, 5), dpi=150)

v2_conf_above = []
v2_conf_total = []
for e in v2_log[1:]:
    cs = e.get("confidence_stats", {})
    v2_conf_above.append(cs.get("above_threshold", 0))
    v2_conf_total.append(cs.get("total", 1))

v2_conf_pct = [a/max(t, 1)*100 for a, t in zip(v2_conf_above, v2_conf_total)]

bars = ax4.bar(range(1, 11), v2_conf_pct, color="#27ae60", alpha=0.8, edgecolor="white")
ax4.axhline(y=97, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Threshold: {0.97}")

for i, (bar, pct) in enumerate(zip(bars, v2_conf_pct)):
    ax4.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width()/2, pct),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax4.set_title("High-Confidence Pseudo-Label Rate per Generation (Round 2)",
              fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Generation", fontsize=11)
ax4.set_ylabel("Percentage above threshold (%)", fontsize=11)
ax4.set_xticks(range(1, 11))
ax4.set_ylim(95, 101)
ax4.legend(fontsize=10)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig3.savefig(OUT_DIR / "confidence_stats.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'confidence_stats.png'}")


# ============================================================
# Summary table
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Round 1: Initial={v1_dice[0]:.4f} -> Best={max(v1_dice):.4f} "
      f"(Improved:{v1_counts[0]} Maintained:{v1_counts[1]} Rollback:{v1_counts[2]})")
print(f"Round 2: Initial={v2_dice[0]:.4f} -> Best={max(v2_dice):.4f} "
      f"(Improved:{v2_counts[0]} Maintained:{v2_counts[1]} Rollback:{v2_counts[2]})")

plt.show()
