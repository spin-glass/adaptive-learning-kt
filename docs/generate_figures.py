"""README 用の図を生成するスクリプト。

Usage:
    python docs/generate_figures.py
"""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl

from src.data.sample import build_sample

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "docs"

N_USERS = 5_000
SEED = 42

# --- Japanese font ---
jp_fonts = [
    f.name for f in fm.fontManager.ttflist
    if "Hiragino" in f.name or "Gothic" in f.name or "Noto Sans CJK" in f.name
]
if jp_fonts:
    plt.rcParams["font.family"] = jp_fonts[0]
plt.rcParams["axes.unicode_minus"] = False

# --- Palette ---
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"


def generate_skill_gap_figure() -> None:
    """ユーザーごとのPart別正答率ヒートマップを生成する。

    「同じ学習者でもスキルごとに知識状態がバラバラ」を示す図。
    """
    result = build_sample(RAW_DIR, n_users=N_USERS, seed=SEED, processed_dir=PROCESSED_DIR)
    df = result.df

    # Active以上 (100問超) のユーザーに限定
    seq_lens = df.group_by("user_id").agg(pl.len().alias("n"))
    active_uids = seq_lens.filter(pl.col("n") > 100).select("user_id")
    active_df = df.join(active_uids, on="user_id").filter(pl.col("correct").is_not_null())

    # ユーザーごと・Part別の正答率
    user_part = (
        active_df.group_by(["user_id", "part"])
        .agg(pl.col("correct").mean().alias("acc"), pl.len().alias("n"))
        .filter(pl.col("n") >= 5)
    )

    # Part間のばらつきが大きいユーザーを選ぶ
    user_spread = (
        user_part.group_by("user_id")
        .agg(
            pl.col("acc").std().alias("std_acc"),
            pl.len().alias("n_parts"),
        )
        .filter(pl.col("n_parts") >= 5)
        .sort("std_acc", descending=True)
    )

    # 上位からばらつきの大きい8人を選択
    import numpy as np
    sample_uids = user_spread.head(50).sample(8, seed=42)["user_id"].to_list()

    part_labels = {
        1: "P1\n写真描写", 2: "P2\n応答", 3: "P3\n会話",
        4: "P4\n説明文", 5: "P5\n文法", 6: "P6\n長文穴埋", 7: "P7\n読解",
    }
    parts = list(range(1, 8))

    # ヒートマップ用の行列を構築
    matrix = []
    labels_y = []
    for i, uid in enumerate(sample_uids):
        row = user_part.filter(pl.col("user_id") == uid)
        acc_map = dict(zip(row["part"].to_list(), row["acc"].to_list()))
        matrix.append([acc_map.get(p, float("nan")) for p in parts])
        labels_y.append(f"学習者 {chr(65 + i)}")

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")

    # 軸ラベル
    ax.set_xticks(range(7))
    ax.set_xticklabels([part_labels[p] for p in parts], fontsize=9)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=10)

    # セル内に数値を表示
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                color = "white" if val < 0.45 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_title(
        "同じ学習者でも、Partごとの正答率はこれだけ違う",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.colorbar(im, ax=ax, label="正答率", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig-skill-gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fig-skill-gap.png'}")


if __name__ == "__main__":
    generate_skill_gap_figure()
