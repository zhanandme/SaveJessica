import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def savefig(name):
    plt.savefig(f"outputs/figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved → outputs/figures/{name}.png")


# -------------------------------------------------------
# 1️⃣ Heatmap simple : Survie par planète et par group_size
# -------------------------------------------------------
def plot_survival_heatmap(df):
    pivot = df.groupby(["planet_name", "group_size"])["survival_rate"].mean().unstack()

    plt.figure(figsize=(10,6))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label="Survival Rate")
    plt.xticks([0,1], ["1 Morty", "3 Mortys"])
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("Survival Rate by Planet and Group Size")
    savefig("survival_heatmap")


# -------------------------------------------------------
# 2️⃣ Bar chart : Steps moyens par planète
# -------------------------------------------------------
def plot_avg_steps(df):
    steps = df.groupby("planet_name")["steps_taken"].mean().sort_values()

    plt.figure(figsize=(10,5))
    steps.plot(kind="bar")
    plt.ylabel("Average Steps")
    plt.title("Average Steps per Planet")
    plt.grid(axis="y", alpha=0.3)
    savefig("avg_steps")


# -------------------------------------------------------
# 3️⃣ Scatter plot : trade-off steps ↔ survival
# -------------------------------------------------------
def plot_efficiency(df):
    avg = df.groupby("planet_name").agg({
        "steps_taken": "mean",
        "survival_rate": "mean"
    })

    plt.figure(figsize=(8,6))
    plt.scatter(avg["steps_taken"], avg["survival_rate"], s=120)

    for planet, row in avg.iterrows():
        plt.text(row["steps_taken"]+0.02, row["survival_rate"]+0.005, planet)

    plt.xlabel("Average Steps")
    plt.ylabel("Average Survival Rate")
    plt.title("Trade-off: Speed vs Survival")
    plt.grid(True, alpha=0.3)
    savefig("tradeoff_efficiency")


# -------------------------------------------------------
# 4️⃣ Evolution du risque : moving average simple
# -------------------------------------------------------
def plot_risk_trend(df, window=50):
    plt.figure(figsize=(12,6))

    for planet in df["planet_name"].unique():
        sub = df[df["planet_name"] == planet].sort_values("trip_id")
        mov = sub["survival_rate"].rolling(window).mean()
        plt.plot(mov, label=planet, alpha=0.7)

    plt.title(f"Survival Rate Trend (moving average window={window})")
    plt.xlabel("Trip index")
    plt.ylabel("Survival MA")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig("risk_trend")


# -------------------------------------------------------
# 5️⃣ UCB bar chart
# -------------------------------------------------------
def plot_ucb_scores(summary_df):
    ordered = summary_df.sort_values("UCB_value")

    plt.figure(figsize=(10,6))
    plt.barh(ordered["planet_name"], ordered["UCB_value"])
    plt.xlabel("UCB score")
    plt.title("UCB Ranking of Planets")
    plt.grid(axis="x", alpha=0.3)
    savefig("ucb_scores")
