"""
Advanced analysis script for the Morty Express Challenge.

Goals:
1. Collect richer statistics per planet
2. Compare performance for 1 vs 3 Mortys
3. Prepare data for UCB (Upper Confidence Bound) strategy
4. Visualize key relationships for strategy design
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from api_client import SphinxAPIClient
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
    plot_episode_summary,
)

# -------------------------------------------------------
# Folder creation
# -------------------------------------------------------
def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/data", exist_ok=True)

# -------------------------------------------------------
# Figure saving helper
# -------------------------------------------------------
def savefig(name):
    path = f"outputs/figures/{name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 figure saved → {path}")

# -------------------------------------------------------
# 1) Distribution survie par planète
# -------------------------------------------------------
def plot_survival_distribution(df):
    plt.figure(figsize=(10,6))

    for planet in df["planet_name"].unique():
        sub = df[df["planet_name"] == planet]
        plt.hist(sub["survival_rate"], bins=20, alpha=0.4, label=planet)

    plt.title("Distribution du taux de survie par planète")
    plt.xlabel("Taux de survie")
    plt.ylabel("Fréquence")
    plt.legend()
    savefig("distribution_survie")

# -------------------------------------------------------
# 2) Tradeoff survie ↔ rapidité
# -------------------------------------------------------
def plot_efficiency(df):
    plt.figure(figsize=(8,6))

    for planet in df["planet_name"].unique():
        sub = df[df["planet_name"] == planet]
        plt.scatter(sub["steps_taken"], sub["survival_rate"], alpha=0.5, label=planet)

    plt.title("Trade-off entre survie et rapidité")
    plt.xlabel("Nombre de steps")
    plt.ylabel("Taux de survie")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("efficiency")

# -------------------------------------------------------
# Save CSVs
# -------------------------------------------------------
def save_dataframes(full_df, summary_df):
    full_df.to_csv("outputs/data/exploration_data_ucb_ready.csv", index=False)
    summary_df.to_csv("outputs/data/summary_planet_stats.csv", index=False)
    print("📁 CSV saved in outputs/data/")


# -------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------
def main():

    ensure_dirs()

    print("="*70)
    print("MORTY EXPRESS CHALLENGE - ADVANCED PLANET ANALYSIS (UCB READY)")
    print("="*70)

    # Step 1: Initialize client
    print("\n1️⃣ Initializing API client...")
    client = SphinxAPIClient()
    print("✓ Client initialized.")

    # Step 2: Start a new episode
    print("\n2️⃣ Starting a new episode...")
    result = client.start_episode()
    print(f"✓ Episode started — Morties in Citadel: {result['morties_in_citadel']}")

    # Step 3: Explore planets
    print("\n3️⃣ Exploring planets with group sizes = 1 and 3...")
    collector = DataCollector(client)
    dfs = []

    for group_size in [1, 3]:
        print(f"\n--- Testing group size = {group_size} ---")
        df = collector.explore_all_planets(trips_per_planet=150, morty_count=group_size)
        df["group_size"] = group_size
        df["survival_rate"] = df["survived"] / df["morties_sent"]
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print("\n✅ Data collected!")
    print(full_df.head())

    # Step 4: Statistics per planet
    print("\n4️⃣ Computing statistics per planet...")

    summary_rows = []
    for (planet_idx, planet_name), sub in full_df.groupby(["planet", "planet_name"]):
        trip_rates = sub["survived"] / sub["morties_sent"]
        mean_sr = trip_rates.mean()
        var_sr = trip_rates.var(ddof=0)
        std_sr = np.sqrt(var_sr)
        avg_steps = sub["steps_taken"].mean()
        efficiency = mean_sr / avg_steps if avg_steps > 0 else 0

        summary_rows.append({
            "planet": int(planet_idx),
            "planet_name": planet_name,
            "trips": len(sub),
            "mean_survival_rate": mean_sr,
            "std_survival_rate": std_sr,
            "var_survival_rate": var_sr,
            "avg_steps": avg_steps,
            "efficiency": efficiency,
        })

        print(f"{planet_name} (index {planet_idx}) → mean={mean_sr:.3f}, std={std_sr:.3f}, efficiency={efficiency:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # Step 5: UCB values
    print("\n5️⃣ Computing UCB scores...")
    total_trips = summary_df["trips"].sum()
    summary_df["UCB_value"] = summary_df["mean_survival_rate"] + \
        np.sqrt(2 * np.log(total_trips) / summary_df["trips"])

    print(summary_df)

    # Step 6: Risk evolution
    print("\n6️⃣ Risk trend analysis...")
    risk_analysis = collector.analyze_risk_changes(full_df)
    for planet_name, data in risk_analysis.items():
        print(f"\n{planet_name}:")
        print(f"  Early Survival Rate: {data['early_survival_rate']:.2f}%")
        print(f"  Late Survival Rate:  {data['late_survival_rate']:.2f}%")
        print(f"  Trend: {data['trend']} ({data['change']:+.2f}%)")

    # Step 7: Save CSV
    print("\n7️⃣ Saving data for strategy training...")
    save_dataframes(full_df, summary_df)

    # Step 8: Visualizations
    print("\n8️⃣ Saving visualizations...")
    plot_survival_distribution(full_df)
    plot_efficiency(full_df)

    print("\n- Survival rates over time...")
    plot_survival_rates(full_df)

    print("- Planet comparison...")
    plot_survival_by_planet(full_df)

    print("- Moving averages...")
    plot_moving_average(full_df, window=10)

    print("- Risk evolution...")
    plot_risk_evolution(full_df)

    print("- Episode summary...")
    plot_episode_summary(full_df)

    print("\n✅ Analysis complete! Data is ready for UCB / Thompson Sampling strategies.")


if __name__ == "__main__":
    main()
