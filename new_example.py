"""
Advanced analysis script for the Morty Express Challenge.

Goals:
1. Collect richer statistics per planet
2. Compare performance for 1 vs 3 Mortys
3. Prepare data for UCB (Upper Confidence Bound) strategy
4. Visualize key relationships for strategy design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from api_client import SphinxAPIClient
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
    plot_episode_summary,
)


def plot_survival_distribution(df):
    """Distribution of survival rate per planet."""
    plt.figure(figsize=(10,6))
    sns.kdeplot(data=df, x="survival_rate", hue="planet_name", fill=True, alpha=0.4)
    plt.title("Distribution de la probabilité de survie par planète")
    plt.xlabel("Taux de survie par voyage")
    plt.ylabel("Densité")
    plt.grid(alpha=0.3)
    plt.show()


def plot_efficiency(df):
    """Tradeoff between survival rate and travel time."""
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="steps_taken", y="survival_rate", hue="planet_name")
    plt.title("Trade-off entre survie et rapidité")
    plt.xlabel("Nombre de pas (steps_taken)")
    plt.ylabel("Taux de survie")
    plt.grid(alpha=0.3)
    plt.show()


def main():
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

    # Step 3: Explore planets with group sizes 1 and 3
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

    # Step 4: Compute extended statistics per planet
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

    # Step 5: Compute UCB scores
    print("\n5️⃣ Computing UCB scores...")
    total_trips = summary_df["trips"].sum()
    summary_df["UCB_value"] = summary_df["mean_survival_rate"] + \
        np.sqrt(2 * np.log(total_trips) / summary_df["trips"])

    print(summary_df)

    # Step 6: Risk analysis (from your collector)
    print("\n6️⃣ Risk trend analysis...")
    risk_analysis = collector.analyze_risk_changes(full_df)
    for planet_name, data in risk_analysis.items():
        print(f"\n{planet_name}:")
        print(f"  Early Survival Rate: {data['early_survival_rate']:.2f}%")
        print(f"  Late Survival Rate:  {data['late_survival_rate']:.2f}%")
        print(f"  Trend: {data['trend']} ({data['change']:+.2f}%)")

    # Step 7: Save enriched data
    print("\n7️⃣ Saving data for strategy training...")
    full_df.to_csv("exploration_data_ucb_ready.csv", index=False)
    summary_df.to_csv("summary_planet_stats.csv", index=False)
    print("✓ Data saved: exploration_data_ucb_ready.csv and summary_planet_stats.csv")

    # Step 8: Visualizations
    print("\n8️⃣ Generating visualizations...")

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
