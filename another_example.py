"""
Analyze survival distributions per planet and compare
sending 1 vs 3 Mortys in the Morty Express Challenge.
"""

from api_client import SphinxAPIClient
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
)
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("SAVE JESSICA - DISTRIBUTION & GROUP SIZE ANALYSIS")
    print("=" * 60)

    # Step 1 — Initialize client and collector
    try:
        client = SphinxAPIClient()
        collector = DataCollector(client)
        print("✓ API client and DataCollector initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return

    # Step 2 — Compare 1 vs 3 Mortys
    print("\n2. Exploring all planets for group sizes = 1 and 3...")
    dfs = []
    for morty_count in [1, 3]:
        print(f"\n--- Testing with {morty_count} Morty/Morties per trip ---")
        df = collector.explore_all_planets(trips_per_planet=100, morty_count=morty_count)
        df["group_size"] = morty_count
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Step 3 — Compute distribution statistics per planet
    print("\n3. Analyzing survival distributions per planet...")
    analysis = []

    for (planet, group_size), sub_df in full_df.groupby(["planet", "group_size"]):
        survival_rates = sub_df["survived"].astype(float)
        mean_sr = survival_rates.mean()
        std_sr = survival_rates.std()
        total_saved = survival_rates.sum() * group_size
        analysis.append({
            "Planet": client.get_planet_name(int(planet)),
            "Group_Size": group_size,
            "Mean_Survival": mean_sr,
            "Std_Deviation": std_sr,
            "Trips": len(sub_df),
            "Total_Saved": total_saved
        })
        print(f"{client.get_planet_name(planet)} | {group_size} Morty(s): "
              f"mean={mean_sr:.3f}, std={std_sr:.3f}, trips={len(sub_df)}")

    analysis_df = pd.DataFrame(analysis)
    print("\nSummary of results:\n")
    print(analysis_df)

    # Step 4 — Compare 1 vs 3 Mortys on each planet
    print("\n4. Comparing performance of 1 vs 3 Mortys per planet...\n")
    for planet in analysis_df["Planet"].unique():
        subset = analysis_df[analysis_df["Planet"] == planet]
        mean_1 = subset.loc[subset["Group_Size"] == 1, "Mean_Survival"].values[0]
        mean_3 = subset.loc[subset["Group_Size"] == 3, "Mean_Survival"].values[0]
        print(f"{planet}:")
        print(f"  → Survival (1 Morty): {mean_1:.2%}")
        print(f"  → Survival (3 Mortys): {mean_3:.2%}")
        if mean_3 > mean_1:
            print("  ✅ Sending 3 Mortys is faster and equally safe!\n")
        else:
            print("  ⚠️ Safer to send 1 Morty at a time.\n")

    # Step 5 — Visualizations
    print("\n5. Generating visualizations...")
    plot_survival_by_planet(full_df)
    plot_survival_rates(full_df)
    plot_moving_average(full_df, window=10)
    plot_risk_evolution(full_df)

    # Additional visualization: bar chart summary
    plt.figure(figsize=(8, 5))
    for planet in analysis_df["Planet"].unique():
        subset = analysis_df[analysis_df["Planet"] == planet]
        plt.bar(
            [f"{planet}-1", f"{planet}-3"],
            subset["Mean_Survival"],
            width=0.4,
            label=planet
        )
    plt.ylabel("Mean survival rate")
    plt.title("Average Survival per Planet (1 vs 3 Mortys)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 6 — Save results
    print("\n6. Saving results...")
    full_df.to_csv("morty_full_data.csv", index=False)
    analysis_df.to_csv("morty_analysis_results.csv", index=False)
    print("✓ Data saved as 'morty_full_data.csv' and 'morty_analysis_results.csv'")

    # Step 7 — Final status
    print("\n7. Checking final status...")
    status = client.get_status()
    print(f"Morties in Citadel: {status['morties_in_citadel']}")
    print(f"Morties on Planet Jessica: {status['morties_on_planet_jessica']}")
    print(f"Morties Lost: {status['morties_lost']}")
    print(f"Steps Taken: {status['steps_taken']}")
    success_rate = (status['morties_on_planet_jessica'] / 1000) * 100
    print(f"Final Success Rate: {success_rate:.2f}%")

    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
