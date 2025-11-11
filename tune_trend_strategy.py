# explorer automatiquement les hyperparamètres

from strategy import TrendSwitchStrategy
from api_client import SphinxAPIClient

def run_experiment(reevaluate_every, drop_threshold, morties_per_trip):
    client = SphinxAPIClient()
    strategy = TrendSwitchStrategy(client)
    client.start_episode()

    strategy.explore_phase(trips_per_planet=30)
    strategy.analyze_planets()
    strategy.execute_strategy(
        morties_per_trip=morties_per_trip,
        reevaluate_every=reevaluate_every,
        drop_threshold=drop_threshold,
    )

    final_status = client.get_status()
    success_rate = (final_status['morties_on_planet_jessica'] / 1000) * 100
    return success_rate

def main():
    configs = [
        (25, 0.10, 3),
        (25, 0.15, 3),
        (50, 0.10, 3),
        (50, 0.05, 2),
        (75, 0.15, 3),
    ]

    results = []
    for reevaluate_every, drop_threshold, morties_per_trip in configs:
        print(f"\nTesting config: reevaluate={reevaluate_every}, drop={drop_threshold}, morties={morties_per_trip}")
        rate = run_experiment(reevaluate_every, drop_threshold, morties_per_trip)
        results.append((rate, reevaluate_every, drop_threshold, morties_per_trip))

    best = max(results, key=lambda x: x[0])
    print("\n=== BEST CONFIGURATION ===")
    print(f"Success Rate: {best[0]:.2f}%")
    print(f"Reevaluate every: {best[1]}")
    print(f"Drop threshold: {best[2]}")
    print(f"Morties per trip: {best[3]}")

if __name__ == "__main__":
    main()
