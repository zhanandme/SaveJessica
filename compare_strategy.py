"""
Compare multiple strategies for the Morty Express Challenge.

This script runs several strategies in sequence, collects their performance,
and prints a summary of results.
"""

from data_collector import DataCollector
from strategy import (
    SimpleGreedyStrategy,
    AdaptiveStrategy,
    run_strategy,
    UCBStrategy,
    AdaptiveSurvivalStrategy,
    ThompsonCyclicStrategy,
)
from api_client import SphinxAPIClient


def evaluate_strategy(strategy_class, client: SphinxAPIClient, collector: DataCollector, explore_trips=30):
    """Run a single strategy and return its final results, safely handling invalid requests."""
    strategy = strategy_class(client, collector)

    client.start_episode()
    strategy.explore_phase(trips_per_planet=explore_trips)
    strategy.analyze_planets()

    try:
        strategy.execute_strategy()
    except requests.exceptions.HTTPError as e:
        print(f"⚠️ HTTP Error during strategy {strategy_class.__name__}: {e}")
        print("Stopping execution of this strategy to prevent crashes.")
    except Exception as e:
        print(f"⚠️ Unexpected error during strategy {strategy_class.__name__}: {e}")
        print("Stopping execution of this strategy.")

    final_status = client.get_status()
    total_morties = final_status.get('morties_on_planet_jessica', 0) + final_status.get('morties_lost', 0)
    success_rate = (final_status['morties_on_planet_jessica'] / total_morties * 100) if total_morties > 0 else 0

    return {
        "strategy": strategy_class.__name__,
        "saved": final_status.get("morties_on_planet_jessica", 0),
        "lost": final_status.get("morties_lost", 0),
        "steps": final_status.get("steps_taken", 0),
        "success_rate": success_rate,
    }



def main():

    client = SphinxAPIClient()
    collector = DataCollector(client)


    # --- Exemple pour ThompsonCyclicStrategy ---
    thompson = ThompsonCyclicStrategy(client)  # pas besoin de collector si le constructeur n'en prend pas
    client.start_episode()

    # exploration initiale
    # si ThompsonCyclicStrategy a sa propre fonction d'exploration, l'utiliser ici
    thompson.execute_strategy(save_csv="thompson_results.csv")

    status = client.get_status()
    print("\n=== ThompsonCyclicStrategy RESULTS ===")
    print(f"Morties Saved: {status['morties_on_planet_jessica']}")
    print(f"Morties Lost: {status['morties_lost']}")
    print(f"Success Rate: {status['morties_on_planet_jessica'] / (status['morties_on_planet_jessica'] + status['morties_lost']) * 100:.2f}%")

    # --- Exemple pour AdaptiveSurvivalStrategy ---
    adaptive = AdaptiveSurvivalStrategy(client, collector)
    client.start_episode()

    # exploration initiale
    collector.explore_all_planets(trips_per_planet=30)
    adaptive.exploration_data = collector.trips_data
    adaptive.analyze_planets()

    # exécution
    adaptive.execute_strategy(morties_per_trip=2, reevaluate_every=20, save_csv="adaptive_results.csv")

    # résultats finaux
    status = client.get_status()
    print("\n=== AdaptiveSurvivalStrategy RESULTS ===")
    print(f"Morties Saved: {status['morties_on_planet_jessica']}")
    print(f"Morties Lost: {status['morties_lost']}")
    print(f"Success Rate: {status['morties_on_planet_jessica'] / (status['morties_on_planet_jessica'] + status['morties_lost']) * 100:.2f}%")


    strategies = [
    ]
    results = []

    for strategy_class in strategies:
        print("\n" + "=" * 60)
        print(f"Running strategy: {strategy_class.__name__}")
        print("=" * 60)
        result = evaluate_strategy(strategy_class, client, collector, explore_trips=30)
        results.append(result)

    print("\n=== COMPARISON RESULTS ===")
    for r in results:
        print(f"\n{r['strategy']}:")
        print(f"  Morties Saved: {r['saved']}")
        print(f"  Morties Lost: {r['lost']}")
        print(f"  Steps Taken: {r['steps']}")
        print(f"  Success Rate: {r['success_rate']:.2f}%")

    best = max(results, key=lambda x: x["saved"])
    print("\n🏆 Best strategy:", best["strategy"], f"({best['success_rate']:.2f}% success)")


if __name__ == "__main__":
    main()
