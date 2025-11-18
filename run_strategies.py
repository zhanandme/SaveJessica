"""
Compare multiple strategies for the Morty Express Challenge.

This script runs several strategies in sequence, collects their performance,
and prints a summary of results.
"""

from strategy import SimpleGreedyStrategy, AdaptiveStrategy, UCBStrategy, AdaptiveSurvivalStrategy, ThompsonCyclicStrategy
from api_client import SphinxAPIClient


def evaluate_strategy(strategy_class, explore_trips=30):
    """Run a single strategy and return its final results."""
    client = SphinxAPIClient()
    strategy = strategy_class(client)

    client.start_episode()
    strategy.explore_phase(trips_per_planet=explore_trips)
    strategy.analyze_planets()

    strategy.execute_strategy()

    final_status = client.get_status()
    success_rate = (final_status['morties_on_planet_jessica'] / 1000) * 100

    return {
        "strategy": strategy_class.__name__,
        "saved": final_status["morties_on_planet_jessica"],
        "lost": final_status["morties_lost"],
        "steps": final_status["steps_taken"],
        "success_rate": success_rate,
    }

def main():
    strategies = [ThompsonCyclicStrategy, AdaptiveSurvivalStrategy]
    results = []

    for strategy_class in strategies:
        print("\n" + "="*60)
        print(f"Running strategy: {strategy_class.__name__}")
        print("="*60)
        result = evaluate_strategy(strategy_class, explore_trips=30)
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
