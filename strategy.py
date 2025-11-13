"""
Strategy template for the Morty Express Challenge.

This file provides a template for implementing your own strategy
to maximize the number of Morties saved.

The challenge: The survival probability of each planet changes over time
(based on the number of trips taken). Your strategy should adapt to these
changing conditions.
"""

from abc import ABC, abstractmethod
from api_client import SphinxAPIClient
from data_collector import DataCollector
import pandas as pd
import math
import random
from collections import deque


class MortyRescueStrategy(ABC):
    """Abstract base class for implementing rescue strategies."""
    
    def __init__(self, client: SphinxAPIClient):
        """
        Initialize the strategy.
        
        Args:
            client: SphinxAPIClient instance
        """
        self.client = client
        self.collector = DataCollector(client)
        self.exploration_data = []
    
    def explore_phase(self, trips_per_planet: int = 30) -> pd.DataFrame:
        """
        Initial exploration phase to understand planet behaviors.
        
        Args:
            trips_per_planet: Number of trips to send to each planet
            
        Returns:
            DataFrame with exploration data
        """
        print("\n=== EXPLORATION PHASE ===")
        df = self.collector.explore_all_planets(
            trips_per_planet=trips_per_planet,
            morty_count=1  # Send 1 Morty at a time during exploration
        )
        self.exploration_data = df
        return df
    
    def analyze_planets(self) -> dict:
        """
        Analyze planet data to determine characteristics.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.exploration_data) == 0:
            raise ValueError("No exploration data available. Run explore_phase() first.")
        
        return self.collector.analyze_risk_changes(self.exploration_data)
    
    @abstractmethod
    def execute_strategy(self):
        """
        Execute the main rescue strategy.
        Must be implemented by subclasses.
        """
        pass


# Choisit la planète avec le meilleur taux de survie moyen pendant l’exploration, et y envoie tous les Morties.
# Ne s’adapte pas aux changements (si la planète devient dangereuse)
class SimpleGreedyStrategy(MortyRescueStrategy):
    """
    Simple greedy strategy: always pick the planet with highest recent success.
    """
    
    def execute_strategy(self, morties_per_trip: int = 3):
        """
        Execute the greedy strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
        """
        print("\n=== EXECUTING GREEDY STRATEGY ===")
        
        # Get current status
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Determine best planet from exploration
        best_planet, best_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Best planet identified: {best_planet_name}")
        print(f"Sending all remaining Morties to {best_planet_name}...")
        
        trips_made = 0
        
        while morties_remaining > 0:
            # Determine how many to send
            morties_to_send = min(morties_per_trip, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(best_planet, morties_to_send)
            
            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


# Re-évalue tous les N voyages (reevaluate_every), mais ne change pas encore réellement de planète.
# Le changement de planète n’est pas implémenté.
class AdaptiveStrategy(MortyRescueStrategy):
    """
    Adaptive strategy: continuously monitor and switch planets if needed.
    """
    
    def execute_strategy(
        self,
        morties_per_trip: int = 3,
        reevaluate_every: int = 50
    ):
        """
        Execute the adaptive strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
            reevaluate_every: Re-evaluate best planet every N trips
        """
        print("\n=== EXECUTING ADAPTIVE STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Initial best planet
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Starting with planet: {current_planet_name}")
        
        trips_since_evaluation = 0
        total_trips = 0
        recent_results = []
        
        while morties_remaining > 0:
            # Send Morties
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(current_planet, morties_to_send)
            
            # Track recent results
            recent_results.append({
                'planet': current_planet,
                'survived': result['survived']
            })
            
            morties_remaining = result['morties_in_citadel']
            trips_since_evaluation += 1
            total_trips += 1
            
            # Re-evaluate strategy periodically
            if trips_since_evaluation >= reevaluate_every and morties_remaining > 0:
                # Check if we should switch planets
                recent_success_rate = sum(
                    r['survived'] for r in recent_results[-reevaluate_every:]
                ) / min(len(recent_results), reevaluate_every)
                
                print(f"\n  Re-evaluating at trip {total_trips}...")
                print(f"  Current planet: {current_planet_name}")
                print(f"  Recent success rate: {recent_success_rate*100:.2f}%")
                
                # TODO: Implement logic to potentially switch planets
                # For now, we stick with the same planet
                
                trips_since_evaluation = 0
            
            if total_trips % 50 == 0:
                print(f"  Progress: {total_trips} trips, "
                      f"{result['morties_on_planet_jessica']} saved")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")




# Change de planète si la survie moyenne des 50 derniers voyages chute de plus de X %.
class TrendSwitchStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=2, reevaluate_every=50, drop_threshold=0.2):
        print("\n=== EXECUTING TREND-SWITCH STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        print(f"Starting on planet: {current_planet_name}")

        recent_results = []
        total_trips = 0

        while morties_remaining > 0:
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(current_planet, morties_to_send)
            morties_remaining = result['morties_in_citadel']

            recent_results.append(result['survived'])
            total_trips += 1

            if total_trips % reevaluate_every == 0 and morties_remaining > 0:
                recent_success = sum(recent_results[-reevaluate_every:]) / reevaluate_every
                print(f"Trip {total_trips}: success rate {recent_success*100:.2f}%")

                # Si la performance baisse trop, on re-choisit une planète
                if recent_success < (1 - drop_threshold):
                    print("  ⚠️ Performance drop detected! Re-evaluating best planet...")
                    current_planet, current_planet_name = self.collector.get_best_planet(
                        self.exploration_data,
                        consider_trend=True
                    )
                    print(f"  → Switching to planet: {current_planet_name}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")
            
class UCBStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, window_size=10):
        """
        Adaptive Upper Confidence Bound strategy for Morty Express Challenge.
        Adjusts number of Mortys sent based on recent survival trend.
        
        Args:
            morties_per_trip: max number of Mortys per trip (1–3)
            c: exploration factor (higher -> more exploration)
            window_size: number of recent trips to consider for trend
        """
        print("\n=== EXECUTING ADAPTIVE UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0  # total trips

        # Planet stats: keep successes, trials, mean, and recent history
        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        # --- Initial exploration: one trip per planet
        print("\nInitial exploration...")
        for planet in range(n_planets):
            if morties_remaining <= 0:
                break

            to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(planet, to_send)
            survived = result["survived"]

            planet_stats[planet]["successes"] += survived
            planet_stats[planet]["trials"] += 1
            planet_stats[planet]["mean"] = survived / to_send
            planet_stats[planet]["history"].append(survived / to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1

            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={planet_stats[planet]['mean']:.2f}")

        # --- Main UCB loop
        print("\nMain loop (UCB selection)...")
        while morties_remaining > 0:
            ucb_scores = {}
            for planet in range(n_planets):
                trials = planet_stats[planet]["trials"]
                mean = planet_stats[planet]["mean"]
                if trials == 0:
                    ucb_scores[planet] = float("inf")
                else:
                    ucb_scores[planet] = mean + c * math.sqrt((2 * math.log(n)) / trials)

            # Select planet with highest UCB
            best_planet = max(ucb_scores, key=ucb_scores.get)
            planet_name = self.client.get_planet_name(best_planet)

            # Adjust Mortys per trip based on recent trend
            recent_rate = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"]) if planet_stats[best_planet]["history"] else 0
            if recent_rate > 0.6:
                to_send = min(3, morties_remaining)
            elif recent_rate > 0.4:
                to_send = min(2, morties_remaining)
            else:
                to_send = min(1, morties_remaining)

            # Send Mortys
            result = self.client.send_morties(best_planet, to_send)
            survived = result["survived"]

            # Update stats
            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = planet_stats[best_planet]["successes"] / (planet_stats[best_planet]["trials"] * morties_per_trip)
            planet_stats[best_planet]["history"].append(survived / to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | UCB={ucb_scores[best_planet]:.2f} | Remaining={morties_remaining}")

        # --- Final results
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")

        print("\n=== Planet Performance Summary ===")
        for planet, stats in planet_stats.items():
            print(f"{self.client.get_planet_name(planet)}: mean={stats['mean']:.3f}, trials={stats['trials']}")

            

class AdaptiveSurvivalStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=2, reevaluate_every=20, drop_threshold=0.4):
        print("\n=== EXECUTING ADAPTIVE SURVIVAL STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        print(f"Starting on planet: {current_planet_name}")

        recent_results = []
        total_trips = 0
        consecutive_losses = 0

        while morties_remaining > 0:
            # Adapter le nombre de Mortys à envoyer
            if len(recent_results) >= 5:
                recent_success = sum(recent_results[-5:]) / 5
            else:
                recent_success = sum(recent_results) / max(1, len(recent_results))
            
            if recent_success >= 0.7:
                morties_to_send = min(3, morties_remaining)
            elif recent_success >= 0.4:
                morties_to_send = min(2, morties_remaining)
            else:
                morties_to_send = 1

            result = self.client.send_morties(current_planet, morties_to_send)
            morties_remaining = result['morties_in_citadel']
            survived = result['survived']

            recent_results.append(1 if survived else 0)
            total_trips += 1

            print(f"Trip {total_trips}: Sent {morties_to_send}, Survived: {survived}")

            # Compter les pertes consécutives
            if not survived:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            # Si trop de pertes consécutives, réduire temporairement les Mortys
            if consecutive_losses >= 3:
                morties_to_send = 1

            # Réévaluer la planète tous les 'reevaluate_every' trips
            if total_trips % reevaluate_every == 0 and morties_remaining > 0:
                recent_success_window = sum(recent_results[-reevaluate_every:]) / reevaluate_every
                print(f"Trip {total_trips}: Recent success rate {recent_success_window*100:.2f}%")
                
                if recent_success_window < (1 - drop_threshold):
                    print("  ⚠️ Performance drop detected! Re-evaluating best planet...")
                    current_planet, current_planet_name = self.collector.get_best_planet(
                        self.exploration_data,
                        consider_trend=True
                    )
                    print(f"  → Switching to planet: {current_planet_name}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")




def run_strategy(strategy_class, explore_trips: int = 30):
    """
    Run a complete strategy from exploration to execution.
    
    Args:
        strategy_class: Strategy class to use
        explore_trips: Number of exploration trips per planet
    """
    # Initialize client and strategy
    client = SphinxAPIClient()
    strategy = strategy_class(client)
    
    # Start new episode
    print("Starting new episode...")
    client.start_episode()
    
    # Exploration phase
    strategy.explore_phase(trips_per_planet=explore_trips)
    
    # Analyze results
    analysis = strategy.analyze_planets()
    print("\nPlanet Analysis:")
    for planet_name, data in analysis.items():
        print(f"  {planet_name}: {data['overall_survival_rate']:.2f}% "
              f"({data['trend']})")
    
    # Execute strategy
    strategy.execute_strategy()


if __name__ == "__main__":
    print("Morty Express Challenge - Strategy Module")
    print("="*60)
    
    print("\nAvailable strategies:")
    print("1. SimpleGreedyStrategy - Pick best planet and stick with it")
    print("2. AdaptiveStrategy - Monitor and adapt to changing conditions")
    
    print("\nExample usage:")
    print("  run_strategy(SimpleGreedyStrategy, explore_trips=30)")
    print("  run_strategy(AdaptiveStrategy, explore_trips=30)")
    
    print("\nTo create your own strategy:")
    print("1. Subclass MortyRescueStrategy")
    print("2. Implement the execute_strategy() method")
    print("3. Use self.client to interact with the API")
    print("4. Use self.collector to analyze data")
    
    # Uncomment to run:
    # run_strategy(AdaptiveStrategy, explore_trips=30)

