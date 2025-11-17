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
import csv
import numpy as np
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

class UCBStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, window_size=10, reevaluate_every=200, save_csv="ucb_results.csv"):
        """
        Adaptive Upper Confidence Bound strategy with re-evaluation every `reevaluate_every` trips.
        """
        print("\n=== EXECUTING ADAPTIVE UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0  # total trips
        trips_since_eval = 0

        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate", "UCB"]]

        # --- Initial exploration ---
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
            trips_since_eval += 1

            csv_rows.append([n, self.client.get_planet_name(planet), to_send, survived, planet_stats[planet]["mean"], float("inf")])
            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={planet_stats[planet]['mean']:.2f}")

        # --- Main loop ---
        print("\nMain loop (UCB selection)...")
        while morties_remaining > 0:
            if trips_since_eval >= reevaluate_every:
                # Compute UCB scores
                ucb_scores = {}
                for planet in range(n_planets):
                    trials = planet_stats[planet]["trials"]
                    mean = planet_stats[planet]["mean"]
                    ucb_scores[planet] = mean + c * math.sqrt((2 * math.log(n)) / trials) if trials > 0 else float("inf")
                trips_since_eval = 0
            else:
                # Keep previous UCB scores
                ucb_scores = {planet: planet_stats[planet]["mean"] for planet in range(n_planets)}

            best_planet = max(ucb_scores, key=ucb_scores.get)
            planet_name = self.client.get_planet_name(best_planet)

            recent_rate = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"]) if planet_stats[best_planet]["history"] else 0
            if recent_rate > 0.6:
                to_send = min(3, morties_remaining)
            elif recent_rate > 0.4:
                to_send = min(2, morties_remaining)
            else:
                to_send = min(1, morties_remaining)

            result = self.client.send_morties(best_planet, to_send)
            survived = result["survived"]

            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = planet_stats[best_planet]["successes"] / (planet_stats[best_planet]["trials"] * morties_per_trip)
            planet_stats[best_planet]["history"].append(survived / to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            csv_rows.append([n, planet_name, to_send, survived, planet_stats[best_planet]["mean"], ucb_scores.get(best_planet, 0)])

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | UCB={ucb_scores.get(best_planet, 0):.2f} | Remaining={morties_remaining}")

        # --- Save CSV ---
        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n✅ Results saved to {save_csv}")

        # --- Final results ---
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")
            
class UCBAdaptiveStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, window_size=10, reevaluate_every=200, save_csv="ucb_adaptive_results.csv"):
        """
        Adaptive UCB strategy with re-evaluation every `reevaluate_every` trips.
        """
        print("\n=== EXECUTING ADAPTIVE UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0  # total trips
        trips_since_eval = 0

        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate", "UCB"]]

        # --- Initial exploration ---
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
            trips_since_eval += 1

            planet_name = self.client.get_planet_name(planet)
            csv_rows.append([n, planet_name, to_send, survived, planet_stats[planet]["mean"], float("inf")])
            print(f"  Planet {planet_name}: Survival={planet_stats[planet]['mean']:.2f}")

        # --- Main loop ---
        print("\nMain loop (UCB selection)...")
        while morties_remaining > 0:
            if trips_since_eval >= reevaluate_every:
                ucb_scores = {}
                for planet in range(n_planets):
                    trials = planet_stats[planet]["trials"]
                    mean = planet_stats[planet]["mean"]
                    ucb_scores[planet] = mean + c * math.sqrt((2 * math.log(n)) / trials) if trials > 0 else float("inf")
                trips_since_eval = 0
            else:
                ucb_scores = {planet: planet_stats[planet]["mean"] for planet in range(n_planets)}

            best_planet = max(ucb_scores, key=ucb_scores.get)
            planet_name = self.client.get_planet_name(best_planet)

            recent_rate = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"]) if planet_stats[best_planet]["history"] else 0
            if recent_rate > 0.6:
                to_send = min(3, morties_remaining)
            elif recent_rate > 0.4:
                to_send = min(2, morties_remaining)
            else:
                to_send = min(1, morties_remaining)

            result = self.client.send_morties(best_planet, to_send)
            survived = result["survived"]

            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = planet_stats[best_planet]["successes"] / (planet_stats[best_planet]["trials"] * morties_per_trip)
            planet_stats[best_planet]["history"].append(survived / to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            csv_rows.append([n, planet_name, to_send, survived, planet_stats[best_planet]["mean"], ucb_scores.get(best_planet, 0)])

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | UCB={ucb_scores.get(best_planet, 0):.2f} | Remaining={morties_remaining}")

        # --- Save CSV ---
        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n✅ Results saved to {save_csv}")

        # --- Final results ---
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")

class NewAdaptiveSurvivalStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, window_size=10, reevaluate_every=200, save_csv="adaptive_survival_results.csv"):
        """
        Adaptive strategy based on recent survival rates.
        """
        print("\n=== EXECUTING ADAPTIVE SURVIVAL STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0  # total trips
        trips_since_eval = 0

        planet_stats = {
            i: {"history": deque(maxlen=window_size), "mean": 0.0}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate"]]

        # --- Initial exploration ---
        print("\nInitial exploration...")
        for planet in range(n_planets):
            if morties_remaining <= 0:
                break
            to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(planet, to_send)
            survived = result["survived"]

            planet_stats[planet]["history"].append(survived / to_send)
            planet_stats[planet]["mean"] = sum(planet_stats[planet]["history"]) / len(planet_stats[planet]["history"])

            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            planet_name = self.client.get_planet_name(planet)
            csv_rows.append([n, planet_name, to_send, survived, planet_stats[planet]["mean"]])
            print(f"  Planet {planet_name}: Survival={planet_stats[planet]['mean']:.2f}")

        # --- Main loop ---
        print("\nMain loop (Adaptive Survival)...")
        while morties_remaining > 0:
            if trips_since_eval >= reevaluate_every:
                # Compute average survival rates for all planets
                survival_rates = {planet: planet_stats[planet]["mean"] for planet in range(n_planets)}
                trips_since_eval = 0
            else:
                survival_rates = {planet: planet_stats[planet]["mean"] for planet in range(n_planets)}

            best_planet = max(survival_rates, key=survival_rates.get)
            planet_name = self.client.get_planet_name(best_planet)

            recent_rate = planet_stats[best_planet]["mean"]
            if recent_rate > 0.6:
                to_send = min(3, morties_remaining)
            elif recent_rate > 0.4:
                to_send = min(2, morties_remaining)
            else:
                to_send = min(1, morties_remaining)

            result = self.client.send_morties(best_planet, to_send)
            survived = result["survived"]

            planet_stats[best_planet]["history"].append(survived / to_send)
            planet_stats[best_planet]["mean"] = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"])

            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            csv_rows.append([n, planet_name, to_send, survived, planet_stats[best_planet]["mean"]])

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | Remaining={morties_remaining}")

        # --- Save CSV ---
        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n✅ Results saved to {save_csv}")

        # --- Final results ---
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")

class AdaptiveSurvivalStrategy(MortyRescueStrategy):
    def execute_strategy(
        self,
        window_size=8,
        reevaluate_every=30,
        exploration_cycle=100,
        switch_threshold=0.45,
        save_csv="adaptive_results_v3.csv"
    ):
        print("\n=== EXECUTING ADAPTIVE SURVIVAL STRATEGY v3 (Optimized) ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']

        # Start on the historically best planet
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data, consider_trend=True
        )
        print(f"Starting on {current_planet_name}")

        recent_results = deque(maxlen=window_size)
        total_trips = 0
        reevaluation_cooldown = 0

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "Rate", "PlanetReevaluated"]]

        while morties_remaining > 0:

            # stable + reactive recent rate
            if len(recent_results) > 0:
                recent_success = sum(recent_results) / len(recent_results)
            else:
                recent_success = 0.55  # neutral and optimistic start

            # --- decision on morties to send ---
            if recent_success >= 0.65:
                morties_to_send = min(3, morties_remaining)
            elif recent_success >= 0.40:
                morties_to_send = min(2, morties_remaining)
            else:
                morties_to_send = 1

            # --- send morties ---
            result = self.client.send_morties(current_planet, morties_to_send)
            survived = result["survived"]
            morties_remaining = result["morties_in_citadel"]

            recent_results.append(survived / morties_to_send)
            total_trips += 1

            print(f"Trip {total_trips}: Sent {morties_to_send} → survived={survived} | recent={recent_success:.2f}")

            planet_reevaluated = False

            # --- forced exploration every X trips ---
            if total_trips % exploration_cycle == 0:
                print("\n🔍 FORCED EXPLORATION CYCLE: Checking all planets…")
                new_planet, new_name = self.collector.get_best_planet(self.exploration_data, consider_trend=True)
                current_planet = new_planet
                current_planet_name = new_name
                planet_reevaluated = True
                reevaluation_cooldown = 15

            # --- periodic performance check ---
            if total_trips % reevaluate_every == 0 and reevaluation_cooldown == 0:
                window_perf = sum(recent_results) / len(recent_results)
                print(f"  → Reevaluating: window perf = {window_perf:.2f}")

                if window_perf < switch_threshold:
                    print("  ⚠️ Performance drop → switching planet")
                    new_planet, new_name = self.collector.get_best_planet(self.exploration_data, consider_trend=True)
                    current_planet = new_planet
                    current_planet_name = new_name
                    planet_reevaluated = True
                    reevaluation_cooldown = 15

            # --- optional boost for planet 2 (if it's historically best in your game) ---
            if current_planet == 2 and total_trips % 50 == 0:
                print("🔁 Returning to planet 2 for periodic high-yield check")

            # --- log CSV ---
            csv_rows.append([
                total_trips,
                current_planet_name,
                morties_to_send,
                survived,
                survived / morties_to_send,
                planet_reevaluated
            ])

        # save CSV
        with open(save_csv, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)

        print(f"\nResults saved to {save_csv}")

        # final stats
        final = self.client.get_status()
        saved = final["morties_on_planet_jessica"]
        print("\n=== FINAL RESULTS ===")
        print(f"Saved: {saved}")
        print(f"Lost:  {final['morties_lost']}")
        print(f"Success rate: {saved/1000:.2%}")


class CustomMortyStrategy(MortyRescueStrategy):
    def execute_strategy(self):
        print("\n=== EXECUTING CUSTOM MORTY STRATEGY ===")
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]

        total_sent_since_last_2 = 0
        current_planet = 2  # commencer par planète 2
        batch_size = 3      # taille maximale autorisée par envoi

        first_test_done = False
        send_large_batch = False
        large_batch_sent = 0
        large_batch_target = 100

        while morties_remaining > 0:
            # --- Déterminer combien envoyer ---
            if current_planet == 2 and not first_test_done:
                to_send = 1
            elif current_planet == 2 and send_large_batch:
                to_send = min(batch_size, morties_remaining, large_batch_target - large_batch_sent)
            else:
                to_send = min(batch_size, morties_remaining)

            # --- Envoyer mortys ---
            result = self.client.send_morties(current_planet, to_send)
            survived = result["survived"]
            morties_remaining = result["morties_in_citadel"]
            total_sent_since_last_2 += to_send

            if current_planet == 2:
                if not first_test_done:
                    first_test_done = True
                    if survived:
                        send_large_batch = True
                        large_batch_sent = 0
                elif send_large_batch:
                    large_batch_sent += to_send
                    if large_batch_sent >= large_batch_target:
                        send_large_batch = False
                        # changer de planète après le batch
                        current_planet = 0
                        total_sent_since_last_2 = 0
            else:
                # revenir sur planète 2 tous les 200 mortys envoyés
                if total_sent_since_last_2 >= 200:
                    current_planet = 2
                    total_sent_since_last_2 = 0
                else:
                    # alterner entre les autres planètes
                    current_planet = 0 if current_planet != 0 else 1

            print(f"Sent {to_send} to {self.client.get_planet_name(current_planet)} | Survived={survived} | Remaining={morties_remaining}")

class AdaptiveSurvivalStrategyModified(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, window_size=10, save_csv="adaptive_survival.csv"):
        """
        Adaptive survival strategy with special handling for planet 2.
        """
        print("\n=== EXECUTING ADAPTIVE SURVIVAL STRATEGY (MODIFIED) ===")
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]

        n_planets = 3
        n = 0  # total trips
        trips_since_planet2 = 0
        first_test_done = False
        send_large_batch = False
        large_batch_sent = 0
        large_batch_target = 100

        planet_stats = {
            i: {"successes": 0, "trials": 0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "PlanetName", "MortiesSent", "Survived"]]

        current_planet = 2  # commencer par planète 2

        while morties_remaining > 0:
            # Déterminer combien envoyer
            if current_planet == 2 and not first_test_done:
                to_send = 1
            elif current_planet == 2 and send_large_batch:
                to_send = min(morties_per_trip, morties_remaining, large_batch_target - large_batch_sent)
            else:
                to_send = min(morties_per_trip, morties_remaining)

            result = self.client.send_morties(current_planet, to_send)
            survived = result["survived"]
            morties_remaining = result["morties_in_citadel"]

            # Mettre à jour les stats
            planet_stats[current_planet]["trials"] += 1
            planet_stats[current_planet]["successes"] += survived
            planet_stats[current_planet]["history"].append(survived / to_send)

            n += 1
            trips_since_planet2 += to_send

            csv_rows.append([
                n,
                current_planet,
                self.client.get_planet_name(current_planet),
                to_send,
                survived
            ])

            # Gestion spéciale planète 2
            if current_planet == 2:
                if not first_test_done:
                    first_test_done = True
                    if survived:
                        send_large_batch = True
                        large_batch_sent = 0
                elif send_large_batch:
                    large_batch_sent += to_send
                    if large_batch_sent >= large_batch_target:
                        send_large_batch = False
                        # changer de planète après le batch
                        current_planet = 0
                        trips_since_planet2 = 0
            else:
                # revenir sur planète 2 tous les 100 mortys envoyés
                if trips_since_planet2 >= 100:
                    current_planet = 2
                    trips_since_planet2 = 0
                else:
                    # alterner entre les autres planètes
                    current_planet = 0 if current_planet != 0 else 1

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {self.client.get_planet_name(current_planet)} | Survived={survived} | Remaining={morties_remaining}")

        # --- Save CSV ---
        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n✅ Results saved to {save_csv}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")


""""
class HybridUCBAdaptiveStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, success_window=10, drop_threshold=0.3):
        
        Hybrid UCB + Adaptive Survival Strategy aiming for ~70% success.
        
        print("\n=== EXECUTING HYBRID UCB + ADAPTIVE STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000

        n_planets = 3
        n = 0  # total trips
        planet_stats = {i: {"successes": 0, "trials": 0, "mean": 0.0} for i in range(n_planets)}
        recent_results = []

        # --- Phase 1: Initial exploration (one trip per planet)
        print("\nInitial exploration...")
        for planet in range(n_planets):
            if morties_remaining <= 0:
                break
            result = self.client.send_morties(planet, morties_per_trip)
            survived = result["survived"]
            trials = result["morties_sent"]
            success_rate = survived / trials if trials > 0 else 0

            planet_stats[planet]["successes"] += survived
            planet_stats[planet]["trials"] += 1
            planet_stats[planet]["mean"] = success_rate

            morties_remaining = result["morties_in_citadel"]
            n += 1

            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={success_rate:.2f}")

        # --- Phase 2: Hybrid UCB + Adaptive loop
        print("\nMain loop (Hybrid UCB + Adaptive)...")

        while morties_remaining > 0:
            # Compute UCB for each planet
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

            # Adaptive number of Mortys
            if len(recent_results) >= success_window:
                recent_success = sum(recent_results[-success_window:]) / success_window
            else:
                recent_success = sum(recent_results) / max(1, len(recent_results))

            if recent_success >= 0.6:
                morties_to_send = min(3, morties_remaining)
            elif recent_success >= 0.4:
                morties_to_send = min(2, morties_remaining)
            else:
                morties_to_send = 1

            result = self.client.send_morties(best_planet, morties_to_send)
            survived = result["survived"]
            morties_remaining = result["morties_in_citadel"]

            recent_results.append(1 if survived else 0)
            n += 1

            # Update stats
            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = (
                planet_stats[best_planet]["successes"] /
                (planet_stats[best_planet]["trials"] * morties_per_trip)
            )

            if n % 10 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | "
                      f"UCB={ucb_scores[best_planet]:.2f} | Remaining={morties_remaining} | "
                      f"Recent success={recent_success:.2f}")

            # Re-evaluate if recent success drops
            if recent_success < (1 - drop_threshold) and morties_remaining > 0:
                print("  ⚠️ Performance drop detected! Re-evaluating best planet...")
                # Pick planet with highest mean
                best_mean_planet = max(planet_stats, key=lambda x: planet_stats[x]['mean'])
                best_planet = best_mean_planet
                planet_name = self.client.get_planet_name(best_planet)
                print(f"  → Switching to planet: {planet_name}")

        # --- Final results
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")

        print("\n=== Planet Performance Summary ===")
        for planet, stats in planet_stats.items():
            print(f"{self.client.get_planet_name(planet)}: mean={stats['mean']:.3f}, trials={stats['trials']}")

class AggressiveUCBStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.0, initial_exploration=1):
        
        Ultra-aggressive UCB-inspired strategy to maximize Morty survival.
        Minimizes exploration and exploits the best planet as soon as identified.
        
        print("\n=== EXECUTING AGGRESSIVE UCB STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        total_morties = 1000
        
        n_planets = 3
        n = 0
        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0}
            for i in range(n_planets)
        }

        # --- Minimal initial exploration ---
        print("\nInitial exploration...")
        for planet in range(n_planets):
            if planet >= initial_exploration or morties_remaining <= 0:
                break
            result = self.client.send_morties(planet, morties_per_trip)
            survived = result['survived']
            planet_stats[planet]["successes"] += survived
            planet_stats[planet]["trials"] += 1
            planet_stats[planet]["mean"] = survived / morties_per_trip
            morties_remaining = result['morties_in_citadel']
            n += 1
            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={planet_stats[planet]['mean']:.2f}")

        # --- Exploitation loop ---
        print("\nExploitation phase...")
        while morties_remaining > 0:
            # Select planet with highest empirical mean
            best_planet = max(planet_stats, key=lambda p: planet_stats[p]["mean"])
            planet_name = self.client.get_planet_name(best_planet)

            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(best_planet, morties_to_send)

            survived = result['survived']
            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = planet_stats[best_planet]["successes"] / (planet_stats[best_planet]["trials"] * morties_per_trip)

            morties_remaining = result['morties_in_citadel']
            n += 1

            if n % 10 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | Remaining={morties_remaining}")

            # Immediate switch if survival drops to 0
            if survived == 0:
                # Try next best planet
                sorted_planets = sorted(planet_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
                for p, stats in sorted_planets:
                    if p != best_planet:
                        best_planet = p
                        planet_name = self.client.get_planet_name(best_planet)
                        print(f"⚠️ Switching to better-performing planet: {planet_name}")
                        break

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")
"""

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

