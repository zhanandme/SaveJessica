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
from datetime import datetime
from typing import Dict, List, Tuple




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

class NewAdaptiveSurvivalStrategy(MortyRescueStrategy):
    """
    Adaptive survival strategy: send Morties based on past survival statistics
    and reevaluate the planet periodically if recent survival rates are low.
    """
    
    def execute_strategy(
        self,
        morties_per_trip: int = 3,
        reevaluate_every: int = 50,
        min_survival_threshold: float = 0.3
    ):
        print("\n=== EXECUTING ADAPTIVE SURVIVAL STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        print(f"Starting with {morties_remaining} Morties in Citadel")

        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        ) #Note: we choose first best planet
        print(f"Starting with planet: {current_planet_name}")
        
        trips_since_evaluation = 0
        total_trips = 0
        recent_results = []
        history = {1: [], 2: [], 3: []}  #Survival history by morties sent

        while morties_remaining > 0:
            survival_scores = {}
            for m in range(1, morties_per_trip+1):
                past = history[m]
                mean = sum(past)/len(past) if past else 0
                var = sum((x - mean)**2 for x in past)/len(past) if past else 0
                survival_scores[m] = mean - 0.5 * var  # Penalize high variance

            if all(v == 0 for v in survival_scores.values()):
                morties_to_send = min(morties_per_trip, morties_remaining)
            else:
                morties_to_send = max(survival_scores, key=survival_scores.get)
                morties_to_send = min(morties_to_send, morties_remaining)

            #Send Morties
            result = self.client.send_morties(current_planet, morties_to_send)

            recent_results.append(result['survived'])
            history[morties_to_send].append(result['survived'])
            morties_remaining = result['morties_in_citadel']
            
            trips_since_evaluation += 1
            total_trips += 1
            
            if trips_since_evaluation >= reevaluate_every and morties_remaining > 0:
                recent_success_rate = sum(recent_results[-reevaluate_every:]) / min(len(recent_results), reevaluate_every)
                print(f"\n  Re-evaluating at trip {total_trips}...")
                print(f"  Current planet: {current_planet_name}")
                print(f"  Recent survival rate: {recent_success_rate*100:.2f}%")

                if recent_success_rate < min_survival_threshold: #Note: better to switch planet when survival rate gets too low
                    new_planet, new_planet_name = self.collector.get_best_planet(
                        self.exploration_data,
                        consider_trend=True
                    )
                    if new_planet != current_planet:
                        print(f"  Switching planet to: {new_planet_name}")
                        current_planet = new_planet
                        current_planet_name = new_planet_name
                
                trips_since_evaluation = 0

            if total_trips % 50 == 0:
                saved = self.client.get_status()['morties_on_planet_jessica']
                print(f"  Progress: {total_trips} trips, {saved} Morties saved")
        
        #Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

class UCBStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, window_size=10, reevaluate_every=200, save_csv="ucb_results.csv"):
        """
        Adaptive Upper Confidence Bound strategy with reevaluation every `reevaluate_every` trips.
        """
        print("\n=== EXECUTING ADAPTIVE UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0
        trips_since_eval = 0

        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate", "UCB"]]

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

        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nResults saved in a csv file {save_csv}")

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

class OptimizedMortyStrategy(MortyRescueStrategy):
    """
        Fusion of UCB and Adaptive Survival strategy:
        - UCB for planet selection
        - Adaptive number of Mortys per trip based on recent success
    """
    def execute_strategy(
        self,
        morties_per_trip=3,
        c=1.5,
        window_size=10,
        reevaluate_every=50,
        save_csv="optimized_results.csv"
    ):       
        print("\n=== EXECUTING OPTIMIZED STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0

        planet_stats = {
            i: {
                "successes": 0,
                "trials": 0,
                "history": deque(maxlen=window_size)
            }
            for i in range(n_planets)
        }
        print("\nInitial exploration...")
        for planet in range(n_planets):
            if morties_remaining <= 0:
                break
            to_send = 1
            if planet == 1:
                to_send = min(2, morties_remaining)

            result = self.client.send_morties(planet, to_send)
            survived = result["survived"]

            planet_stats[planet]["successes"] += survived
            planet_stats[planet]["trials"] += 1
            planet_stats[planet]["history"].append(survived / to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1

            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={survived/to_send:.2f}")

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate", "UCB"]]

        while morties_remaining > 0:
            ucb_scores = {}
            for planet in range(n_planets):
                trials = planet_stats[planet]["trials"]
                mean = sum(planet_stats[planet]["history"]) / len(planet_stats[planet]["history"]) if planet_stats[planet]["history"] else 0
                ucb_scores[planet] = mean + c * math.sqrt((2 * math.log(n)) / trials) if trials > 0 else float("inf")

            best_planet = max(ucb_scores, key=ucb_scores.get) #Note: choses best planet according to UCB
            planet_name = self.client.get_planet_name(best_planet)

            recent_rate = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"]) if planet_stats[best_planet]["history"] else 0
            morties_to_send = max(1, int(morties_per_trip * recent_rate + 0.5))
            morties_to_send = min(morties_to_send, morties_remaining)

            result = self.client.send_morties(best_planet, morties_to_send)
            survived = result["survived"]

            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["history"].append(survived / morties_to_send)

            morties_remaining = result["morties_in_citadel"]
            n += 1

            csv_rows.append([n, planet_name, morties_to_send, survived, sum(planet_stats[best_planet]["history"])/len(planet_stats[best_planet]["history"]), ucb_scores.get(best_planet, 0)])

            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {morties_to_send} to {planet_name} | recent_rate={recent_rate:.2f} | UCB={ucb_scores[best_planet]:.2f} | Remaining={morties_remaining}")

        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nResults saved in a csv file {save_csv}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")


class ThompsonCyclicStrategy(MortyRescueStrategy):
    """
    Thompson Sampling strategy with cyclic phases for each planet.
    """
    PLANET_PERIODS = {0: 10, 1: 20, 2: 200}
    def __init__(self, client):
        super().__init__(client)
        #Parameters that we use for each phase of each planet are a=success+1 and b=failure+1 
        self.phase_stats: Dict[int, List[Tuple[int,int]]] = {
            planet: [(1,1) for _ in range(period)]
            for planet, period in self.PLANET_PERIODS.items()
        }

    def _current_phase(self, planet_idx: int, trip_num: int) -> int:
        return trip_num % self.PLANET_PERIODS[planet_idx]

    def _choose_planet(self, trip_num: int) -> int:
        sampled_probs = {}
        for planet, phases in self.phase_stats.items():
            phase = self._current_phase(planet, trip_num)
            a, b = phases[phase]
            sampled_probs[planet] = random.betavariate(a, b)
        best_planet = max(sampled_probs, key=sampled_probs.get)
        return best_planet

    def _choose_morties_count(self, planet_idx: int, trip_num: int) -> int:
        phase = self._current_phase(planet_idx, trip_num)
        a, b = self.phase_stats[planet_idx][phase]
        prob = a / (a + b)
        if prob > 0.7:
            return 3
        elif prob > 0.4:
            return 2
        else:
            return 1

    def execute_strategy(self, save_csv="thompson_cyclic_results.csv"):
        print("\n=== EXECUTING THOMPSON CYCLIC STRATEGY ===")
        status = self.client.get_status()
        morties_remaining = status.get('morties_in_citadel', 0)
        trip_num = 0

        csv_rows = [["Trip", "Planet", "MortiesSent", "MortiesSaved",
                     "SurvivedBool", "Phase", "Timestamp"]]

        while morties_remaining > 0:
            planet_idx = self._choose_planet(trip_num)
            phase = self._current_phase(planet_idx, trip_num)
            morties_to_send = self._choose_morties_count(planet_idx, trip_num)
            morties_to_send = min(morties_to_send, morties_remaining)
            result = self.client.send_morties(planet_idx, morties_to_send)
            survived_field = result.get("survived")
            survived_bool = bool(survived_field)
            morties_saved = int(survived_field) if isinstance(survived_field,int) else int(survived_bool) * morties_to_send

            a, b = self.phase_stats[planet_idx][phase]
            if survived_bool:
                a += 1
            else:
                b += 1
            self.phase_stats[planet_idx][phase] = (a, b)

            morties_remaining = result.get('morties_in_citadel', morties_remaining)
            trip_num += 1

            csv_rows.append([
                trip_num, self.client.get_planet_name(planet_idx),
                morties_to_send, morties_saved, int(survived_bool),
                phase, datetime.utcnow().isoformat()
            ])

            if trip_num % 50 == 0 or morties_remaining <= 0:
                print(f"Trip {trip_num}: Planet {self.client.get_planet_name(planet_idx)}, "
                      f"Sent={morties_to_send}, Survived={survived_bool}, Phase={phase}, Remaining={morties_remaining}")

            if trip_num > 20000:
                print("Reached 20000 trips, so quitting.")
                break

        try:
            with open(save_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            print(f"\nResults are saved in a csv file {save_csv}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        final_status = self.client.get_status()
        total_possible = final_status.get('morties_on_planet_jessica', 0) + final_status.get('morties_lost', 0)
        success_pct = (final_status.get('morties_on_planet_jessica', 0) / total_possible * 100) if total_possible > 0 else 0.0
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status.get('morties_on_planet_jessica')}")
        print(f"Morties Lost: {final_status.get('morties_lost')}")
        print(f"Success Rate: {success_pct:.2f}%")


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

