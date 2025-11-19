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



# Score : [40%, 43%] 
# Review : Too easy, does not adapt to changing planet conditions + not robust
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



# Score : [45%, 48%]
# Review : It can be improved by adding planet switching, see AdaptiveStrategy below
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



# Score : [47%, 66%]
# Review : Enhanced UCB strategy with adaptive morty sending based on recent performance but needs more tuning(didn't succeed)
# Results are variable due to randomness, and we could not reach a good score consistently despite our research.
class UCBStrategy(MortyRescueStrategy):
    def execute_strategy(self, morties_per_trip=3, c=1.5, window_size=10, reevaluate_every=200, save_csv="outputs/data/ucb_results.csv"):
        """
        Adaptive Upper Confidence Bound strategy with reevaluation every `reevaluate_every` trips.
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
            c: Exploration parameter for UCB
            window_size: Size of the moving window for recent performance
            reevaluate_every: Number of trips between UCB score recalculations
            save_csv: Filename to save the results CSV
        Better performance by adjusting morties sent based on recent survival rates.
        """
        print("\n=== EXECUTING ADAPTIVE UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_morties = 1000
        n_planets = 3
        n = 0
        trips_since_eval = 0

        # Initialize stats
        planet_stats = {
            i: {"successes": 0, "trials": 0, "mean": 0.0, "history": deque(maxlen=window_size)}
            for i in range(n_planets)
        }

        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "SurvivalRate", "UCB"]]

        print("\nInitial exploration...")

        # Initial exploration: we send one trip to each planet
        for planet in range(n_planets):
            if morties_remaining <= 0:
                break
            to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(planet, to_send)
            survived = result["survived"]

            # update stats for each planet
            planet_stats[planet]["successes"] += survived
            planet_stats[planet]["trials"] += 1
            planet_stats[planet]["mean"] = survived / to_send
            planet_stats[planet]["history"].append(survived / to_send)

            # update remaining morties
            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            csv_rows.append([n, self.client.get_planet_name(planet), to_send, survived, planet_stats[planet]["mean"], float("inf")])
            print(f"  Planet {self.client.get_planet_name(planet)}: Survival={planet_stats[planet]['mean']:.2f}")

        print("\nMain loop (UCB selection)...")
        while morties_remaining > 0: # while we have morties to send
            if trips_since_eval >= reevaluate_every: # reevaluate UCB scores
                ucb_scores = {}
                for planet in range(n_planets):
                    trials = planet_stats[planet]["trials"]
                    mean = planet_stats[planet]["mean"]
                    # Calculate UCB score for planet using UCB1 formula
                    ucb_scores[planet] = mean + c * math.sqrt((2 * math.log(n)) / trials) if trials > 0 else float("inf")
                trips_since_eval = 0 # reset counter
            else:
                # Calculate UCB scores without reevaluation
                ucb_scores = {planet: planet_stats[planet]["mean"] for planet in range(n_planets)}

            # find best planet
            best_planet = max(ucb_scores, key=ucb_scores.get)
            planet_name = self.client.get_planet_name(best_planet)

            # calculate recent survival rate
            recent_rate = sum(planet_stats[best_planet]["history"]) / len(planet_stats[best_planet]["history"]) if planet_stats[best_planet]["history"] else 0
            
            # ==== decision on morties to send ====
            # if good recent survival = send more
            # if poor recent survival = send fewer
            if recent_rate > 0.6:
                to_send = min(3, morties_remaining)
            elif recent_rate > 0.4:
                to_send = min(2, morties_remaining)
            else:
                to_send = min(1, morties_remaining)

            # send morties
            result = self.client.send_morties(best_planet, to_send)
            survived = result["survived"]

            # update stats
            planet_stats[best_planet]["successes"] += survived
            planet_stats[best_planet]["trials"] += 1
            planet_stats[best_planet]["mean"] = planet_stats[best_planet]["successes"] / (planet_stats[best_planet]["trials"] * morties_per_trip)
            planet_stats[best_planet]["history"].append(survived / to_send)

            # update remaining morties
            morties_remaining = result["morties_in_citadel"]
            n += 1
            trips_since_eval += 1

            # log CSV
            csv_rows.append([n, planet_name, to_send, survived, planet_stats[best_planet]["mean"], ucb_scores.get(best_planet, 0)])

            # periodic status print
            if n % 20 == 0 or morties_remaining <= 0:
                print(f"Trip {n}: Sent {to_send} to {planet_name} | mean={planet_stats[best_planet]['mean']:.2f} | UCB={ucb_scores.get(best_planet, 0):.2f} | Remaining={morties_remaining}")

        # save CSV
        with open(save_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nResults saved in a csv file {save_csv}")

        # final stats
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/total_morties)*100:.2f}%")
            



# Score : [47%, 64%]
# Review : This is the third optimized version of our adaptive strategy, focusing on dynamic morty sending and planet switching.
# The scores were very unpredictable, we don't know why and could not stabilize them despite our efforts and time(we tried to develop also this stategy with some UCB in it, it didn't work).
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
        
        # see how many morties we have
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']

        # We start on the “best” planet based on our previously collected exploration data.
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data, consider_trend=True
        )
        print(f"Starting on {current_planet_name}")

        recent_results = deque(maxlen=window_size) # store recent survival rates
        total_trips = 0
        reevaluation_cooldown = 0 # to avoid switching planets too frequently

        # for debugging and final analysis
        csv_rows = [["Trip", "Planet", "MortiesSent", "Survived", "Rate", "PlanetReevaluated"]]

        while morties_remaining > 0:

            # if we have info, reactive recent rate
            if len(recent_results) > 0:
                recent_success = sum(recent_results) / len(recent_results)
            
            # neutral and optimistic start
            else:
                recent_success = 0.55  

            # ==== decision on morties to send ====
            # if good recent survival = send more
            # if poor recent survival = send fewer
            if recent_success >= 0.65:
                morties_to_send = min(3, morties_remaining)
            elif recent_success >= 0.40:
                morties_to_send = min(2, morties_remaining)
            else:
                morties_to_send = 1

            # send morties 
            result = self.client.send_morties(current_planet, morties_to_send)
            survived = result["survived"]
            morties_remaining = result["morties_in_citadel"]

            recent_results.append(survived / morties_to_send)
            total_trips += 1

            print(f"Trip {total_trips}: Sent {morties_to_send} -> survived={survived} | recent={recent_success:.2f}")

            planet_reevaluated = False

            # forced exploration every X trips to avoid beeing stuck on the same one 
            if total_trips % exploration_cycle == 0:
                print("\nExploration cycle")
                new_planet, new_name = self.collector.get_best_planet(self.exploration_data, consider_trend=True)
                current_planet = new_planet
                current_planet_name = new_name
                planet_reevaluated = True
                reevaluation_cooldown = 15

            # We check if teh current planet is still good 
            if total_trips % reevaluate_every == 0 and reevaluation_cooldown == 0:
                window_perf = sum(recent_results) / len(recent_results)
                print(f"-> Reevaluating: window perf = {window_perf:.2f}")

                # If performance is below a threshold, we switch to the next best planet.
                if window_perf < switch_threshold:
                    print("Performance below threshold -> switching planet")
                    new_planet, new_name = self.collector.get_best_planet(self.exploration_data, consider_trend=True)
                    current_planet = new_planet
                    current_planet_name = new_name
                    planet_reevaluated = True
                    reevaluation_cooldown = 15

            # Thanks to info we know that purge is great so we try to jump time to time to maybe bust the score 
            if current_planet == 2 and total_trips % 50 == 0:
                print("Returning to planet 2 as a boost strategy")

            # log CSV 
            csv_rows.append([total_trips,current_planet_name,morties_to_send,survived,survived / morties_to_send,planet_reevaluated])

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





# Score : [68%, 73.6%]
# Review : This strategy uses Thompson Sampling with cyclic phases to adaptively choose planets and morty counts.
# It works well by capturing periodic changes in planet survival rates.
class ThompsonCyclicStrategy(MortyRescueStrategy):
    """
    Thompson Sampling strategy with cyclic phases for each planet.
    Each planet is assumed to follow a repeating performance cycle
    """

    # Give by the class 
    PLANET_PERIODS = {0: 10, 1: 20, 2: 200}

    def __init__(self, client):
        super().__init__(client)
        # For every planet, we maintain a list of (a, b) parameters representing
        # the Beta distribution for each phase of its cyclic behavior.
        #
        # a = number of successes + 1
        # b = number of failures + 1
        #
        # We initialize everything at (1, 1) which is a uniform Beta prior.
        self.phase_stats: Dict[int, List[Tuple[int,int]]] = {
            planet: [(1,1) for _ in range(period)]
            for planet, period in self.PLANET_PERIODS.items()
        }

    def _current_phase(self, planet_idx, trip_num):
        """
        Determine which “phase” of the cycle we are currently in for a given planet.
        """
        return trip_num % self.PLANET_PERIODS[planet_idx]

    def _choose_planet(self, trip_num):
        """
        Perform Thompson Sampling across planets.

        For each planet we select its current phase and sample a probability from Beta(a, b)
        We choose the planet with the highest sampled probability.
        """
        sampled_probs = {}
        for planet, phases in self.phase_stats.items():
            phase = self._current_phase(planet, trip_num)
            a, b = phases[phase]
            sampled_probs[planet] = random.betavariate(a, b)
        best_planet = max(sampled_probs, key=sampled_probs.get)
        return best_planet

    def _choose_morties_count(self, planet_idx, trip_num):
        """
        Decide how many morties to send for the selected planet.
        We use the calcul: a / (a + b) for the current phase.
        Higher estimated survival = send more morties.
        """
        phase = self._current_phase(planet_idx, trip_num)
        a, b = self.phase_stats[planet_idx][phase]
        prob = a / (a + b)

        # to balance risk and potential reward
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

            # send and see how it go 
            result = self.client.send_morties(planet_idx, morties_to_send)
            survived_field = result.get("survived")
            # convert into boolean (more pratical)
            survived_bool = bool(survived_field)
            morties_saved = int(survived_field) if isinstance(survived_field,int) else int(survived_bool) * morties_to_send

            # update our beta parameter (for the planet/phase)
            a, b = self.phase_stats[planet_idx][phase]
            # reward sucees
            if survived_bool:
                a += 1
            # penalize fail
            else:
                b += 1
            self.phase_stats[planet_idx][phase] = (a, b)

            morties_remaining = result.get('morties_in_citadel', morties_remaining)
            trip_num += 1

            # for the analysis and debug on the strategy
            csv_rows.append([trip_num, self.client.get_planet_name(planet_idx),morties_to_send, morties_saved, int(survived_bool),phase, datetime.utcnow().isoformat()])

            # just for having some info during the play
            if trip_num % 50 == 0 or morties_remaining <= 0:
                print(f"Trip {trip_num}: Planet {self.client.get_planet_name(planet_idx)}, "
                      f"Sent={morties_to_send}, Survived={survived_bool}, Phase={phase}, Remaining={morties_remaining}")

            # to avoid infinite loops
            if trip_num > 20000:
                print("Reached 20000 trips, so quitting.")
                break

        # save the result 
        try:
            with open(save_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            print(f"\nResults are saved in a csv file {save_csv}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # metric 
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


"""
Pour commencer ce projet, nous avons d'abord testé les stratégies données de départ, nous avons observé des scores plutôt prometteurs, 
même si elles restent faibles. Pour mieux comprendre le fonctionnement des planètes et leurs variations, nous avons implémenté une phase d'exploration.
Durant cette phase, nous avons essayé de générer un maximum de fichiers csv pour analyser les données, et de visualiser des résultats à l'aide du fichier
visualizations.py. Malheureusement, nous n'avons pas réussi à correctement interpréter ces résultats, nous sommes également passées par la création de
notre fichier de visualisation mais sans succès. Ce que nous avons fait c'est de passer les fichiers csv à un chat pour qu'il nous aide à 
comprendre la multitude de données obtenues. Heuresement pour nous, dans le discord de la classe nous avons eu la connaissance des périodes de chaque
planète, ce qui nous a permis de mieux comprendre les variations des planètes. 
Notre deuxième phase de dévéloppement a été la création de plusieurs stratégies basées sur certaines notions de cours. Nous avons commencé par améliorer 
la stratégie AdaptiveStrategy en y ajoutant une phase d'exploration et en implémentant un système de changement de planète lorsque la performance
de la planète actuelle devient trop faible. Malgrè ce premier ajout les résultats n'étaient pas satisfaisants. Le score obtenu n'était pas foncièrement
meilleur et surtout très instable(on a observé que cette instabilité dépendait en partie par le choix de la première planète explorée, nous obtenions
de meilleurs résultats en commençant par la planète Purge). Graçe aux fréquences, nous avons continué d'adpater notre stratégie. Malheureusement, malgrè
nos nombreux essais, nous n'avons jamais eu de vrais améliorations(plus on avancé, plus nos résultats devenaient inprevisibles). Nous avons donc décidé
de suivre vos conseils et de tenter d'implémenter une stratégie basée sur l'algorithme UCB. Nous avons réussi à implémenter cette stratégie UCBStrategy,
mais les résultats n'étaient pas tellement meilleurs que ceux de notre stratégie Adaptive. Une des méthodes que nous avons testé en termes de tests était
de collecter des fichiers csv run de chaque stratégie faite et de les passer à un chatbot pour qu'il nous aide à analyser les données et à nous suggérer 
des améliorations et surtout comprendre les différeces entre nos runs et ce qui a impacté notre performance. Cette méthode nous a permis d'améliorer 
certains détails dans nos stratégies, mais nous n'avons jamais réussi à stabiliser nos scores.
Après nos multitudes essais d'améliorations, nous avons enfin décidé à chercher une nouvelle stratégie plus adaptée. Lors de nos recherches nous sommes 
tombées sur ThompsonCyclicStrategy. Cette stratégie est basée sur l'algorithme de Thompson Sampling, cela nous permet d'avoir une approche bayésienne pour
la sélection des planètes. De plus, l'aspect cyclique de la stratégie nous permet d'explorer les différentes phases de chaque planète, en adaptant notre 
choix en fonction des variations périodiques. Nous avons implémenté cette stratégie et nous avons observé des résultats plus stables et prometteurs. 
Notre objectif avec ce projet était d'atteindre un score au-dessus de 70%, c'est chose faite avec cette stratégie.
Si nous avions plus de temps, nous aurions aimé travailler sur la visualisation des données pour mieux comprendre les variations des planètes et pour
afficher nos résultats après nos runs, pour ne pas à avoir à analyser des fichiers csv(nous avons essayé d'utiliser le fichier de visualisation existant,
mais nous n'avons pas réussi à comprendre et l'adapter à nos besoins, du moins pas suffisamment pour en tirer des conclusions utilisables).
Nous avons également de la chance que les périodes des planètes nous ont été données.
"""