"""
Data collection and analysis functions for the Morty Express Challenge.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from api_client import SphinxAPIClient
import os
from datetime import datetime
from numpy.fft import fft, fftfreq


class DataCollector:
    """Collects and analyzes data from the challenge."""

    def __init__(self, client: SphinxAPIClient, output_dir: str = "outputs/data"):
        """
        Initialize the data collector.

        Args:
            client: SphinxAPIClient instance
            output_dir: folder to save CSV files
        """
        self.client = client
        self.trips_data = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # --- Exploration functions ---
    def explore_planet(self, planet: int, num_trips: int, morty_count: int = 1) -> pd.DataFrame:
        """
        Send multiple trips to a single planet to observe its behavior.
        """
        print(f"\nExploring {self.client.get_planet_name(planet)} with {morty_count} Morty(ies)...")

        trips = []
        for i in range(num_trips):
            try:
                result = self.client.send_morties(planet, morty_count)
                trip_data = {
                    'trip_number': i + 1,
                    'planet': planet,
                    'planet_name': self.client.get_planet_name(planet),
                    'morties_sent': result['morties_sent'],
                    'survived': result['survived'],
                    'steps_taken': result['steps_taken'],
                    'morties_in_citadel': result['morties_in_citadel'],
                    'morties_on_planet_jessica': result['morties_on_planet_jessica'],
                    'morties_lost': result['morties_lost']
                }
                trips.append(trip_data)
                self.trips_data.append(trip_data)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_trips} trips")

            except Exception as e:
                print(f"Warning: Error on trip {i + 1}: {e}")
                continue  # continue instead of break

        df = pd.DataFrame(trips)
        if not df.empty:
            survival_rate = df['survived'].mean() * 100
            print(f"Results for {self.client.get_planet_name(planet)}: Survival Rate {survival_rate:.2f}%")

        return df

    def explore_all_planets(self, trips_per_planet: int = 30, morty_count: int = 1) -> pd.DataFrame:
        """
        Explore all three planets for a given morty_count.
        """
        self.client.start_episode()
        self.trips_data = []

        all_data = []
        for planet in [0, 1, 2]:
            df = self.explore_planet(planet, trips_per_planet, morty_count)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    # --- FFT / Phase Analysis ---
    def calculate_dominant_period(self, df: pd.DataFrame, column: str = 'survived') -> Dict[int, int]:
        """
        Calculate the dominant period (in trips) for each planet using FFT.
        """
        periods = {}
        for planet in df['planet'].unique():
            planet_data = df[df['planet'] == planet]
            signal = planet_data[column].astype(float).values
            N = len(signal)
            yf = fft(signal - np.mean(signal))
            xf = fftfreq(N, d=1)[:N//2]
            magnitudes = np.abs(yf[:N//2])

            # ignore zero frequency
            xf_nonzero = xf[1:]
            magnitudes_nonzero = magnitudes[1:]

            if len(magnitudes_nonzero) == 0:
                periods[planet] = None
                continue

            dominant_idx = np.argmax(magnitudes_nonzero)
            dominant_freq = xf_nonzero[dominant_idx]
            period = int(round(1 / dominant_freq)) if dominant_freq != 0 else None
            periods[planet] = period

            print(f"Planet {planet} ({self.client.get_planet_name(planet)}): Dominant period ≈ {period} trips")

        return periods

    def build_phase_survival_map(self, df: pd.DataFrame, periods: Dict[int, int]) -> Dict[int, pd.DataFrame]:
        """
        Build phase-survival map for each planet.
        """
        phase_maps = {}
        for planet, period in periods.items():
            if period is None:
                continue

            planet_data = df[df['planet'] == planet].copy()
            planet_data['phase'] = (planet_data['trip_number'] - 1) % period
            phase_map = planet_data.groupby('phase')['survived'].mean().reset_index()
            phase_maps[planet] = phase_map

            print(f"Phase-survival map for Planet {planet} ({self.client.get_planet_name(planet)}):\n{phase_map}\n")

        return phase_maps

    # --- Utility functions ---
    def save_data(self, df: pd.DataFrame, morty_count: int = 1, planet: int = None):
        """
        Save collected trip data to CSV.
        """
        if df is None or df.empty:
            print("No data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        planet_str = f"_planet{planet}" if planet is not None else ""
        filename = f"{self.output_dir}/morty{morty_count}{planet_str}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"📁 Data saved to {filename}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load trip data from CSV.
        """
        df = pd.read_csv(filename)
        self.trips_data = df.to_dict('records')
        print(f"Loaded {len(df)} trips from {filename}")
        return df

    def explore_all_morty_counts(self, morty_counts: List[int] = [1, 2, 3], trips_per_planet: int = 150):
        """
        Explore all planets for multiple morty counts.
        """
        all_results = {}
        for count in morty_counts:
            print(f"\n=== Exploring all planets with {count} Morty(ies) ===")
            combined_df = self.explore_all_planets(trips_per_planet, morty_count=count)
            self.save_data(combined_df, morty_count=count)
            all_results[count] = combined_df
        return all_results
    
    def analyze_risk_changes(self, df: pd.DataFrame) -> Dict:
        """Analyse l’évolution de la survie sur chaque planète."""
        analysis = {}
        for planet in df['planet'].unique():
            planet_data = df[df['planet'] == planet].copy()
            planet_name = planet_data['planet_name'].iloc[0]

            mid_point = len(planet_data) // 2
            early = planet_data.iloc[:mid_point]
            late = planet_data.iloc[mid_point:]

            early_survival = early['survived'].mean()
            late_survival = late['survived'].mean()
            change = late_survival - early_survival

            analysis[planet_name] = {
                'planet': planet,
                'early_survival_rate': early_survival * 100,
                'late_survival_rate': late_survival * 100,
                'change': change * 100,
                'trend': 'improving' if change > 0 else 'worsening',
                'total_trips': len(planet_data),
                'overall_survival_rate': planet_data['survived'].mean() * 100
            }
        return analysis

    def get_best_planet(self, df: pd.DataFrame, consider_trend: bool = True) -> Tuple[int, str]:
        """Retourne la planète avec le meilleur taux de survie ou tendance positive."""
        analysis = self.analyze_risk_changes(df)
        if consider_trend:
            # Chercher planète avec tendance améliorante
            improving_planets = {k:v for k,v in analysis.items() if v['trend'] == 'improving'}
            if improving_planets:
                best_name = max(improving_planets, key=lambda x: improving_planets[x]['overall_survival_rate'])
                return analysis[best_name]['planet'], best_name
        # Sinon, planète avec meilleure survie globale
        best_name = max(analysis, key=lambda x: analysis[x]['overall_survival_rate'])
        return analysis[best_name]['planet'], best_name



# --- Example usage ---
if __name__ == "__main__":
    try:
        client = SphinxAPIClient()
        collector = DataCollector(client)
        print("✅ Data Collector initialized!\n")

        # Explore with 1, 2, 3 Mortys
        collector.explore_all_morty_counts([1, 2, 3], trips_per_planet=150)

        # Load combined data and analyze dominant periods / phase maps
        combined_df = pd.concat(collector.trips_data, ignore_index=True) if collector.trips_data else pd.DataFrame()
        periods = collector.calculate_dominant_period(combined_df)
        phase_maps = collector.build_phase_survival_map(combined_df, periods)

    except Exception as e:
        print(f"❌ Error: {e}")
