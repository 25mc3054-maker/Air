from typing import Dict, Any, List, Optional

class LongRangeScenarioEngine:
    """
    Five-Year Long-Range Pollution-Risk and Policy Scenario Engine (Delhi-NCR).
    
    IMPORTANT SCIENTIFIC GOVERNANCE:
    This module DOES NOT produce deterministic hour-by-hour AQI forecasts 5 years in advance.
    Instead, it implements aggregated multi-year scenario simulations based on:
      - NCAP (National Clean Air Programme) emission trajectories
      - Regional biomass burning trends
      - Climate-induced atmospheric stagnation frequency
      - Sectoral policy intervention sensitivities
    """
    SCENARIOS = {
        "bau": {
            "name": "Business As Usual (BAU)",
            "description": "Historical growth in vehicle fleet (+4%/yr) and regional industrial activity with current GRAP enforcement levels.",
            "emission_factor_annual": 1.015,
            "stagnation_days_annual": 32,
            "biomass_reduction_pct": 0.0
        },
        "ncap_strict": {
            "name": "Accelerated NCAP + GRAP IV Compliance",
            "description": "Strict compliance with NCAP targets: 35% reduction in particulate load by 2028, mandatory EV bus transition, active dust control.",
            "emission_factor_annual": 0.94,
            "stagnation_days_annual": 28,
            "biomass_reduction_pct": 50.0
        },
        "climate_stagnation": {
            "name": "Adverse Climate / Boundary Layer Stagnation",
            "description": "Increased winter cold-pool frequency, reduced post-monsoon surface wind speeds, and delayed boundary layer expansion.",
            "emission_factor_annual": 1.02,
            "stagnation_days_annual": 45,
            "biomass_reduction_pct": 10.0
        },
        "clean_transition": {
            "name": "Comprehensive Regional Clean Transition",
            "description": "Rapid industrial boiler electrification, zero-tolerance in-situ crop residue management, 100% BS-VI and EV adoption.",
            "emission_factor_annual": 0.88,
            "stagnation_days_annual": 24,
            "biomass_reduction_pct": 85.0
        }
    }

    def generate_5year_outlook(
        self,
        base_year: int = 2024,
        baseline_annual_pm25: float = 108.5
    ) -> Dict[str, Any]:
        """
        Simulates 5-year annual risk trajectories across all 4 scenarios.
        """
        years = [base_year + i for i in range(5)]
        trajectories = {}

        for sc_id, sc in self.SCENARIOS.items():
            pm_series = []
            severe_days_series = []
            curr_pm = baseline_annual_pm25

            for yr_idx, yr in enumerate(years):
                if yr_idx > 0:
                    curr_pm = curr_pm * sc["emission_factor_annual"]
                
                # Estimate annual severe days (AQI > 400 / PM2.5 > 250)
                # Severe days correlate non-linearly with annual mean and stagnation days
                base_severe = sc["stagnation_days_annual"] * (curr_pm / 100.0) ** 1.3
                severe_days = int(round(max(5, min(90, base_severe))))
                
                pm_series.append(round(curr_pm, 1))
                severe_days_series.append(severe_days)

            trajectories[sc_id] = {
                "scenario_name": sc["name"],
                "description": sc["description"],
                "years": years,
                "annual_mean_pm25": pm_series,
                "projected_severe_days_per_year": severe_days_series,
                "winter_episode_risk_level": "CRITICAL" if pm_series[-1] > 110 else ("HIGH" if pm_series[-1] > 80 else "MODERATE")
            }

        return {
            "outlook_type": "Long-Range Scenario Outlook (5-Year Aggregated Policy Simulation)",
            "disclaimer": "SCIENTIFIC NOTICE: This projection models multi-year policy and meteorological risk scenarios. It is NOT a deterministic hourly AQI prediction.",
            "baseline_year": base_year,
            "baseline_annual_mean_pm25": baseline_annual_pm25,
            "horizon_years": years,
            "scenarios": trajectories,
            "sectoral_breakdown_est": {
                "vehicular_transport": "36%",
                "road_and_construction_dust": "24%",
                "industrial_combustion": "18%",
                "seasonal_biomass_burning": "14%",
                "domestic_and_waste": "8%"
            }
        }

scenario_engine = LongRangeScenarioEngine()
