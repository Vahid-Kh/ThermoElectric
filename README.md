# Energy and Exergy Analysis of Heat Exchanger with TEG and Thermal Resistances

This project focuses on the energy and exergy analysis of a novel heat exchanger design incorporating Thermoelectric Generators (TEGs) and considering the impact of thermal resistances. The aim is to develop an efficient solution for waste heat recovery and conversion into electricity.

## Why This Project?

Traditional methods for converting heat to electricity often face limitations. Following technology assessments in 2023, it was concluded that:

* **Simplified Design:** The most competitive technologies for heat-to-electricity conversion should aim for a minimum number of heat exchanger (HX) surfaces, ideally a single surface, as opposed to multiple surfaces often required by systems like Organic Rankine Cycles (ORC).
* **Continuous Operation:** Continuous operation is preferable to cyclical processes for improved efficiency and practicality.

Furthermore, evaluations of technologies for converting waste heat into useful effects (such as HVAC&R, carbon capture, etc.) have highlighted that:

* **Heat Recovery as Heat:** Recovering waste heat directly as heat is often the most competitive approach.
* **Synergistic Systems:** Systems that combine heat recovery with power generation (heat + power) or offer possibilities for synergy (e.g., dehumidification and water harvesting) are highly promising.

This project addresses these findings by exploring a TEG-integrated heat exchanger system designed for continuous operation and direct heat recovery with simultaneous electricity generation.

## The Opportunity

* **Abundant Waste Heat:** Approximately two-thirds of the Total Primary Energy Supply is ultimately lost as waste heat.
* **Low-Grade Heat Challenge:** A significant portion of this waste heat is at temperatures below 100°C. For many industrial processes, there are currently no widely available or economically viable means of recovering this low-grade heat.
* **Our Concept:** This project aims to develop a concept that can effectively recover this low-temperature waste heat and simultaneously convert a portion of it into valuable electricity.

## Current Status

* **Technology Maturity:** Over the past 1.5 years, extensive benchtop testing has been performed to investigate critical parameters, including:
    * The effect of pressure on heat exchanger surfaces.
    * Performance of various Thermal Interface Materials (TIMs).
    * Part-to-part variations in components.
    * Products from several different suppliers.
* **Prototype Development:** Various Thermo-Electric Generator Heat Exchangers (TEGHEX) have been developed and tested. These prototypes serve to:
    * Benchmark the performance of different TEG modules.
    * Evaluate various operational scenarios.
    * Explore diverse applications for the technology.

## Code Overview

The code included in this project provides the tools and calculations for performing:

* **Energy Analysis:** Quantifying the energy flows, conversions, and losses within the TEGHEX system.
* **Exergy Analysis:** Evaluating the thermodynamic performance and identifying sources of irreversibility to optimize system efficiency.
* **TEG Integration:** Modeling the performance of thermoelectric generators within the heat exchanger.
* **Thermal Resistance Modeling:** Accounting for the impact of various thermal resistances (e.g., contact resistance, material resistance) on overall system performance.


## Diagram / Visualization 

![TEG descreption, thermal resistance and schematics](./2ndLawTDN/ToGitHub.svg)

## Getting Started

*(Users should add instructions on how to set up and run the code here. This might include dependencies, required software, and execution steps.)*

```bash
# Example:
# git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
# cd your-repo-name
# pip install -r requirements.txt
# python main_analysis.py



