import numpy as np  # For ln, and to handle potential division by zero gracefully
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

# --- Constants and Helper Functions ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)
BOLTZMANN_CONST = 1.38e-23  # J/K


def celsius_to_kelvin(T_celsius):
    return T_celsius + 273.15

def kelvin_to_celsius(T_kelvin):
    return T_kelvin - 273.15

# --- Thermal Resistance Estimation Functions ---

def resistance_conduction(thickness_m, area_m2, k_W_mK):
    if k_W_mK * area_m2 == 0: return np.inf
    return thickness_m / (k_W_mK * area_m2)

def resistance_convection(h_W_m2K, area_m2):
    if h_W_m2K * area_m2 == 0: return np.inf
    return 1 / (h_W_m2K * area_m2)

def resistance_radiation(epsilon, T1_K, T2_K, area_m2):
    # This is a linearized radiation resistance, often used for simplified analysis.
    # For more accuracy in heat flux, use the direct Stefan-Boltzmann law.
    if area_m2 == 0 or epsilon == 0: return np.inf
    # Avoid issues if T1 or T2 are very close or equal, leading to large denominators
    if abs(T1_K - T2_K) < 1e-3: # or T1_K < T2_K: # only for heat transfer from T1 to T2
         # A very small non-zero temperature difference or T1 not hotter than T2 means high resistance to this specific heat flow
        h_rad_approx = epsilon * STEFAN_BOLTZMANN_CONST * 4 * ((T1_K + T2_K) / 2)**3
    else:
        h_rad_approx = epsilon * STEFAN_BOLTZMANN_CONST * (T1_K**2 + T2_K**2) * (T1_K + T2_K)

    if h_rad_approx == 0: return np.inf
    return 1 / (h_rad_approx * area_m2)


# --- Core Thermal Calculation Functions ---

def calculate_heat_flux_through_stack(
    T_hot_K, # Temperature of the hot side of the TEG stack
    T_in_fluid_K,
    T_out_fluid_K,
    R_TEG_A,      # Resistance of TEG module itself
    R_HX_cond_A,  # Resistance of HX conductive parts (wall)
    R_HX_conv_A   # Resistance of HX convective part (fluid film)
):
    T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
    # R_stack_A is the total resistance from TEG hot side to the average fluid temperature
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A # Resistances are per unit area A
    if R_stack_A <= 0: # Avoid division by zero or negative resistance
        return 0 if T_hot_K <= T_f_avg_K else np.inf
    return max(0, (T_hot_K - T_f_avg_K) / R_stack_A)


def calculate_radiative_heat_flux_loss(
    T_surface_K,    # Temperature of the emitting surface
    T_ambient_rad_K, # Temperature of the surroundings for radiation
    emissivity      # Emissivity of the T_surface_K
):
    return emissivity * STEFAN_BOLTZMANN_CONST * (T_surface_K**4 - T_ambient_rad_K**4)


def calculate_convective_heat_flux_loss(
    T_surface_K, # Temperature of the convecting surface
    T_air_K,     # Temperature of the ambient air
    h_conv_air   # Convective heat transfer coefficient
):
    return h_conv_air * (T_surface_K - T_air_K)

# --- Exergy Analysis Functions ---

def calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K):
    if q_stack <= 0 or T_out_fluid_K <= T_in_fluid_K: # Ensure T_out > T_in for log
        return 0.0
    delta_T = T_out_fluid_K - T_in_fluid_K
    # Using np.log for natural logarithm
    return q_stack - (q_stack / delta_T) * T_dead_K * np.log(T_out_fluid_K / T_in_fluid_K)


def calculate_exergy_destruction_component(q_flux, T_hot_K, T_cold_K, T_dead_K):
    if q_flux <= 0 or T_hot_K <= T_cold_K: # Ensure T_hot > T_cold for positive destruction
        return 0.0
    # Ensure T_cold_K and T_hot_K are positive for division
    if T_cold_K <= 0 or T_hot_K <=0: return np.inf # Or handle as an error
    return q_flux * T_dead_K * (1.0 / T_cold_K - 1.0 / T_hot_K)

# --- Case 1: Known T_hot_contact_surface (heat supplied by conduction) ---
def analyze_conduction_known_T_surface_with_exergy(
    T_hot_contact_surface_C, # Temperature of the surface the TEG is touching
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,        # TEG's own thermal resistance (K.m^2/W)
    R_HX_cond_A,    # Heat exchanger conduction resistance (K.m^2/W)
    R_HX_conv_A,    # Heat exchanger convection resistance (K.m^2/W)
    T_dead_C
):
    # Convert all temperatures to Kelvin
    T_hot_contact_K = celsius_to_kelvin(T_hot_contact_surface_C)
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)
    T_f_avg_K = (T_in_K + T_out_K) / 2

    # Heat flux through stack (from contact surface to fluid)
    q_stack = calculate_heat_flux_through_stack(
        T_hot_contact_K, T_in_K, T_out_K,
        R_TEG_A, R_HX_cond_A, R_HX_conv_A
    )

    # Temperature drops across components
    # T_hot_contact_K is Ts (TEG hot side)
    T_TEG_cold_K = T_hot_contact_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A
    # T_f_avg_K is the effective cold side for R_HX_conv_A

    # Exergy input from the heat conducted into the TEG at T_hot_contact_K
    # This is the exergy associated with q_stack at the hot contact surface temperature
    ex_in_source_effective = q_stack * (1 - T_dead_K / T_hot_contact_K) if q_stack > 0 and T_hot_contact_K > 0 else 0.0

    # Exergy recovered by fluid
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)

    # Exergy destruction
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, T_hot_contact_K, T_TEG_cold_K, T_dead_K)
    ex_dest_HX_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)
    ex_dest_HX_conv = calculate_exergy_destruction_component(q_stack, T_HX_wall_K, T_f_avg_K, T_dead_K) # Destruction in HX convection to fluid

    # Total destruction in stack
    ex_dest_stack_total = ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_conv

    # Check: ex_in_source_effective should ideally balance ex_recovered + ex_dest_stack_total
    # Some small differences might occur due to T_f_avg vs (T_in, T_out) in exergy calculation for fluid
    # A more precise exergy balance would track entropy generation in each step.

    return {
        'T_hot_contact_C': T_hot_contact_surface_C,
        'q_stack': q_stack,
        'ex_in_source_effective': ex_in_source_effective,
        'ex_recovered': ex_recovered,
        'ex_dest_TEG': ex_dest_TEG,
        'ex_dest_HX_cond': ex_dest_HX_cond,
        'ex_dest_HX_conv': ex_dest_HX_conv,
        'ex_dest_stack_total': ex_dest_stack_total,
        'T_TEG_cold_K': T_TEG_cold_K,
        'T_HX_wall_K': T_HX_wall_K,
    }

# --- Case 2: Known q_conducted_input_W_m2, solve for TEG hot surface temperature Ts ---
def analyze_conduction_known_q_input_with_exergy(
    q_conducted_input_W_m2, # Heat flux conducted into the TEG's hot side
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A,
    emissivity_TEG_exposed_surface, # Emissivity of the TEG's *other* exposed surface (e.g., top)
    T_surr_rad_C,                 # Ambient temp for radiation from exposed surface
    h_conv_air_TEG_exposed,       # Convection coeff for TEG's exposed surface
    T_air_C,                      # Ambient air temp for convection from exposed surface
    T_dead_C,
    initial_Ts_guess_C=200.0
):
    # Convert to Kelvin
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_f_avg_K = (T_in_K + T_out_K) / 2
    T_surr_rad_K = celsius_to_kelvin(T_surr_rad_C)
    T_air_K = celsius_to_kelvin(T_air_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)
    R_stack_A_total = R_TEG_A + R_HX_cond_A + R_HX_conv_A

    # Energy balance to find Ts (TEG hot surface temperature)
    def energy_balance_Ts(Ts_K_scalar):
        Ts_K = Ts_K_scalar # fsolve gives an array
        if Ts_K <= 0: return 1e9 # Invalid temperature

        # Heat through the stack (TEG hot side Ts_K to fluid average T_f_avg_K)
        q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A_total) if R_stack_A_total > 0 else 0

        # Losses from the TEG's *other* exposed surface (e.g., top surface)
        q_rad_loss_teg_exposed = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_TEG_exposed_surface)
        q_conv_loss_teg_exposed = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air_TEG_exposed)

        # Balance: Heat conducted IN = Heat through stack + Heat lost from exposed TEG surface
        return q_conducted_input_W_m2 - (q_stack + q_rad_loss_teg_exposed + q_conv_loss_teg_exposed)

    # Solve for Ts_K
    Ts_K_solution, info, ier, mesg = fsolve(energy_balance_Ts, celsius_to_kelvin(initial_Ts_guess_C), full_output=True)
    Ts_K = Ts_K_solution[0]
    success = (ier == 1)

    if not success:
        # Try with a different guess or wider bounds if using a bounded solver
        # For fsolve, a different initial guess might help.
        # One common issue is Ts_K becoming too low or T_f_avg_K being higher than Ts_K initially.
        # Try to ensure initial_Ts_guess_C is reasonably above T_f_avg_C
        initial_Ts_guess_K_alt = T_f_avg_K + 50 # Alternative guess
        Ts_K_solution, info, ier, mesg = fsolve(energy_balance_Ts, initial_Ts_guess_K_alt, full_output=True)
        Ts_K = Ts_K_solution[0]
        success = (ier == 1)
        if not success:
             return {'solver_success': False, 'solver_message': mesg.strip(), 'Ts_C': np.nan, 'energy_balance_residual': energy_balance_Ts(Ts_K) if Ts_K else None}


    # Recalculate fluxes with the solved Ts_K
    q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A_total) if R_stack_A_total > 0 else 0
    q_rad_loss_teg_exposed = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_TEG_exposed_surface)
    q_conv_loss_teg_exposed = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air_TEG_exposed)

    # Exergy streams
    # Exergy input is based on the heat conducted in at the solved TEG hot surface temperature Ts_K
    ex_in_total = q_conducted_input_W_m2 * (1 - T_dead_K / Ts_K) if Ts_K > 0 else 0.0
    ex_recovered_fluid = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)

    # Temperature drops and exergy destructions in the stack
    T_TEG_cold_K = Ts_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A
    # T_f_avg_K is the final cold temperature for the stack for exergy destruction in HX conv

    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, Ts_K, T_TEG_cold_K, T_dead_K)
    ex_dest_HX_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)
    ex_dest_HX_conv = calculate_exergy_destruction_component(q_stack, T_HX_wall_K, T_f_avg_K, T_dead_K) # Destruction in HX convection

    # Exergy associated with losses from the TEG's exposed surface
    ex_loss_rad_teg_exposed = q_rad_loss_teg_exposed * (1 - T_dead_K / Ts_K) if Ts_K > 0 else 0.0
    ex_loss_conv_teg_exposed = q_conv_loss_teg_exposed * (1 - T_dead_K / Ts_K) if Ts_K > 0 else 0.0

    # Total exergy destruction/loss in the system
    # ex_dest_system_total = ex_in_total - ex_recovered_fluid
    # This can be broken down:
    ex_dest_components_sum = ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_conv + ex_loss_rad_teg_exposed + ex_loss_conv_teg_exposed
    # The "Other" exergy destruction accounts for any mismatch, e.g. if q_conducted_input is not perfectly q_stack + q_losses due to solver precision
    # or exergy of heat that enters but is not accounted for in q_stack or defined losses.
    # Ideally: ex_in_total = ex_recovered_fluid + ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_conv + ex_loss_rad_teg_exposed + ex_loss_conv_teg_exposed
    ex_dest_unaccounted = ex_in_total - (ex_recovered_fluid + ex_dest_components_sum)


    eta_ex_system = ex_recovered_fluid / ex_in_total if ex_in_total > 0 else 0.0

    return {
        'solver_success': True,
        'solver_message': mesg.strip() if not success else "Solver converged.",
        'Ts_C': kelvin_to_celsius(Ts_K),
        'q_conducted_input_W_m2': q_conducted_input_W_m2,
        'q_stack': q_stack,
        'q_rad_loss_teg_exposed': q_rad_loss_teg_exposed,
        'q_conv_loss_teg_exposed': q_conv_loss_teg_exposed,
        'ex_in_total': ex_in_total,
        'ex_recovered_fluid': ex_recovered_fluid,
        'ex_dest_TEG': ex_dest_TEG,
        'ex_dest_HX_cond': ex_dest_HX_cond,
        'ex_dest_HX_conv': ex_dest_HX_conv,
        'ex_loss_rad_teg_exposed': ex_loss_rad_teg_exposed,
        'ex_loss_conv_teg_exposed': ex_loss_conv_teg_exposed,
        'ex_dest_unaccounted': ex_dest_unaccounted, # Should be close to zero
        'eta_ex_system': eta_ex_system,
        'T_TEG_cold_K': T_TEG_cold_K,
        'T_HX_wall_K': T_HX_wall_K,
        'T_f_avg_K': T_f_avg_K
    }


# --- Example Run: Adjusting All Variables ---
if __name__ == '__main__':
    # Define adjustable parameters for the TEG stack and HX
    # These are resistances PER UNIT AREA (K.m^2/W or m^2.K/W)
    R_TEG_A_val      = 0.015  # TEG conduction resistance (K.m^2/W)
    R_HX_cond_A_val  = 0.005  # HX conduction resistance (K.m^2/W)
    R_HX_conv_A_val  = 0.010  # HX convection resistance (K.m^2/W)

    # Fluid temperatures
    T_in_fluid_C_val   = 40.0  # °C
    T_out_fluid_C_val  = 90.0  # °C

    # Dead-state (ambient for exergy calculations)
    T_dead_C_val       = 25.0  # °C

    # --- Scenario 1: Known Hot Contact Surface Temperature ---
    T_hot_contact_surface_C_val = 300.0 # °C - Temperature of the surface the TEG is touching

    print("--- Running Scenario 1: Known Hot Contact Surface Temperature ---")
    results_scenario1 = analyze_conduction_known_T_surface_with_exergy(
        T_hot_contact_surface_C_val,
        T_in_fluid_C_val,
        T_out_fluid_C_val,
        R_TEG_A_val,
        R_HX_cond_A_val,
        R_HX_conv_A_val,
        T_dead_C_val
    )
    print(f"Scenario 1 q_stack: {results_scenario1['q_stack']:.2f} W/m^2")
    print(f"Scenario 1 Exergy Recovered: {results_scenario1['ex_recovered']:.2f} W/m^2")
    print(f"Scenario 1 Exergy Input (effective for q_stack): {results_scenario1['ex_in_source_effective']:.2f} W/m^2")
    print(f"Scenario 1 Exergy Destruction in TEG: {results_scenario1['ex_dest_TEG']:.2f} W/m^2")
    print(f"Scenario 1 Exergy Destruction in HX Cond: {results_scenario1['ex_dest_HX_cond']:.2f} W/m^2")
    print(f"Scenario 1 Exergy Destruction in HX Conv: {results_scenario1['ex_dest_HX_conv']:.2f} W/m^2")
    print(f"Scenario 1 Total Stack Exergy Destruction: {results_scenario1['ex_dest_stack_total']:.2f} W/m^2")
    print("-" * 30)


    # --- Scenario 2: Known Conducted Heat Input into TEG ---
    # Use q_stack from Scenario 1 as an example for q_conducted_input_W_m2
    # OR define an independent q_conducted_input_W_m2 value.
    q_conducted_input_W_m2_val = results_scenario1.get('q_stack', 20000.0) # Example: 20 kW/m^2 or from scenario 1
    # q_conducted_input_W_m2_val = 25000.0 # Alternatively, set a specific value

    # Parameters for losses from the TEG's *other* exposed surface (e.g., top, if bottom is on hot source)
    emissivity_TEG_exposed_val = 0.8      # Emissivity of the TEG's exposed surface
    T_surr_rad_C_val           = 25.0     # Surrounding temp for radiation from TEG exposed surface (°C)
    h_conv_air_TEG_exposed_val = 12.0     # Convective coefficient for TEG exposed surface (W/m^2·K)
    T_air_C_val                = 25.0     # Ambient air for convection from TEG exposed surface (°C)
    initial_Ts_guess_C_val     = T_hot_contact_surface_C_val - 10 # Guess based on Scenario 1, or a bit above fluid temp

    print(f"\n--- Running Scenario 2: Known Conducted Heat Input ({q_conducted_input_W_m2_val:.2f} W/m^2) ---")
    results_scenario2 = analyze_conduction_known_q_input_with_exergy(
        q_conducted_input_W_m2_val,
        T_in_fluid_C_val,
        T_out_fluid_C_val,
        R_TEG_A_val,
        R_HX_cond_A_val,
        R_HX_conv_A_val,
        emissivity_TEG_exposed_val,
        T_surr_rad_C_val,
        h_conv_air_TEG_exposed_val,
        T_air_C_val,
        T_dead_C_val,
        initial_Ts_guess_C_val
    )

    # --- Print Key Outputs for Scenario 2 ---
    if results_scenario2.get('solver_success'):
        print(f"Solver converged: {results_scenario2['solver_message']}")
        print(f"Solved TEG Hot Surface Temp (Ts): {results_scenario2['Ts_C']:.2f} °C")
        print(f"Heat Conducted into TEG: {results_scenario2['q_conducted_input_W_m2']:.2f} W/m^2")
        print(f"Heat through Stack (q_stack): {results_scenario2['q_stack']:.2f} W/m^2")
        print(f"Heat Loss (Radiation from TEG exposed surf): {results_scenario2['q_rad_loss_teg_exposed']:.2f} W/m^2")
        print(f"Heat Loss (Convection from TEG exposed surf): {results_scenario2['q_conv_loss_teg_exposed']:.2f} W/m^2")
        # Verify energy balance: q_conducted_input = q_stack + q_rad_loss + q_conv_loss
        balance_check = results_scenario2['q_conducted_input_W_m2'] - (results_scenario2['q_stack'] + results_scenario2['q_rad_loss_teg_exposed'] + results_scenario2['q_conv_loss_teg_exposed'])
        print(f"Energy Balance Check (should be near 0): {balance_check:.4f} W/m^2")

        print(f"\nExergy Analysis (Scenario 2):")
        print(f"Total Exergy Input: {results_scenario2['ex_in_total']:.2f} W/m^2")
        print(f"Exergy Recovered by Fluid: {results_scenario2['ex_recovered_fluid']:.2f} W/m^2 (Electricity Potential Basis)")
        print(f"Exergy System Efficiency (eta_ex_system): {results_scenario2['eta_ex_system']:.3f}")

        print(f"Exergy Destruction in TEG: {results_scenario2['ex_dest_TEG']:.2f} W/m^2")
        print(f"Exergy Destruction in HX Conduction: {results_scenario2['ex_dest_HX_cond']:.2f} W/m^2")
        print(f"Exergy Destruction in HX Convection: {results_scenario2['ex_dest_HX_conv']:.2f} W/m^2")
        print(f"Exergy Loss (Radiation from TEG exposed surf): {results_scenario2['ex_loss_rad_teg_exposed']:.2f} W/m^2")
        print(f"Exergy Loss (Convection from TEG exposed surf): {results_scenario2['ex_loss_conv_teg_exposed']:.2f} W/m^2")
        print(f"Exergy Unaccounted/Residual: {results_scenario2['ex_dest_unaccounted']:.4f} W/m^2")

        # --- Exergy Distribution Pie Chart (Scenario 2) ---
        ex_in = results_scenario2['ex_in_total']
        if ex_in > 0: # Proceed with plotting only if there's exergy input
            ex_recovered = results_scenario2['ex_recovered_fluid']
            ex_dest_TEG = results_scenario2['ex_dest_TEG']
            ex_dest_HX_cond = results_scenario2['ex_dest_HX_cond']
            ex_dest_HX_conv = results_scenario2['ex_dest_HX_conv'] # Added this
            ex_loss_rad = results_scenario2['ex_loss_rad_teg_exposed']
            ex_loss_conv = results_scenario2['ex_loss_conv_teg_exposed']
            # Sum of known destructions and losses
            ex_accounted_losses_destructions = ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_conv +  ex_loss_rad + ex_loss_conv
            # "Other" is what's left from input after accounting for recovered and known losses/destructions
            ex_other_or_unaccounted = ex_in - (ex_recovered + ex_accounted_losses_destructions)
            # Ensure non-negative, can happen due to float precision or model structure
            ex_other_or_unaccounted = max(0, ex_other_or_unaccounted)


            exergy_labels = [
                f"Recovered ({ex_recovered:.1f} W/m²)",
                f"TEG Dest. ({ex_dest_TEG:.1f} W/m²)",
                f"HX Cond. Dest. ({ex_dest_HX_cond:.1f} W/m²)",
                f"HX Conv. Dest. ({ex_dest_HX_conv:.1f} W/m²)", # Added
                f"Rad. Loss (TEG surf) ({ex_loss_rad:.1f} W/m²)",
                f"Conv. Loss (TEG surf) ({ex_loss_conv:.1f} W/m²)",
                f"Other/Unaccounted ({ex_other_or_unaccounted:.1f} W/m²)"
            ]
            exergy_vals = [ex_recovered, ex_dest_TEG, ex_dest_HX_cond, ex_dest_HX_conv, ex_loss_rad, ex_loss_conv, ex_other_or_unaccounted]
            # Filter out zero or very small values for cleaner pie chart
            threshold = 0.005 * sum(exergy_vals) # e.g. 0.5% of total
            exergy_labels_filtered = [label for val, label in zip(exergy_vals, exergy_labels) if val > threshold]
            exergy_vals_filtered = [val for val in exergy_vals if val > threshold]


            plt.figure(figsize=(8, 8)) # Increased size for better label readability
            plt.pie(exergy_vals_filtered, labels=exergy_labels_filtered, autopct='%1.1f%%', startangle=90)
            plt.title(f'Exergy Distribution for Conducted Input ({q_conducted_input_W_m2_val:.0f} W/m²)\nTotal Exergy In: {ex_in:.1f} W/m²')
            plt.axis('equal')
            plt.tight_layout() # Adjust layout

            # --- Stack Thermal Resistance Pie Chart ---
            # These are the resistances from TEG hot side to average fluid temp
            R_stack_components = [R_TEG_A_val, R_HX_cond_A_val, R_HX_conv_A_val]
            R_labels = [
                f"TEG Resist. ({R_TEG_A_val:.4f})",
                f"HX Cond. Resist. ({R_HX_cond_A_val:.4f})",
                f"HX Conv. Resist. ({R_HX_conv_A_val:.4f})"
            ]
            plt.figure(figsize=(7, 7))
            plt.pie(R_stack_components, labels=R_labels, autopct='%1.1f%%', startangle=90)
            plt.title('TEG Stack Thermal Resistance Breakdown (K.m²/W)')
            plt.axis('equal')
            plt.tight_layout()


            # --- Waterfall Plot: Energy Breakdown (Scenario 2) ---
            q_input = results_scenario2['q_conducted_input_W_m2']
            q_stack_val = results_scenario2['q_stack']
            q_rad_loss_val = results_scenario2['q_rad_loss_teg_exposed']
            q_conv_loss_val = results_scenario2['q_conv_loss_teg_exposed']

            stages_energy = ['Conducted Input', 'Rad. Loss (TEG surf)', 'Conv. Loss (TEG surf)', 'Heat to Fluid (Stack)']
            # Input is positive, losses are negative, final recovered is positive in context of what's left
            # Waterfall typically shows flow: Input -> what remains after loss1 -> what remains after loss2 -> final useful
            # So delta for useful part is just its value.
            # Initial input, then subtract losses, then the useful part is what's left.
            # Let's show input, then losses, then the part that goes to stack.
            # q_input = q_stack_val + q_rad_loss_val + q_conv_loss_val
            values_energy_waterfall = [q_input, -q_rad_loss_val, -q_conv_loss_val, q_stack_val]
            # The last element isn't a delta but the final result after subtractions
            # Or, Input, Loss1, Loss2, Useful (all positive, but interpret losses as subtractions)

            data_energy = {
                'labels': ['Conducted Input', 'Rad. Loss (TEG surf)', 'Conv. Loss (TEG surf)', 'Net to Fluid (Stack)'],
                'values': [q_input, -q_rad_loss_val, -q_conv_loss_val, q_stack_val] # Losses are negative
            }
            # For a true waterfall, we need to calculate running totals for the 'bottom'
            running_total_energy = 0
            bottoms_energy = []
            plot_values_energy = []

            # Start with full input
            bottoms_energy.append(0)
            plot_values_energy.append(q_input)
            running_total_energy = q_input

            # Radiative loss
            bottoms_energy.append(running_total_energy - q_rad_loss_val) # top of rad loss bar
            plot_values_energy.append(-q_rad_loss_val) # negative value for loss
            running_total_energy -= q_rad_loss_val

            # Convective loss
            bottoms_energy.append(running_total_energy - q_conv_loss_val) # top of conv loss bar
            plot_values_energy.append(-q_conv_loss_val) # negative value for loss
            running_total_energy -= q_conv_loss_val

            # Heat to stack (should be equal to running_total_energy now)
            bottoms_energy.append(0) # starts from baseline after losses
            plot_values_energy.append(q_stack_val)


            plt.figure(figsize=(8, 6))
            bar_colors_energy = ['blue', 'red', 'orange', 'green']
            plt.bar(data_energy['labels'], plot_values_energy, color=bar_colors_energy, bottom=bottoms_energy) # exclude last bottom for q_stack
            plt.xticks(rotation=25, ha='right')
            plt.ylabel('Energy Flux (W/m²)')
            plt.title('Waterfall: Energy Balance (Scenario 2)')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()


            # --- Waterfall Plot: Exergy Breakdown (Scenario 2) ---
            ex_labels_waterfall = [
                'Total Exergy Input',
                'Rad. Ex. Loss (TEG surf)',
                'Conv. Ex. Loss (TEG surf)',
                'TEG Ex. Dest.',
                'HX Cond. Ex. Dest.',
                'HX Conv. Ex. Dest.', # Added
                'Unaccounted Ex. Dest.',
                'Recovered Exergy'
            ]
            ex_values_waterfall = [
                ex_in,
                -ex_loss_rad, # negative for loss
                -ex_loss_conv, # negative for loss
                -ex_dest_TEG,  # negative for destruction
                -ex_dest_HX_cond, # negative for destruction
                -ex_dest_HX_conv,  # negative for destruction
                -ex_other_or_unaccounted, # negative for destruction/unaccounted
                ex_recovered # positive as the final desired output
            ]

            running_total_exergy = 0
            bottoms_exergy = []
            plot_values_exergy = []

            # Start with full exergy input
            bottoms_exergy.append(0)
            plot_values_exergy.append(ex_in)
            running_total_exergy = ex_in

            # Subsequent losses/destructions
            for val in ex_values_waterfall[1:-1]: # iterate through negative values
                bottoms_exergy.append(running_total_exergy + val) # val is negative, so this is subtraction
                plot_values_exergy.append(val)
                running_total_exergy += val

            # Recovered exergy (should be equal to running_total_exergy)
            bottoms_exergy.append(0) # starts from baseline after losses
            plot_values_exergy.append(ex_recovered)


            plt.figure(figsize=(10, 7)) # Wider for more labels
            bar_colors_exergy = ['blue', 'red', 'salmon', 'indianred', 'lightcoral', 'rosybrown', 'grey', 'green'] # Expanded colors
            plt.bar(ex_labels_waterfall, plot_values_exergy, color=bar_colors_exergy, bottom=bottoms_exergy)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Exergy Flow (W/m²)')
            plt.title('Waterfall: Exergy Balance (Scenario 2)')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            plt.show()
        else:
            print("Exergy input is zero or negative, skipping exergy plots for Scenario 2.")

    else:
        print(f"Solver FAILED for Scenario 2: {results_scenario2.get('solver_message', 'Unknown error')}")
        print(f"Energy balance residual at last Ts guess: {results_scenario2.get('energy_balance_residual')}")


    # --- DataFrame Output for Scenario 2 ---
    if results_scenario2.get('solver_success'):
        # Define units for each parameter in Scenario 2 results
        units_sc2 = {
            'solver_success': '', 'solver_message': '',
            'Ts_C': '°C',
            'q_conducted_input_W_m2': 'W/m²',
            'q_stack': 'W/m²',
            'q_rad_loss_teg_exposed': 'W/m²',
            'q_conv_loss_teg_exposed': 'W/m²',
            'ex_in_total': 'W/m²',
            'ex_recovered_fluid': 'W/m²',
            'ex_dest_TEG': 'W/m²',
            'ex_dest_HX_cond': 'W/m²',
            'ex_dest_HX_conv': 'W/m²',
            'ex_loss_rad_teg_exposed': 'W/m²',
            'ex_loss_conv_teg_exposed': 'W/m²',
            'ex_dest_unaccounted': 'W/m²',
            'eta_ex_system': '',
            'T_TEG_cold_K': 'K', 'T_HX_wall_K': 'K', 'T_f_avg_K': 'K'
        }
        # Format the results_scenario2 values
        formatted_results_sc2 = {k: f"{v:.4f}" if isinstance(v, (float, np.float_)) else str(v) for k, v in results_scenario2.items()}
        df_results_sc2 = pd.DataFrame(list(formatted_results_sc2.items()), columns=['Parameter', 'Value'])
        df_results_sc2['Unit'] = df_results_sc2['Parameter'].map(units_sc2).fillna('')
        print("\n--- Scenario 2 Detailed Results Table ---")
        print(df_results_sc2.to_string(index=False))


    # --- DataFrame Output for Resistances ---
    # These are the area-specific resistances used in calculations
    resistances_data = {
        "R_TEG_A (TEG internal)": R_TEG_A_val,
        "R_HX_cond_A (HX wall)": R_HX_cond_A_val,
        "R_HX_conv_A (HX fluid film)": R_HX_conv_A_val
    }
    df_resistances = pd.DataFrame(list(resistances_data.items()), columns=['Parameter', 'Value'])
    df_resistances['Unit'] = 'K.m²/W'
    print("\n--- TEG Stack Component Resistances (Area-Specific) ---")
    print(df_resistances.to_string(index=False))

    import numpy as np  # For ln and array operations
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Constants and Helper Functions ---

    BOLTZMANN_CONST = 1.38e-23  # J/K (unused in conduction-only model)


    def celsius_to_kelvin(T_celsius):
        return T_celsius + 273.15


    def kelvin_to_celsius(T_kelvin):
        return T_kelvin - 273.15


    # --- Thermal Resistance Calculation ---

    def resistance_conduction(thickness_m, area_m2, k_W_mK):
        """
        Calculate conduction thermal resistance: R = thickness / (k * area).
        """
        return thickness_m / (k_W_mK * area_m2)


    # --- Core Thermal Calculation Functions ---

    def calculate_heat_flux_through_stack(
            T_source_K,
            T_in_fluid_K,
            T_out_fluid_K,
            R_source_cond_A,
            R_TEG_A,
            R_HX_cond_A,
            R_HX_conv_A
    ):
        """
        Compute heat flux from source to fluid via conduction through stack.
        """
        T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
        R_stack_A = R_source_cond_A + R_TEG_A + R_HX_cond_A + R_HX_conv_A
        if R_stack_A <= 0:
            return 0.0
        return max(0.0, (T_source_K - T_f_avg_K) / R_stack_A)


    # --- Exergy Analysis Functions ---

    def calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K):
        if q_stack <= 0 or T_out_fluid_K == T_in_fluid_K:
            return 0.0
        delta_T = T_out_fluid_K - T_in_fluid_K
        return q_stack - (q_stack / delta_T) * T_dead_K * np.log(T_out_fluid_K / T_in_fluid_K)


    def calculate_exergy_destruction_component(q_flux, T_hot_K, T_cold_K, T_dead_K):
        if q_flux <= 0 or T_hot_K <= T_cold_K:
            return 0.0
        return q_flux * T_dead_K * (1.0 / T_cold_K - 1.0 / T_hot_K)


    # --- Analysis for Conduction-Only Source (Case 1) ---

    def analyze_conduction_case_with_exergy(
            T_source_C,
            T_in_fluid_C,
            T_out_fluid_C,
            thickness_source_m,
            k_source_W_mK,
            area_m2,
            R_TEG_A,
            R_HX_cond_A,
            R_HX_conv_A,
            T_dead_C
    ):
        # Convert temperatures
        T_source_K = celsius_to_kelvin(T_source_C)
        T_in_K = celsius_to_kelvin(T_in_fluid_C)
        T_out_K = celsius_to_kelvin(T_out_fluid_C)
        T_dead_K = celsius_to_kelvin(T_dead_C)

        # Compute conduction resistance from source surface
        R_source_cond_A = resistance_conduction(thickness_source_m, area_m2, k_source_W_mK)

        # Heat flux through the stack
        q_stack = calculate_heat_flux_through_stack(
            T_source_K, T_in_K, T_out_K,
            R_source_cond_A, R_TEG_A, R_HX_cond_A, R_HX_conv_A
        )

        # Temperature drops
        T_TEG_cold_K = T_source_K - q_stack * R_source_cond_A - q_stack * R_TEG_A
        T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A

        # Exergy streams
        ex_in_source = q_stack * (1 - T_dead_K / T_source_K) if q_stack > 0 else 0.0
        ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)
        ex_dest_source = calculate_exergy_destruction_component(q_stack, T_source_K, T_TEG_cold_K + q_stack * R_TEG_A,
                                                                T_dead_K)
        ex_dest_TEG = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K + q_stack * R_TEG_A,
                                                             T_HX_wall_K + q_stack * R_HX_cond_A, T_dead_K)
        ex_dest_cond = calculate_exergy_destruction_component(q_stack, T_HX_wall_K + q_stack * R_HX_cond_A, T_HX_wall_K,
                                                              T_dead_K)
        ex_dest_conv = max(0.0, ex_in_source - ex_dest_source - ex_dest_TEG - ex_dest_cond - ex_recovered)

        return {
            'q_stack': q_stack,
            'ex_in_source': ex_in_source,
            'ex_recovered': ex_recovered,
            'ex_dest_source': ex_dest_source,
            'ex_dest_TEG': ex_dest_TEG,
            'ex_dest_cond': ex_dest_cond,
            'ex_dest_conv': ex_dest_conv
        }


    # --- Example Run ---
    if __name__ == '__main__':
        # Parameters
        area = 1.0  # m²
        T_source_C = 500.0  # °C
        T_in_fluid_C = 40.0  # °C
        T_out_fluid_C = 90.0  # °C
        T_dead_C = 25.0  # °C

        # Material layers for source conduction
        thickness_source = 1e-3  # m (adhesive layer)
        k_source = 0.5  # W/mK

        # Stack resistances from existing components (per unit area)
        R_TEG_A = resistance_conduction(4e-3, area, 1.5)  # TEG
        R_HX_cond_A = resistance_conduction(2e-3, area, 167.0)  # HX wall
        R_HX_conv_A = 1 / (2000.0 * area)  # HX convection

        # Analyze
        results = analyze_conduction_case_with_exergy(
            T_source_C,
            T_in_fluid_C,
            T_out_fluid_C,
            thickness_source,
            k_source,
            area,
            R_TEG_A,
            R_HX_cond_A,
            R_HX_conv_A,
            T_dead_C
        )

        # Display results
        print("Conduction-Only Case Results:")
        for key, val in results.items():
            print(f"{key:15s}: {val:.3f}")

        # Plot exergy distribution
        labels = list(results.keys())[1:]
        values = [results[k] for k in labels]
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Exergy Distribution (Conduction Only)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

