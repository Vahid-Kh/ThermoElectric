import numpy as np # For ln, and to handle potential division by zero gracefully
from scipy.optimize import fsolve

# --- Constants and Helper Functions ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)

def celsius_to_kelvin(T_celsius):
    """Converts temperature from Celsius to Kelvin."""
    return T_celsius + 273.15

def kelvin_to_celsius(T_kelvin):
    """Converts temperature from Kelvin to Celsius."""
    return T_kelvin - 273.15

# --- Core Thermal Calculation Functions (from previous response, with minor adjustments) ---

def calculate_average_fluid_temperature(T_in_fluid_C, T_out_fluid_C):
    """Calculates the average fluid temperature in Celsius."""
    return (T_in_fluid_C + T_out_fluid_C) / 2

def calculate_heat_flux_through_stack(
    T_hot_surface_K, # TEG hot side temperature in Kelvin
    T_in_fluid_K,
    T_out_fluid_K,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A
):
    """
    Calculates the heat flux through the TEG-MCHX stack.
    This function is generic for calculating q_stack if T_hot_surface is known.
    """
    T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A
    if R_stack_A <= 0: # Avoid division by zero or negative resistance
        # print("Warning: R_stack_A is zero or negative.")
        return 0 if T_hot_surface_K <= T_f_avg_K else float('inf')
    q_stack = (T_hot_surface_K - T_f_avg_K) / R_stack_A
    return q_stack

def calculate_radiative_heat_flux_loss(
    T_surface_K, # Emitting surface temperature in Kelvin
    T_ambient_rad_K, # Ambient temperature for radiation in Kelvin
    emissivity
):
    """Calculates the radiative heat flux loss from a surface."""
    q_rad_loss = emissivity * STEFAN_BOLTZMANN_CONST * (T_surface_K**4 - T_ambient_rad_K**4)
    return q_rad_loss

def calculate_convective_heat_flux_loss(
    T_surface_K, # Surface temperature in Kelvin
    T_air_K, # Ambient air temperature for convection in Kelvin
    h_conv_air
):
    """Calculates the convective heat flux loss from a surface."""
    q_conv_loss = h_conv_air * (T_surface_K - T_air_K)
    return q_conv_loss

# --- Exergy Analysis Functions ---

def calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K):
    """
    Calculates the exergy flux recovered by the heating fluid.
    ex_recovered = q_stack * (1 - T_dead_K / T_lm_exergy_K)
    where T_lm_exergy_K = (T_out_fluid_K - T_in_fluid_K) / ln(T_out_fluid_K / T_in_fluid_K)
    """
    if q_stack == 0 or T_out_fluid_K == T_in_fluid_K:
        return 0.0
    if T_out_fluid_K <= 0 or T_in_fluid_K <=0: # Should not happen with K temps
        return float('nan')

    # Avoid division by zero if T_out_fluid_K is very close to T_in_fluid_K but q_stack is not zero
    # This case should ideally be handled by q_stack being zero.
    # However, for numerical stability with very small differences:
    delta_T_fluid = T_out_fluid_K - T_in_fluid_K
    if abs(delta_T_fluid) < 1e-6: # Effectively T_out = T_in
         return 0.0


    # Using L'Hopital's rule for ln(x)/(x-1) as x->1, if T_out/T_in is close to 1.
    # T_lm_exergy_K = delta_T_fluid / np.log(T_out_fluid_K / T_in_fluid_K)
    # ex_recovered = q_stack * (1 - T_dead_K / T_lm_exergy_K)

    # Direct formula: m*cp*[(T_out-T_in) - T_dead*ln(T_out/T_in)]
    # m*cp = q_stack / (T_out-T_in)
    ex_recovered = q_stack - (q_stack / delta_T_fluid) * T_dead_K * np.log(T_out_fluid_K / T_in_fluid_K)
    return ex_recovered


def calculate_exergy_destruction_component(q_flux, T_hot_K, T_cold_K, T_dead_K):
    """Calculates exergy destruction flux for heat transfer Q from T_hot to T_cold."""
    if q_flux == 0 or T_hot_K == T_cold_K : # No heat transfer or no temp diff
        return 0.0
    if T_hot_K <= 0 or T_cold_K <= 0: # Invalid temperatures
        return float('nan')
    # Ensure T_hot_K >= T_cold_K for this formula to represent destruction due to Q flowing from H to L
    # If q_flux is already directional (e.g. q_stack is always positive if T_hot > T_cold_effective)
    # then this check might not be needed if T_hot_K and T_cold_K are correctly ordered.
    if T_hot_K < T_cold_K: # Should not happen if T_hot and T_cold are boundaries for positive q_flux
        # print(f"Warning: T_hot_K ({T_hot_K}) < T_cold_K ({T_cold_K}) in exergy destruction calculation.")
        return float('nan') # Or handle as heat flowing other way round, this indicates an issue.

    ex_dest = q_flux * T_dead_K * (1.0 / T_cold_K - 1.0 / T_hot_K)
    return max(0, ex_dest) # Destruction cannot be negative

# --- Case 1 Analysis (Known T0) ---
def analyze_case1_with_exergy(
    T0_C,
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A,
    T_dead_C,
    emissivity_T0_surface=None, # Optional: for radiative loss from T0
    T_ambient_rad_C=None      # Optional: for radiative loss from T0
):
    """
    Analyzes Case 1: Known T0 (TEG Hot Side Temp).
    Calculates heat flux, exergy recovered, and exergy destructions.
    All area-specific resistances (R_A in m^2*K/W), fluxes in W/m^2.
    """
    T0_K = celsius_to_kelvin(T0_C)
    T_in_fluid_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_fluid_K = celsius_to_kelvin(T_out_fluid_C)
    T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
    T_dead_K = celsius_to_kelvin(T_dead_C)

    # Calculate heat flux through stack
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A
    if R_stack_A <= 0:
        q_stack = 0  # Or handle error
    else:
        q_stack = (T0_K - T_f_avg_K) / R_stack_A

    if q_stack < 0: # If T0 is lower than T_f_avg, heat flows other way or no heat transfer.
        # print("Warning: q_stack is negative in Case 1. T0 < T_f_avg. Setting q_stack and exergies to 0.")
        q_stack = 0.0 # For this analysis, assume we are interested in heat from T0 to fluid

    results = {"q_stack": q_stack}

    # Intermediate temperatures (all in Kelvin)
    # T0_K -> (R_TEG_A) -> T_TEG_cold_K -> (R_HX_cond_A) -> T_HX_wall_K -> (R_HX_conv_A) -> T_f_avg_K
    T_TEG_cold_K = T0_K - q_stack * R_TEG_A if q_stack > 0 else T0_K
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A if q_stack > 0 else T_TEG_cold_K
    # T_HX_wall_K is the temperature of the wall from which convection to fluid happens.
    # Check: T_HX_wall_K should also be T_f_avg_K + q_stack * R_HX_conv_A (if q_stack > 0)

    # Exergy Analysis
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K)
    results["ex_recovered_fluid"] = ex_recovered

    # Exergy input to the stack (from source at T0_K that results in q_stack)
    ex_in_stack = q_stack * (1 - T_dead_K / T0_K) if T0_K > 0 and q_stack > 0 else 0
    results["ex_in_stack_at_T0"] = ex_in_stack

    # Component-wise exergy destruction in the stack
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, T0_K, T_TEG_cold_K, T_dead_K)
    ex_dest_HX_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)

    # Exergy destruction in fluid heating (from T_HX_wall_K to fluid T_in->T_out)
    ex_in_to_fluid_conv_process = q_stack * (1 - T_dead_K / T_HX_wall_K) if T_HX_wall_K > 0 and q_stack > 0 else 0
    ex_dest_HX_fluid_heating = ex_in_to_fluid_conv_process - ex_recovered
    ex_dest_HX_fluid_heating = max(0, ex_dest_HX_fluid_heating) # Destruction cannot be negative

    results["ex_dest_TEG"] = ex_dest_TEG
    results["ex_dest_HX_cond"] = ex_dest_HX_cond
    results["ex_dest_HX_fluid_heating"] = ex_dest_HX_fluid_heating

    ex_dest_total_stack = ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_fluid_heating
    results["ex_dest_total_stack"] = ex_dest_total_stack
    # Alternative check for ex_dest_total_stack: ex_in_stack - ex_recovered. Should be close.
    results["ex_dest_total_stack_check"] = ex_in_stack - ex_recovered


    # Exergy efficiency of the stack process (relative to exergy input at T0 for q_stack)
    if ex_in_stack > 0:
        eta_ex_stack = ex_recovered / ex_in_stack
    else:
        eta_ex_stack = 0.0
    results["eta_ex_stack"] = eta_ex_stack

    # Optional: Radiative loss from T0 surface
    if emissivity_T0_surface is not None and T_ambient_rad_C is not None:
        T_ambient_rad_K = celsius_to_kelvin(T_ambient_rad_C)
        q_rad_loss_T0 = calculate_radiative_heat_flux_loss(T0_K, T_ambient_rad_K, emissivity_T0_surface)
        results["q_rad_loss_from_T0"] = q_rad_loss_T0
        # Exergy associated with this heat loss (if T_ambient_rad_K is T_dead_K, this is exergy destroyed)
        ex_q_rad_loss_T0 = q_rad_loss_T0 * (1 - T_dead_K / T0_K) if T0_K > 0 else 0
        results["ex_q_rad_loss_from_T0"] = ex_q_rad_loss_T0
    else:
        results["q_rad_loss_from_T0"] = 0.0
        results["ex_q_rad_loss_from_T0"] = 0.0

    return results

# --- Case 2 Analysis (Known q_source, Solve for Ts) ---
def analyze_case2_with_exergy(
    q_source_total, # Total external heat flux supplied to the surface (W/m^2)
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A,
    emissivity_Ts_surface,
    T_surr_rad_C, # Surrounding temp for radiation from Ts
    h_conv_air,   # Convective HTC from Ts to air
    T_air_C,      # Ambient air temp for convection from Ts
    T_dead_C,
    initial_Ts_guess_C=200.0
):
    """
    Analyzes Case 2: Known q_source, solves for surface temperature Ts.
    Calculates Ts, heat fluxes, exergy recovered, and exergy destructions.
    """
    T_in_fluid_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_fluid_K = celsius_to_kelvin(T_out_fluid_C)
    T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
    T_surr_rad_K = celsius_to_kelvin(T_surr_rad_C)
    T_air_K = celsius_to_kelvin(T_air_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)

    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A

    # Energy balance equation for fsolve: q_source - q_stack - q_rad_loss - q_conv_loss = 0
    def energy_balance_Ts(Ts_K_guess):
        if Ts_K_guess <=0 : # Invalid temperature
            return 1e9 # Large residual

        # q_stack based on current Ts_K_guess
        current_q_stack = (Ts_K_guess - T_f_avg_K) / R_stack_A if R_stack_A > 0 else (float('inf') if Ts_K_guess > T_f_avg_K else 0)
        current_q_stack = max(0, current_q_stack) # Assume heat only flows from Ts to fluid

        current_q_rad_loss = calculate_radiative_heat_flux_loss(Ts_K_guess, T_surr_rad_K, emissivity_Ts_surface)
        current_q_conv_loss = calculate_convective_heat_flux_loss(Ts_K_guess, T_air_K, h_conv_air)

        return q_source_total - current_q_stack - current_q_rad_loss - current_q_conv_loss

    initial_Ts_K_guess = celsius_to_kelvin(initial_Ts_guess_C)
    Ts_K_solved, info, ier, mesg = fsolve(energy_balance_Ts, initial_Ts_K_guess, full_output=True)

    results = {"solver_message": mesg, "solver_success": (ier == 1)}

    if ier != 1: # Solver failed
        # print(f"Solver failed for Ts in Case 2: {mesg}")
        results.update({
            "Ts_C": float('nan'), "q_stack": float('nan'), "q_rad_loss": float('nan'),
            "q_conv_loss": float('nan'), "ex_recovered_fluid": float('nan'),
            "ex_in_total_at_Ts": float('nan'), "ex_dest_TEG": float('nan'),
            "ex_dest_HX_cond": float('nan'), "ex_dest_HX_fluid_heating": float('nan'),
            "ex_dest_total_stack": float('nan'), "ex_dest_total_stack_check": float('nan'),
            "ex_q_rad_loss_from_Ts": float('nan'), "ex_q_conv_loss_from_Ts": float('nan'),
            "ex_dest_system_total": float('nan'), "eta_ex_system": float('nan')
        })
        return results

    Ts_K = Ts_K_solved[0]
    Ts_C = kelvin_to_celsius(Ts_K)
    results["Ts_C"] = Ts_C

    # Recalculate fluxes with solved Ts_K
    q_stack = (Ts_K - T_f_avg_K) / R_stack_A if R_stack_A > 0 else 0
    q_stack = max(0, q_stack) # Ensure non-negative

    q_rad_loss = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_Ts_surface)
    q_conv_loss = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air)

    results["q_stack"] = q_stack
    results["q_rad_loss"] = q_rad_loss
    results["q_conv_loss"] = q_conv_loss
    results["q_balance_check"] = q_source_total - (q_stack + q_rad_loss + q_conv_loss)


    # Intermediate temperatures
    T_TEG_cold_K = Ts_K - q_stack * R_TEG_A if q_stack > 0 else Ts_K
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A if q_stack > 0 else T_TEG_cold_K

    # Exergy Analysis
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K)
    results["ex_recovered_fluid"] = ex_recovered

    # Total exergy input to the system (associated with q_source_total at Ts_K)
    ex_in_total_at_Ts = q_source_total * (1 - T_dead_K / Ts_K) if Ts_K > 0 and q_source_total > 0 else 0
    results["ex_in_total_at_Ts"] = ex_in_total_at_Ts

    # Exergy destruction in stack components
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, Ts_K, T_TEG_cold_K, T_dead_K)
    ex_dest_HX_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)

    ex_in_to_fluid_conv_process = q_stack * (1 - T_dead_K / T_HX_wall_K) if T_HX_wall_K > 0 and q_stack > 0 else 0
    ex_dest_HX_fluid_heating = ex_in_to_fluid_conv_process - ex_recovered
    ex_dest_HX_fluid_heating = max(0, ex_dest_HX_fluid_heating)

    results["ex_dest_TEG"] = ex_dest_TEG
    results["ex_dest_HX_cond"] = ex_dest_HX_cond
    results["ex_dest_HX_fluid_heating"] = ex_dest_HX_fluid_heating

    ex_dest_total_stack = ex_dest_TEG + ex_dest_HX_cond + ex_dest_HX_fluid_heating
    results["ex_dest_total_stack"] = ex_dest_total_stack

    # Check: exergy input to stack - exergy recovered by fluid
    ex_in_to_stack_at_Ts = q_stack * (1 - T_dead_K / Ts_K) if Ts_K > 0 and q_stack > 0 else 0
    results["ex_dest_total_stack_check"] = ex_in_to_stack_at_Ts - ex_recovered

    # Exergy associated with heat losses (exergy leaving the system with losses)
    # This is exergy that could have been used if T_s was T_dead, but it's lost at T_s.
    ex_q_rad_loss = q_rad_loss * (1 - T_dead_K / Ts_K) if Ts_K > 0 and q_rad_loss > 0 else 0
    ex_q_conv_loss = q_conv_loss * (1 - T_dead_K / Ts_K) if Ts_K > 0 and q_conv_loss > 0 else 0
    # These terms are often considered as part of the total system exergy destruction if the
    # reference "input exergy" is ex_in_total_at_Ts and "useful output" is only ex_recovered.
    results["ex_q_rad_loss_from_Ts"] = ex_q_rad_loss
    results["ex_q_conv_loss_from_Ts"] = ex_q_conv_loss

    # Overall system exergy destruction: (Total Exergy In) - (Useful Exergy Out)
    # Here, useful exergy out is ex_recovered. The exergy with losses is considered destroyed from usefulness.
    ex_dest_system_total = ex_in_total_at_Ts - ex_recovered
    results["ex_dest_system_total"] = ex_dest_system_total
    # Check: ex_dest_system_total should be ex_dest_total_stack + ex_q_rad_loss + ex_q_conv_loss
    # (q_source - q_stack)*(1-T_dead/Ts) = (q_rad_loss+q_conv_loss)*(1-T_dead/Ts)
    # ex_in_total_at_Ts = ex_in_to_stack_at_Ts + ex_q_rad_loss + ex_q_conv_loss
    # ex_dest_system_total = (ex_in_to_stack_at_Ts - ex_recovered) + ex_q_rad_loss + ex_q_conv_loss
    #                      = ex_dest_total_stack_check + ex_q_rad_loss + ex_q_conv_loss
    results["ex_dest_system_total_check"] = ex_dest_total_stack + ex_q_rad_loss + ex_q_conv_loss


    # Overall system exergy efficiency
    if ex_in_total_at_Ts > 0:
        eta_ex_system = ex_recovered / ex_in_total_at_Ts
    else:
        eta_ex_system = 0.0
    results["eta_ex_system"] = eta_ex_system

    return results

# --- Example Usage ---
if __name__ == "__main__":
    # Define Dead State Temperature
    T_DEAD_C = 25.0  # Celsius

    # --- Example for Case 1 ---
    print("--- Case 1 Analysis ---")
    # Parameters from previous example
    R_TEG_A_c1 = 0.01    # m^2*K/W
    R_HX_cond_A_c1 = 0.002 # m^2*K/W
    R_HX_conv_A_c1 = 0.005 # m^2*K/W
    T0_C_c1 = 300.0
    T_in_fluid_C_c1 = 40.0
    T_out_fluid_C_c1 = 90.0
    emissivity_c1 = 0.8
    T_ambient_rad_C_c1 = 25.0

    case1_results = analyze_case1_with_exergy(
        T0_C_c1, T_in_fluid_C_c1, T_out_fluid_C_c1,
        R_TEG_A_c1, R_HX_cond_A_c1, R_HX_conv_A_c1,
        T_DEAD_C,
        emissivity_T0_surface=emissivity_c1,
        T_ambient_rad_C=T_ambient_rad_C_c1
    )
    for key, value in case1_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # --- Example for Case 2 ---
    print("\n--- Case 2 Analysis ---")
    q_source_c2 = 30000.0       # W/m^2
    T_in_fluid_C_c2 = 40.0
    T_out_fluid_C_c2 = 90.0     # Note: T_out is fixed, implies flow rate adjusts or q_stack determines this.
                                # More realistically, T_out might vary.
                                # For this model structure, T_in and T_out define T_f_avg.
                                # q_stack is then determined by (Ts - T_f_avg)/R_stack.
                                # If q_source implies a certain q_stack, T_out could be an output instead.
                                # The current formulation fixes T_in, T_out for T_f_avg calculation.
    R_TEG_A_c2 = 0.01
    R_HX_cond_A_c2 = 0.002
    R_HX_conv_A_c2 = 0.005
    emissivity_c2 = 0.8
    T_surr_rad_C_c2 = 25.0
    h_conv_air_c2 = 15.0        # W/(m^2*K)
    T_air_C_c2 = 25.0
    initial_Ts_guess_c2 = 1000.0

    case2_results = analyze_case2_with_exergy(
        q_source_c2, T_in_fluid_C_c2, T_out_fluid_C_c2,
        R_TEG_A_c2, R_HX_cond_A_c2, R_HX_conv_A_c2,
        emissivity_c2, T_surr_rad_C_c2, h_conv_air_c2, T_air_C_c2,
        T_DEAD_C, initial_Ts_guess_C=initial_Ts_guess_c2
    )

    if case2_results["solver_success"]:
        print(f"  Solver for Ts: {case2_results['solver_message']}")
        for key, value in case2_results.items():
            if key not in ["solver_message", "solver_success"]:
                 if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                 else:
                    print(f"  {key}: {value}")
    else:
        print(f"  Case 2 Solver FAILED: {case2_results['solver_message']}")