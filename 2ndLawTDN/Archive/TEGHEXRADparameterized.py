import numpy as np  # For ln, and to handle potential division by zero gracefully
from scipy.optimize import fsolve

# --- Constants and Helper Functions ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)

def celsius_to_kelvin(T_celsius):
    """Converts temperature from Celsius to Kelvin."""
    return T_celsius + 273.15

def kelvin_to_celsius(T_kelvin):
    """Converts temperature from Kelvin to Celsius."""
    return T_kelvin - 273.15

# --- Core Thermal Calculation Functions ---

def calculate_heat_flux_through_stack(
    T_hot_surface_K,
    T_in_fluid_K,
    T_out_fluid_K,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A
):
    """
    Calculates heat flux through the TEG-MCHX stack given a known hot surface temperature.
    """
    T_f_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A
    if R_stack_A <= 0:
        return 0 if T_hot_surface_K <= T_f_avg_K else np.inf
    return max(0, (T_hot_surface_K - T_f_avg_K) / R_stack_A)


def calculate_radiative_heat_flux_loss(
    T_surface_K,
    T_ambient_rad_K,
    emissivity
):
    """Radiative heat flux loss from a surface."""
    return emissivity * STEFAN_BOLTZMANN_CONST * (T_surface_K**4 - T_ambient_rad_K**4)


def calculate_convective_heat_flux_loss(
    T_surface_K,
    T_air_K,
    h_conv_air
):
    """Convective heat flux loss from a surface."""
    return h_conv_air * (T_surface_K - T_air_K)

# --- Exergy Analysis Functions ---

def calculate_exergy_recovered_by_fluid(q_stack, T_in_fluid_K, T_out_fluid_K, T_dead_K):
    """Exergy flux recovered by the heating fluid."""
    if q_stack <= 0 or T_out_fluid_K == T_in_fluid_K:
        return 0.0
    delta_T = T_out_fluid_K - T_in_fluid_K
    return q_stack - (q_stack / delta_T) * T_dead_K * np.log(T_out_fluid_K / T_in_fluid_K)


def calculate_exergy_destruction_component(q_flux, T_hot_K, T_cold_K, T_dead_K):
    """Exergy destruction flux for heat transfer from T_hot to T_cold."""
    if q_flux <= 0 or T_hot_K <= T_cold_K:
        return 0.0
    return q_flux * T_dead_K * (1.0 / T_cold_K - 1.0 / T_hot_K)

# --- Case 1: Known T_source (radiation source) ---
def analyze_case1_with_exergy(
    T_source_C,
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A,
    T_dead_C,
    emissivity_source=None,
    T_ambient_rad_C=None
):
    """
    Case 1: specified source temperature (from radiation source).
    Uses T_source in Kelvin as the hot boundary for the TEG stack.
    Returns heat fluxes and exergy terms per unit area.
    """
    # Convert temperatures to Kelvin
    T_source_K = celsius_to_kelvin(T_source_C)
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)
    T_f_avg_K = (T_in_K + T_out_K) / 2

    # Heat flux through stack
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A
    q_stack = calculate_heat_flux_through_stack(
        T_source_K, T_in_K, T_out_K,
        R_TEG_A, R_HX_cond_A, R_HX_conv_A
    )

    # Intermediate temperatures in the stack
    T_TEG_cold_K = T_source_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A

    # Exergy input from source at T_source
    ex_in_source = q_stack * (1 - T_dead_K / T_source_K) if q_stack > 0 else 0.0

    # Exergy recovered by fluid
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)

    # Exergy destruction in each stack component
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, T_source_K, T_TEG_cold_K, T_dead_K)
    ex_dest_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)
    ex_dest_conv_fluid = ex_in_source - ex_dest_TEG - ex_dest_cond - ex_recovered
    ex_dest_conv_fluid = max(0.0, ex_dest_conv_fluid)

    # Radiative loss from source and its exergy
    q_rad_loss = 0.0
    ex_rad_loss = 0.0
    if emissivity_source is not None and T_ambient_rad_C is not None:
        T_amb_rad_K = celsius_to_kelvin(T_ambient_rad_C)
        q_rad_loss = calculate_radiative_heat_flux_loss(T_source_K, T_amb_rad_K, emissivity_source)
        ex_rad_loss = q_rad_loss * (1 - T_dead_K / T_source_K)

    return {
        'q_stack': q_stack,
        'q_rad_loss_source': q_rad_loss,
        'ex_in_source': ex_in_source,
        'ex_recovered': ex_recovered,
        'ex_dest_TEG': ex_dest_TEG,
        'ex_dest_cond': ex_dest_cond,
        'ex_dest_conv_fluid': ex_dest_conv_fluid,
        'ex_rad_loss_source': ex_rad_loss
    }

# --- Case 2: Known q_source, solve for surface temperature Ts ---
def analyze_case2_with_exergy(
    q_source_total,
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A,
    emissivity_Ts_surface,
    T_surr_rad_C,
    h_conv_air,
    T_air_C,
    T_dead_C,
    initial_Ts_guess_C=200.0
):
    """
    Case 2: specified total heat flux from source; solves for surface temperature Ts.
    Returns Ts, heat fluxes, and exergy terms per unit area.
    """
    # Convert to Kelvin
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_f_avg_K = (T_in_K + T_out_K) / 2
    T_surr_rad_K = celsius_to_kelvin(T_surr_rad_C)
    T_air_K = celsius_to_kelvin(T_air_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)

    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A

    # Energy balance for fsolve
    def energy_balance_Ts(Ts_K):
        if Ts_K <= 0:
            return 1e9
        q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A) if R_stack_A > 0 else 0
        q_rad = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_Ts_surface)
        q_conv = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air)
        return q_source_total - (q_stack + q_rad + q_conv)

    # Solve for Ts
    Ts_K, info, ier, mesg = fsolve(energy_balance_Ts, celsius_to_kelvin(initial_Ts_guess_C), full_output=True)
    Ts_K = Ts_K[0]
    success = (ier == 1)
    Ts_C = kelvin_to_celsius(Ts_K) if success else np.nan

    # Initialize result dict
    results = {'solver_success': success, 'solver_message': mesg.strip(), 'Ts_C': Ts_C}
    if not success:
        return results

    # Compute fluxes
    q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A)
    q_rad = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_Ts_surface)
    q_conv = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air)
    results.update({'q_stack': q_stack, 'q_rad_loss': q_rad, 'q_conv_loss': q_conv,
                    'q_balance_check': q_source_total - (q_stack + q_rad + q_conv)})

    # Intermediate temperatures
    T_TEG_cold_K = Ts_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A

    # Exergy terms
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)
    ex_in_total = q_source_total * (1 - T_dead_K / Ts_K)
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, Ts_K, T_TEG_cold_K, T_dead_K)
    ex_dest_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)
    ex_q_rad = q_rad * (1 - T_dead_K / Ts_K)
    ex_q_conv = q_conv * (1 - T_dead_K / Ts_K)
    ex_dest_stack = ex_dest_TEG + ex_dest_cond + (ex_in_total - ex_dest_TEG - ex_dest_cond - ex_recovered)
    ex_dest_system = ex_in_total - ex_recovered
    eta_ex_system = ex_recovered / ex_in_total if ex_in_total > 0 else 0.0

    # Pack results
    results.update({
        'ex_in_total': ex_in_total,
        'ex_recovered': ex_recovered,
        'ex_dest_TEG': ex_dest_TEG,
        'ex_dest_cond': ex_dest_cond,
        'ex_q_rad_loss': ex_q_rad,
        'ex_q_conv_loss': ex_q_conv,
        'ex_dest_total_stack': ex_dest_stack,
        'ex_dest_system_total': ex_dest_system,
        'eta_ex_system': eta_ex_system
    })

    return results

# --- Example Usage ---
if __name__ == '__main__':
    # Dead state
    T_dead_C = 25.0

    # Case 1 Demo
    c1 = analyze_case1_with_exergy(
        T_source_C=300.0, T_in_fluid_C=40.0, T_out_fluid_C=90.0,
        R_TEG_A=0.01, R_HX_cond_A=0.002, R_HX_conv_A=0.005,
        T_dead_C=T_dead_C, emissivity_source=0.8, T_ambient_rad_C=25.0
    )
    print('Case 1 Results:', {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in c1.items()})

    # Case 2 Demo
    c2 = analyze_case2_with_exergy(
        q_source_total=30000.0, T_in_fluid_C=40.0, T_out_fluid_C=90.0,
        R_TEG_A=0.01, R_HX_cond_A=0.002, R_HX_conv_A=0.005,
        emissivity_Ts_surface=0.8, T_surr_rad_C=25.0,
        h_conv_air=15.0, T_air_C=25.0,
        T_dead_C=T_dead_C, initial_Ts_guess_C=1000.0
    )
    print('Case 2 Results:', c2)
