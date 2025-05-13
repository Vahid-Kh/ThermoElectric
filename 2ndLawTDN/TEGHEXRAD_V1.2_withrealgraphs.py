import numpy as np  # For ln, and to handle potential division by zero gracefully
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# --- Constants and Helper Functions ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)

def celsius_to_kelvin(T_celsius):
    return T_celsius + 273.15

def kelvin_to_celsius(T_kelvin):
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
    return emissivity * STEFAN_BOLTZMANN_CONST * (T_surface_K**4 - T_ambient_rad_K**4)


def calculate_convective_heat_flux_loss(
    T_surface_K,
    T_air_K,
    h_conv_air
):
    return h_conv_air * (T_surface_K - T_air_K)

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
    # Convert all temperatures to Kelvin
    T_source_K = celsius_to_kelvin(T_source_C)
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)

    # Heat flux through stack
    q_stack = calculate_heat_flux_through_stack(
        T_source_K, T_in_K, T_out_K,
        R_TEG_A, R_HX_cond_A, R_HX_conv_A
    )

    # Temperature drops across components
    T_TEG_cold_K = T_source_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A

    # Exergy input from source
    ex_in_source = q_stack * (1 - T_dead_K / T_source_K) if q_stack > 0 else 0.0

    # Exergy recovered by fluid
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)

    # Exergy destruction in TEG and conduction
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, T_source_K, T_TEG_cold_K, T_dead_K)
    ex_dest_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)

    # Remaining destruction in fluid convection
    ex_dest_conv_fluid = max(0.0, ex_in_source - ex_dest_TEG - ex_dest_cond - ex_recovered)

    # Optional radiative losses and exergy losses
    q_rad_loss = ex_rad_loss = 0.0
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
    # Convert to Kelvin
    T_in_K = celsius_to_kelvin(T_in_fluid_C)
    T_out_K = celsius_to_kelvin(T_out_fluid_C)
    T_f_avg_K = (T_in_K + T_out_K) / 2
    T_surr_rad_K = celsius_to_kelvin(T_surr_rad_C)
    T_air_K = celsius_to_kelvin(T_air_C)
    T_dead_K = celsius_to_kelvin(T_dead_C)
    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A

    # Energy balance to find Ts
    def energy_balance_Ts(Ts_K):
        if Ts_K <= 0: return 1e9
        q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A)
        q_rad = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_Ts_surface)
        q_conv = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air)
        return q_source_total - (q_stack + q_rad + q_conv)

    Ts_K, info, ier, mesg = fsolve(energy_balance_Ts, celsius_to_kelvin(initial_Ts_guess_C), full_output=True)
    Ts_K = Ts_K[0]
    success = (ier == 1)
    if not success:
        return {'solver_success': False, 'solver_message': mesg.strip(), 'Ts_C': np.nan}

    # Compute heat fluxes at solved Ts
    q_stack = max(0, (Ts_K - T_f_avg_K) / R_stack_A)
    q_rad = calculate_radiative_heat_flux_loss(Ts_K, T_surr_rad_K, emissivity_Ts_surface)
    q_conv = calculate_convective_heat_flux_loss(Ts_K, T_air_K, h_conv_air)

    # Exergy streams
    ex_in_total = q_source_total * (1 - T_dead_K / Ts_K)
    ex_recovered = calculate_exergy_recovered_by_fluid(q_stack, T_in_K, T_out_K, T_dead_K)
    T_TEG_cold_K = Ts_K - q_stack * R_TEG_A
    T_HX_wall_K = T_TEG_cold_K - q_stack * R_HX_cond_A
    ex_dest_TEG = calculate_exergy_destruction_component(q_stack, Ts_K, T_TEG_cold_K, T_dead_K)
    ex_dest_cond = calculate_exergy_destruction_component(q_stack, T_TEG_cold_K, T_HX_wall_K, T_dead_K)
    ex_q_rad = q_rad * (1 - T_dead_K / Ts_K)
    ex_q_conv = q_conv * (1 - T_dead_K / Ts_K)
    ex_dest_system_total = ex_in_total - ex_recovered
    eta_ex_system = ex_recovered / ex_in_total if ex_in_total > 0 else 0.0

    return {
        'solver_success': True,
        'Ts_C': kelvin_to_celsius(Ts_K),
        'q_stack': q_stack,
        'q_rad_loss': q_rad,
        'q_conv_loss': q_conv,
        'ex_in_total': ex_in_total,
        'ex_recovered': ex_recovered,
        'ex_dest_TEG': ex_dest_TEG,
        'ex_dest_cond': ex_dest_cond,
        'ex_q_rad_loss': ex_q_rad,
        'ex_q_conv_loss': ex_q_conv,
        'ex_dest_system_total': ex_dest_system_total,
        'eta_ex_system': eta_ex_system
    }

# --- Example Run: Adjusting All Variables ---
if __name__ == '__main__':
    # Define adjustable parameters
    STEFAN_BOLTZMANN_CONST = 5.67e-8       # W/(m^2·K^4)
    R_TEG_A        = 0.015                # TEG conduction resistance (m^2·K/W)
    R_HX_cond_A    = 0.005                # HX conduction resistance (m^2·K/W)
    R_HX_conv_A    = 0.010                # HX convection resistance (m^2·K/W)

    # Fluid temperatures
    T_in_fluid_C   = 30.0                 # °C
    T_out_fluid_C  = 80.0                 # °C

    # Dead-state (ambient)
    T_dead_C       = 25.0                 # °C

    # Case 1 inputs
    T_source_C         = 250.0            # Source temperature (°C)
    emissivity_source  = 0.85             # Source emissivity
    T_ambient_rad_C    = 25.0             # Surrounding for radiation (°C)

    # Run Case 1
    results_case1 = analyze_case1_with_exergy(
        T_source_C,
        T_in_fluid_C,
        T_out_fluid_C,
        R_TEG_A,
        R_HX_cond_A,
        R_HX_conv_A,
        T_dead_C,
        emissivity_source,
        T_ambient_rad_C
    )

    # Case 2 inputs
    q_source_total     = results_case1['q_rad_loss_source'] + results_case1['q_stack']
    emissivity_Ts      = 0.8              # Surface emissivity
    T_surr_rad_C       = 25.0             # Surrounding for radiation (°C)
    h_conv_air         = 12.0             # Convective coefficient (W/m^2·K)
    T_air_C            = 25.0             # Ambient air (°C)
    initial_Ts_guess_C = 400.0            # Initial guess surface temp (°C)

    # Run Case 2
    results_case2 = analyze_case2_with_exergy(
        q_source_total,
        T_in_fluid_C,
        T_out_fluid_C,
        R_TEG_A,
        R_HX_cond_A,
        R_HX_conv_A,
        emissivity_Ts,
        T_surr_rad_C,
        h_conv_air,
        T_air_C,
        T_dead_C,
        initial_Ts_guess_C
    )
    print("Case 2 Results:", results_case2)

    # --- Exergy Distribution Pie Chart ---
    # Gather exergy components from Case 2
    Ex_in = results_case2['ex_in_total']
    Ex_recovered = results_case2['ex_recovered']
    Ex_dest_TEG = results_case2['ex_dest_TEG']
    Ex_dest_cond = results_case2['ex_dest_cond']
    Ex_rad_loss = results_case2['ex_q_rad_loss']
    Ex_conv_loss = results_case2['ex_q_conv_loss']
    Ex_dest_other = Ex_in - (Ex_recovered + Ex_dest_TEG + Ex_dest_cond + Ex_rad_loss + Ex_conv_loss)

    exergy_labels = [
        f"Recovered ({Ex_recovered:.1f} W)",
        f"TEG Dest. ({Ex_dest_TEG:.1f} W)",
        f"Cond. Dest. ({Ex_dest_cond:.1f} W)",
        f"Rad. Loss ({Ex_rad_loss:.1f} W)",
        f"Conv. Loss ({Ex_conv_loss:.1f} W)",
        f"Other ({Ex_dest_other:.1f} W)"
    ]
    exergy_vals = [Ex_recovered, Ex_dest_TEG, Ex_dest_cond, Ex_rad_loss, Ex_conv_loss, Ex_dest_other]

    plt.figure(figsize=(7, 7))
    plt.pie(exergy_vals, labels=exergy_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Exergy Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # --- Thermal Resistance Pie Chart ---
    # Individual resistances including radiation
    # Compute radiation resistance
    T_source_K = celsius_to_kelvin(T_source_C)
    T_amb_K = celsius_to_kelvin(T_ambient_rad_C)
    q_rad_source = calculate_radiative_heat_flux_loss(T_source_K, T_amb_K, emissivity_source)
    R_rad_A = (T_source_K - T_amb_K) / q_rad_source if q_rad_source > 0 else np.inf
    R_vals = [R_TEG_A, R_HX_cond_A, R_HX_conv_A, R_rad_A]
    R_labels = [
        f"TEG Cond ({R_TEG_A:.4f})",
        f"HX Cond ({R_HX_cond_A:.4f})",
        f"HX Conv ({R_HX_conv_A:.4f})",
        f"Radiation ({R_rad_A:.4f})"
    ]

    plt.figure(figsize=(7, 7))
    plt.pie(R_vals, labels=R_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Thermal Resistance Breakdown')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()