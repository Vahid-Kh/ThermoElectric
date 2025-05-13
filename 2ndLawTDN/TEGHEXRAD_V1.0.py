import numpy as np  # For ln, and to handle potential division by zero gracefully
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# --- Constants and Helper Functions ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)
BOLTZMANN_CONST = 1.38e-23  # J/K


def celsius_to_kelvin(T_celsius):
    return T_celsius + 273.15

def kelvin_to_celsius(T_kelvin):
    return T_kelvin - 273.15

# --- Thermal Resistance Estimation Functions ---

def resistance_conduction(thickness_m, area_m2, k_W_mK):
    return thickness_m / (k_W_mK * area_m2)

def resistance_convection(h_W_m2K, area_m2):
    return 1 / (h_W_m2K * area_m2)

def resistance_radiation(epsilon, T1_K, T2_K, area_m2):
    return 1 / (epsilon * STEFAN_BOLTZMANN_CONST * area_m2 * (T1_K**2 + T2_K**2) * (T1_K + T2_K))

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
    T_in_fluid_C   = 40.0                 # °C
    T_out_fluid_C  = 90.0                 # °C


    # Dead-state (ambient)
    T_dead_C       = 25.0                 # °C

    # Case 1 inputs
    T_source_C         = 500.0            # Source temperature (°C)
    emissivity_source  = 0.8           # Source emissivity
    T_ambient_rad_C    = 25.0             # Surrounding for radiation (°C)

    # Geometry and materials
    area = 1  # m2
    thickness_adhesive = 1e-3  # m
    thickness_cu = 1e-3  # m
    thickness_teg = 2e-3  # m
    thickness_hx_wall = 1e-3  # m
    k_adhesive = 0.5  # W/mK
    k_cu = 400  # W/mK
    k_teg = 1.5  # W/mK
    k_hx_wall = 20  # W/mK

    # Environment
    Ts_C = 1000.0
    T_in_C = T_in_fluid_C
    T_out_C = T_out_fluid_C
    Ta_C = T_dead_C
    emissivity = emissivity_source
    h_conv = 25  # W/m2K

    # Temperatures
    Ts_K = celsius_to_kelvin(Ts_C)
    Tavg_fluid_K = celsius_to_kelvin((T_in_C + T_out_C) / 2)
    Ta_K = celsius_to_kelvin(Ta_C)

    # Resistances
    R1_surface_cond = resistance_conduction(thickness_adhesive, area, k_adhesive)
    R2_rad_air = resistance_radiation(emissivity, Ts_K, Ta_K, area)
    R3_adhesive_hot = resistance_conduction(thickness_adhesive, area, k_adhesive)
    R4_cu_col_hot = resistance_conduction(thickness_cu, area, k_cu)
    R5_TEG = resistance_conduction(thickness_teg, area, k_teg)
    R6_cu_col_cold = resistance_conduction(thickness_cu, area, k_cu)
    R7_adhesive_cold = resistance_conduction(thickness_adhesive, area, k_adhesive)
    R8_HX_cond = resistance_conduction(thickness_hx_wall, area, k_hx_wall)
    R9_HX_conv = resistance_convection(h_conv, area)

    import matplotlib.pyplot as plt

    # Define the resistances
    # Define the actual resistance values
    resistances = {
        "R1_surface_cond":R1_surface_cond,
        "R2_rad_air":R2_rad_air,
        "R3_adhesive_hot":R3_adhesive_hot,
        "R4_cu_col_hot":R4_cu_col_hot,
        "R5_TEG":R5_TEG,
        "R6_cu_col_cold":R6_cu_col_cold,
        "R7_adhesive_cold":R7_adhesive_cold,
        "R8_HX_cond":R8_HX_cond,
        "R9_HX_conv":R9_HX_conv

    }

    # Dummy q
    q_stack = 500.0

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
    # --- Print Key Outputs ---
    # Electricity produced by TEG (exergy recovered)
    print(f"Electricity produced (TEG exergy): {results_case2['ex_recovered']:.2f} W/m2")
    # Heat recovered by fluid in heat exchanger (stack heat flux)
    print(f"Heat recovered (fluid side): {results_case2['q_stack']:.2f} W/m^2")
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

    # --- Energy Recovered vs Lost Pie Chart ---
    # Compute recovered and lost exergy
    recovered_exergy = results_case2['ex_recovered']
    lost_exergy = results_case2['ex_dest_system_total']

    # --- Lost Exergy Breakdown Pie Chart ---
    # Breakdown of lost exergy into contributing destruction and losses
    ex_dest_TEG = results_case2['ex_dest_TEG']
    ex_dest_cond = results_case2['ex_dest_cond']
    ex_q_rad_loss = results_case2['ex_q_rad_loss']
    ex_q_conv_loss = results_case2['ex_q_conv_loss']
    ex_other_loss = lost_exergy - (ex_dest_TEG + ex_dest_cond + ex_q_rad_loss + ex_q_conv_loss)


    # Define energy stages
    stages_energy = ['Total Input', 'Radiative Loss', 'Convective Loss', 'Recovered Heat']
    values_energy = [
        q_source_total,
        results_case2['q_rad_loss'],
        results_case2['q_conv_loss'],
        results_case2['q_stack']
    ]
    # Compute deltas for waterfall
    deltas_energy = [values_energy[0]] + [-values_energy[1], -values_energy[2], values_energy[3]]
    cum = np.cumsum(deltas_energy)

    # --- Waterfall Plot: Exergy Breakdown ---
    # Define exergy stages
    stages_exergy = ['Total Exergy Input', 'Radiative Exergy Loss', 'Convective Exergy Loss',
                     'TEG Destruction', 'HX Destruction', 'Recovered Exergy']
    values_exergy = [
        results_case2['ex_in_total'],
        results_case2['ex_q_rad_loss'],
        results_case2['ex_q_conv_loss'],
        results_case2['ex_dest_TEG'],
        results_case2['ex_dest_cond'],
        results_case2['ex_recovered']
    ]
    # Compute deltas for waterfall (losses negative)
    deltas_exergy = [values_exergy[0]] + [-values_exergy[1], -values_exergy[2],
                                          -values_exergy[3], -values_exergy[4], values_exergy[5]]
    cum_ex = np.cumsum(deltas_exergy)

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

    # Temperature calculation along path
    T2_K = Ts_K - q_stack * R5_TEG
    T3_K = T2_K - q_stack * R3_adhesive_hot
    T4_K = T3_K - q_stack * R4_cu_col_hot
    T5_K = Ts_K - q_stack * (R5_TEG + R3_adhesive_hot + R4_cu_col_hot)
    T6_K = Tavg_fluid_K

    temps_C = [Ts_C,
               kelvin_to_celsius(T4_K),
               kelvin_to_celsius(T3_K),
               kelvin_to_celsius(T2_K),
               kelvin_to_celsius(T6_K),
               Ta_C]

    resistances = [0,
                   R1_surface_cond,
                   R2_rad_air,
                   R3_adhesive_hot,
                   R4_cu_col_hot,
                   R5_TEG, R6_cu_col_cold, R7_adhesive_cold, R8_HX_cond, R9_HX_conv]


    # Calculate cumulative resistances
    cumulative_resistances = [sum(resistances[:i + 1]) for i in range(len(resistances))]

    plt.figure(figsize=(7, 7))
    plt.pie(exergy_vals, labels=exergy_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Exergy Distribution')
    plt.axis('equal')
    plt.tight_layout()


    plt.figure(figsize=(7, 7))
    plt.pie(R_vals, labels=R_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Thermal Resistance Breakdown')
    plt.axis('equal')
    plt.tight_layout()

    labels_el = ['Recovered', 'Lost']
    values_el = [recovered_exergy, lost_exergy]
    colors_el = ['green', 'red']

    plt.figure(figsize=(6, 6))
    plt.pie(values_el, labels=labels_el, autopct='%1.1f%%', colors=colors_el, startangle=90)
    plt.title('Energy (Exergy) Recovered vs Lost')
    plt.axis('equal')
    plt.tight_layout()


    loss_labels = [
        f'TEG Destr. ({ex_dest_TEG:.1f} W)',
        f'Cond. Destr. ({ex_dest_cond:.1f} W)',
        f'Rad. Loss ({ex_q_rad_loss:.1f} W)',
        f'Conv. Loss ({ex_q_conv_loss:.1f} W)',
        f'Other ({ex_other_loss:.1f} W)'
    ]
    loss_vals = [ex_dest_TEG, ex_dest_cond, ex_q_rad_loss, ex_q_conv_loss, ex_other_loss]

    plt.figure(figsize=(7, 7))
    plt.pie(loss_vals, labels=loss_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Lost Exergy Breakdown')
    plt.axis('equal')
    plt.tight_layout()


    plt.figure(figsize=(8, 5))
    plt.bar(range(len(deltas_energy)), deltas_energy, bottom=np.append([0], cum[:-1]),
            color=['blue', 'red', 'red', 'green'], edgecolor='black')
    plt.xticks(range(len(stages_energy)), stages_energy)
    plt.ylabel('Energy Flux (W/m²)')
    plt.title('Waterfall: Energy Breakdown')
    plt.tight_layout()


    plt.figure(figsize=(8, 5))
    plt.bar(range(len(deltas_exergy)), deltas_exergy, bottom=np.append([0], cum_ex[:-1]),
            color=['blue', 'red', 'red', 'red', 'red', 'green'], edgecolor='black')
    plt.xticks(range(len(stages_exergy)), stages_exergy, rotation=45, ha='right')
    plt.ylabel('Exergy Flow (W/m²)')
    plt.title('Waterfall: Exergy Breakdown')
    plt.tight_layout()


    # -----------------------------
    # Helper Functions
    # -----------------------------
    def celsius_to_kelvin(T_C):
        """Convert temperature in Celsius to Kelvin."""
        return T_C + 273.15


    def kelvin_to_celsius(T_K):
        """Convert temperature in Kelvin to Celsius."""
        return T_K - 273.15


    def resistance_conduction(thickness, area, k):
        """
        Calculate the conduction thermal resistance.

        R_cond = thickness / (k * area)

        Parameters:
            thickness : float
                Material thickness in meters (m)
            area : float
                Cross-sectional area in square meters (m²)
            k : float
                Thermal conductivity in W/(m·K)

        Returns:
            float: Thermal resistance in m²K/W
        """
        return thickness / (k * area)


    def resistance_radiation(emissivity, Ts_K, Ta_K, area, sigma=5.67e-8):
        """
        Calculate the radiation thermal resistance using a linearized approach.

        First, an effective radiative heat transfer coefficient is computed as:

            h_rad = emissivity * sigma * T_avg³,   where T_avg = (Ts_K + Ta_K) / 2.

        Then the radiation resistance is:

            R_rad = 1 / (h_rad * area)

        Parameters:
            emissivity : float
                Emissivity of the surface (0 to 1)
            Ts_K : float
                Surface temperature in Kelvin
            Ta_K : float
                Ambient (or surrounding) temperature in Kelvin
            area : float
                Radiative area in m²
            sigma : float, optional
                Stefan–Boltzmann constant (default 5.67e-8 W/m²·K⁴)

        Returns:
            float: Radiation resistance in m²K/W
        """
        T_avg = (Ts_K + Ta_K) / 2
        h_rad = emissivity * sigma * (T_avg ** 3)
        return 1 / (h_rad * area)


    def resistance_convection(h_conv, area):
        """
        Calculate the convection resistance.

        R_conv = 1 / (h_conv * area)

        Parameters:
            h_conv : float
                Convective heat transfer coefficient in W/m²K
            area : float
                Surface area in m²

        Returns:
            float: Convection resistance in m²K/W
        """
        return 1 / (h_conv * area)



    # -----------------------------
    # Convert Temperatures to Kelvin
    # -----------------------------
    # Average fluid temperature if needed (not used in these specific resistance calculations)


    # -----------------------------
    # Calculate Resistances
    # -----------------------------
    # 1. Conduction from the source surface (adhesive) to the external environment
    R1_surface_cond = resistance_conduction(thickness_adhesive, area, k_adhesive)
    # 2. Radiation resistance from the surface to ambient air
    R2_rad_air = resistance_radiation(emissivity_source, Ts_K, Ta_K, area)
    # 3. Conduction resistance through the hot adhesive layer
    R3_adhesive_hot = resistance_conduction(thickness_adhesive, area, k_adhesive)
    # 4. Conduction resistance through the hot copper collector
    R4_cu_col_hot = resistance_conduction(thickness_cu, area, k_cu)
    # 5. Conduction resistance through the TEG element
    R5_TEG = resistance_conduction(thickness_teg, area, k_teg)
    # 6. Conduction through the cold copper collector
    R6_cu_col_cold = resistance_conduction(thickness_cu, area, k_cu)
    # 7. Conduction resistance through the cold adhesive layer
    R7_adhesive_cold = resistance_conduction(thickness_adhesive, area, k_adhesive)
    # 8. Conduction resistance through the heat exchanger wall
    R8_HX_cond = resistance_conduction(thickness_hx_wall, area, k_hx_wall)
    # 9. Convection resistance from the heat exchanger surface to the fluid
    R9_HX_conv = resistance_convection(h_conv, area)

    # -----------------------------
    # Store and Plot the Resistances
    # -----------------------------
    resistances = {
        "R1_surface_cond": R1_surface_cond,
        "R2_rad_air": R2_rad_air,
        "R3_adhesive_hot": R3_adhesive_hot,
        "R4_cu_col_hot": R4_cu_col_hot,
        "R5_TEG": R5_TEG,
        "R6_cu_col_cold": R6_cu_col_cold,
        "R7_adhesive_cold": R7_adhesive_cold,
        "R8_HX_cond": R8_HX_cond,
        "R9_HX_conv": R9_HX_conv
    }

    # -----------------------------
    # Optional: Print the resistance values
    # -----------------------------
    print("Calculated Thermal Resistances:")
    for name, value in resistances.items():
        print(f"{name:20s}: {value:.6f} m²K/W")


    # Plot the resistances using a bar chart.
    plt.figure(figsize=(10, 6))
    plt.bar(resistances.keys(), resistances.values(), color="skyblue")
    plt.xlabel('Resistance Types')
    plt.ylabel('Thermal Resistance (m²·K/W)')
    plt.title('Thermal Resistances for Each Component')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Create a list of resistances

    labels = ['Start', 'Surface Cond', 'Rad Air', 'Adhesive Hot', 'Cu Col Hot', 'TEG', 'Cu Col Cold', 'Adhesive Cold',
              'HX Cond', 'HX Conv']


    plt.show()



