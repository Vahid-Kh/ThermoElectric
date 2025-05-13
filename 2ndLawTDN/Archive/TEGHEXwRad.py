import numpy as np
from scipy.optimize import fsolve


def celsius_to_kelvin(T_celsius):
    """Converts temperature from Celsius to Kelvin."""
    return T_celsius + 273.15

STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2 K^4)


def calculate_average_fluid_temperature(T_in_fluid_C, T_out_fluid_C):
    """Calculates the average fluid temperature."""
    return (T_in_fluid_C + T_out_fluid_C) / 2

def calculate_heat_flux_through_stack_case1(
    T0_C,
    T_in_fluid_C,
    T_out_fluid_C,
    R_TEG_A,
    R_HX_cond_A,
    R_HX_conv_A
):
    """
    Calculates the heat flux through the TEG-MCHX stack for Case 1.

    Args:
        T0_C (float): Temperature of the hot surface (TEG hot side) in Celsius.
        T_in_fluid_C (float): Inlet fluid temperature in Celsius.
        T_out_fluid_C (float): Outlet fluid temperature in Celsius.
        R_TEG_A (float): Area-specific thermal resistance of TEG (m^2*K/W).
        R_HX_cond_A (float): Area-specific conductive thermal resistance of MCHX (m^2*K/W).
        R_HX_conv_A (float): Area-specific convective thermal resistance from MCHX to fluid (m^2*K/W).

    Returns:
        float: Heat flux through the stack (W/m^2).
    """
    T0_K = celsius_to_kelvin(T0_C)
    T_f_avg_C = calculate_average_fluid_temperature(T_in_fluid_C, T_out_fluid_C)
    T_f_avg_K = celsius_to_kelvin(T_f_avg_C)

    R_stack_A = R_TEG_A + R_HX_cond_A + R_HX_conv_A
    if R_stack_A == 0:
        return float('inf') # Avoid division by zero
    q_stack = (T0_K - T_f_avg_K) / R_stack_A
    return q_stack

def calculate_radiative_heat_flux_loss(
    T_surface_C,
    T_ambient_C,
    emissivity
):
    """
    Calculates the radiative heat flux loss from a surface.

    Args:
        T_surface_C (float): Temperature of the emitting surface in Celsius.
        T_ambient_C (float): Temperature of the ambient/surroundings in Celsius.
        emissivity (float): Emissivity of the surface (0 to 1).

    Returns:
        float: Radiative heat flux loss (W/m^2).
    """
    T_surface_K = celsius_to_kelvin(T_surface_C)
    T_ambient_K = celsius_to_kelvin(T_ambient_C)

    q_rad_loss = emissivity * STEFAN_BOLTZMANN_CONST * (T_surface_K**4 - T_ambient_K**4)
    return q_rad_loss

# Example Usage for Case 1:
# These are placeholder values for resistances.
R_TEG_A_example = 0.01  # m^2*K/W
R_HX_cond_A_example = 0.002 # m^2*K/W
R_HX_conv_A_example = 0.005 # m^2*K/W

T0_C_case1 = 900.0
T_in_fluid_C_case1 = 40.0
T_out_fluid_C_case1 = 90.0

q_stack_case1 = calculate_heat_flux_through_stack_case1(
    T0_C_case1,
    T_in_fluid_C_case1,
    T_out_fluid_C_case1,
    R_TEG_A_example,
    R_HX_cond_A_example,
    R_HX_conv_A_example
)
print(f"Case 1 - Heat flux through stack: {q_stack_case1:.2f} W/m^2")

# Optional: Calculate radiative loss from the surface at T0
emissivity_example = 0.8
T_ambient_rad_C_example = 25.0
q_rad_loss_case1 = calculate_radiative_heat_flux_loss(
    T0_C_case1,
    T_ambient_rad_C_example,
    emissivity_example
)
print(f"Case 1 - Radiative heat flux loss from T0 surface: {q_rad_loss_case1:.2f} W/m^2")


from scipy.optimize import fsolve

def calculate_convective_heat_flux_loss(
    T_surface_C,
    T_air_C,
    h_conv_air
):
    """
    Calculates the convective heat flux loss from a surface.

    Args:
        T_surface_C (float): Temperature of the surface in Celsius.
        T_air_C (float): Temperature of the ambient air in Celsius.
        h_conv_air (float): Convective heat transfer coefficient (W/(m^2*K)).

    Returns:
        float: Convective heat flux loss (W/m^2).
    """
    T_surface_K = celsius_to_kelvin(T_surface_C)
    T_air_K = celsius_to_kelvin(T_air_C)
    q_conv_loss = h_conv_air * (T_surface_K - T_air_K)
    return q_conv_loss

def solve_surface_temperature_case2(
    q_source,
    T_in_fluid_C,
    T_out_fluid_C,
    R_stack_A, # Combined R_TEG_A + R_HX_cond_A + R_HX_conv_A
    emissivity,
    T_surr_rad_C,
    h_conv_air,
    T_air_C,
    initial_Ts_guess_C=200.0 # Initial guess for Ts in Celsius
):
    """
    Solves for the surface temperature (Ts) in Case 2 and calculates heat fluxes.

    Args:
        q_source (float): External heat flux supplied to the surface (W/m^2).
        T_in_fluid_C (float): Inlet fluid temperature in Celsius.
        T_out_fluid_C (float): Outlet fluid temperature in Celsius.
        R_stack_A (float): Total area-specific thermal resistance of TEG-MCHX stack (m^2*K/W).
        emissivity (float): Emissivity of the surface for radiative loss.
        T_surr_rad_C (float): Surrounding temperature for radiation in Celsius.
        h_conv_air (float): Convective heat transfer coefficient to air (W/(m^2*K)).
        T_air_C (float): Ambient air temperature for convection in Celsius.
        initial_Ts_guess_C (float): Initial guess for surface temperature in Celsius.

    Returns:
        tuple: (Ts_C, q_stack, q_rad_loss, q_conv_loss) or None if solver fails
               Ts_C is in Celsius. Fluxes are in W/m^2.
    """
    T_f_avg_C = calculate_average_fluid_temperature(T_in_fluid_C, T_out_fluid_C)
    T_f_avg_K = celsius_to_kelvin(T_f_avg_C)
    T_surr_rad_K = celsius_to_kelvin(T_surr_rad_C)
    T_air_K = celsius_to_kelvin(T_air_C)

    # Define the equation to solve: q_source - q_stack - q_rad_loss - q_conv_loss = 0
    def energy_balance_equation(Ts_K):
        if R_stack_A <= 0: # Prevent division by zero or invalid resistance
            q_stack_val = float('inf') if Ts_K > T_f_avg_K else float('-inf')
        else:
            q_stack_val = (Ts_K - T_f_avg_K) / R_stack_A

        q_rad_loss_val = emissivity * STEFAN_BOLTZMANN_CONST * (Ts_K**4 - T_surr_rad_K**4)
        # Ensure radiation loss is positive if Ts_K < T_surr_rad_K (net gain)
        # The formula naturally handles this, but physically loss implies Ts_K > T_surr_rad_K

        q_conv_loss_val = h_conv_air * (Ts_K - T_air_K)

        return q_source - q_stack_val - q_rad_loss_val - q_conv_loss_val

    # Initial guess for Ts in Kelvin
    initial_Ts_guess_K = celsius_to_kelvin(initial_Ts_guess_C)

    # Solve for Ts_K
    Ts_K_solution, info, ier, mesg = fsolve(energy_balance_equation, initial_Ts_guess_K, full_output=True)

    if ier == 1: # Solution found
        Ts_K_final = Ts_K_solution[0]
        Ts_C_final = Ts_K_final - 273.15

        # Calculate final fluxes with the solved Ts
        q_stack_final = (Ts_K_final - T_f_avg_K) / R_stack_A if R_stack_A > 0 else float('nan')
        q_rad_loss_final = emissivity * STEFAN_BOLTZMANN_CONST * (Ts_K_final**4 - T_surr_rad_K**4)
        q_conv_loss_final = h_conv_air * (Ts_K_final - T_air_K)
        return Ts_C_final, q_stack_final, q_rad_loss_final, q_conv_loss_final
    else:
        print(f"Solver failed to find a solution for Ts: {mesg}")
        return None, None, None, None


# Example Usage for Case 2:
# These are placeholder values.
q_source_example = 20000.0  # W/m^2 (e.g., 20 kW/m^2)
R_stack_A_total_example = R_TEG_A_example + R_HX_cond_A_example + R_HX_conv_A_example # from Case 1 example
emissivity_case2 = 0.8
T_surr_rad_C_case2 = 25.0
h_conv_air_example = 10.0  # W/(m^2*K)
T_air_C_case2 = 25.0
T_in_fluid_C_case2 = 40.0
T_out_fluid_C_case2 = 90.0


results_case2 = solve_surface_temperature_case2(
    q_source_example,
    T_in_fluid_C_case2,
    T_out_fluid_C_case2,
    R_stack_A_total_example,
    emissivity_case2,
    T_surr_rad_C_case2,
    h_conv_air_example,
    T_air_C_case2,
    initial_Ts_guess_C=900.0 # Provide a reasonable guess
)

if results_case2[0] is not None:
    Ts_C_solved, q_stack_case2, q_rad_loss_case2, q_conv_loss_case2 = results_case2
    print(f"\nCase 2 Results:")
    print(f"  Solved Surface Temperature (Ts): {Ts_C_solved:.2f} °C")
    print(f"  Heat flux through stack (q_stack): {q_stack_case2:.2f} W/m^2")
    print(f"  Radiative heat flux loss (q_rad_loss): {q_rad_loss_case2:.2f} W/m^2")
    print(f"  Convective heat flux loss (q_conv_loss): {q_conv_loss_case2:.2f} W/m^2")
    print(f"  Sum of losses + stack flux: {q_stack_case2 + q_rad_loss_case2 + q_conv_loss_case2:.2f} W/m^2 (should be close to q_source)")

