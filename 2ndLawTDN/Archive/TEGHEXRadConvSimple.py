import numpy as np
from scipy.optimize import fsolve

# Physical constants
sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m2K4)

# Boundary conditions and inputs (from user)
T_in_fluid_C = 40.0          # °C
T_out_fluid_C = 90.0         # °C
T_dead_C = 25.0              # °C (ambient/dead state)

# Case 1 source conditions
T_source_C = 300          # °C
emissivity_source_base = 0.8
extended_surf_rat_to_HX_surf = 1.6
emissivity_source = emissivity_source_base * extended_surf_rat_to_HX_surf
T_ambient_rad_C = 25.0       # °C for radiation sink

# Geometry and material properties
A = 1.0                      # m2 heat transfer area
# Layer thicknesses (m)
thickness_adhesive = 1e-3
thickness_cu = 1e-3
thickness_teg = 4e-3
thickness_hx_wall = 2e-3
# Thermal conductivities (W/mK)
k_adhesive = 0.5
k_cu = 400.0
k_teg = 1.5
k_hx_wall = 167.0

# Convective heat transfer coefficients (user must specify or tune)
h_conv_source = 10.0         # W/m2K (convection from source to interface)
h_conv_fluid = 500.0         # W/m2K (convection from HEx wall to water)

# Helper functions

def linearized_rad_coeff(epsilon, T1, T2):
    """Compute linearized radiative heat transfer coeff between surfaces at T1, T2 (K)."""
    return epsilon * sigma * (T1**2 + T2**2) * (T1 + T2)

# Conduction resistances
R_cond_adh = thickness_adhesive / (k_adhesive * A)
R_cond_cu = thickness_cu / (k_cu * A)
R_cond_teg = thickness_teg / (k_teg * A)
R_cond_wall = thickness_hx_wall / (k_hx_wall * A)
R_cond_total = R_cond_adh + R_cond_cu + R_cond_teg + R_cond_wall

# Convection to fluid resistance
R_conv_fluid = 1.0 / (h_conv_fluid * A)

# Average fluid temperature
T_fluid_avg_C = 0.5 * (T_in_fluid_C + T_out_fluid_C)

# Solve for interface temperature
T_source_K = T_source_C + 273.15
T_ambient_rad_K = T_ambient_rad_C + 273.15

def residual(Ti_C):
    """Residual of heat balance at interface (°C)."""
    Ti_K = Ti_C + 273.15
    # Convective heat from source
    q_conv = h_conv_source * A * (T_source_C - Ti_C)
    # Radiative heat from source
    h_rad = linearized_rad_coeff(emissivity_source, T_source_K, Ti_K)
    q_rad = h_rad * A * (T_source_C - Ti_C)
    # Heat conducted through layers and convected to fluid
    R_chain = R_cond_total + R_conv_fluid
    q_out = (Ti_C - T_fluid_avg_C) / R_chain
    # Balance
    return q_conv + q_rad - q_out

# Initial guess for interface temperature
Ti_guess = 150.0  # °C
Ti_solution_C, = fsolve(residual, Ti_guess)

# Compute final heat fluxes
Ti_solution_K = Ti_solution_C + 273.15
q_conv = h_conv_source * A * (T_source_C - Ti_solution_C)
q_rad = linearized_rad_coeff(emissivity_source, T_source_K, Ti_solution_K) * A * (T_source_C - Ti_solution_C)
q_out = (Ti_solution_C - T_fluid_avg_C) / (R_cond_total + R_conv_fluid)
q_total = q_conv + q_rad

# Display results
print(f"Interface Temperature: {Ti_solution_C:.2f} °C")
print(f"Heat flux from convection: {q_conv:.2f} W")
print(f"Heat flux from radiation: {q_rad:.2f} W")
print(f"Heat flux out to fluid: {q_out:.2f} W")
print(f"Total heat absorbed from source: {q_total:.2f} W")

# Placeholder for TEG power calculation
def teg_power(T_hot, T_cold, alpha, R_internal, I):
    """
    Calculate TEG power output given hot/cold temps (°C), Seebeck coeff alpha (V/K),\
    internal resistance (Ohm), and current I (A).
    """
    dT = T_hot - T_cold
    V = alpha * dT
    P_elec = V * I - I**2 * R_internal
    return P_elec

# Example usage of TEG power placeholder
# alpha = 200e-6  # V/K (example)
# R_internal = 0.5  # Ohm (example)
# I = V / (R_internal + external_load)
# P = teg_power(T_source_C, T_fluid_avg_C, alpha, R_internal, I)
# print(f"TEG Electrical Power: {P:.2f} W")
