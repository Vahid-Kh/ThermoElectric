import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from functions import *
import matplotlib.pyplot as plt


# --- Constants and Boundary Conditions ---
# Temperatures
T_in_fluid_C = 40.0  # °C
T_out_fluid_C = 65.0  # °C
T_dead_C = 25.0  # °C
T_source_C = 300.0  # Source temperature (°C)
T_ambient_rad_C = 25.0  # Surrounding for radiation from source (°C)

# Emissivity
emissivity_source_base = 0.8  # Source emissivity base
extended_surf_rat_to_HX_surf = 1.6  # Beam shaping extended surface ratio - this seems to modify the effective emissivity
emissivity_source = emissivity_source_base * extended_surf_rat_to_HX_surf
# If emissivity > 1, cap it at 0.99 for physical sense in standard radiation equations,
# or interpret 'extended_surf_rat_to_HX_surf' as an area enhancement factor for radiation.
# Assuming it's an effective emissivity for the exchange, but capping at a high value if >1.
if emissivity_source > 0.99:  # Capping at a high value, typically emissivity is <=1
    print(
        f"Warning: Calculated emissivity_source ({emissivity_source}) > 1. Capping at 0.99 for calculations. This might reflect an enhanced effective area instead of pure emissivity.")


# Geometry and materials
area = 1.0  # m^2 (Assumed reference area for resistances)
thickness_adhesive = 1e-4  # m
thickness_cu = 1e-3  # m  It is beacuse copper +solder used
thickness_teg = 4e-3  # m
thickness_hx_wall = 10e-3  # m

k_adhesive = 0.5  # W/mK
k_cu = 400  # W/mK
k_teg = 1.5  # W/mK
k_hx_wall = 167.0  # W/mK - Aluminum 6061-T6

# Assumed parameters (IMPORTANT - these significantly affect results)
h_fluid = 4000.0  # W/m^2K (Assumed convective heat transfer coefficient for water in HEX)




# Stefan-Boltzmann constant
sigma = 5.67e-8  # W/m^2K^4


# --- Helper Functions ---
def celsius_to_kelvin(T_C):
    return T_C + 273.15


# Convert temperatures to Kelvin
T_in_fluid_K = celsius_to_kelvin(T_in_fluid_C)
T_out_fluid_K = celsius_to_kelvin(T_out_fluid_C)
T_dead_K = celsius_to_kelvin(T_dead_C)
T_source_K = celsius_to_kelvin(T_source_C)
T_ambient_rad_K = celsius_to_kelvin(T_ambient_rad_C)

T_fluid_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2.0
T_fluid_avg_C = (T_in_fluid_C + T_out_fluid_C) / 2.0

# --- Thermal Resistance Calculations ---
# Note: The PDF Figure 4 shows "Source -> R_rad_source -> R_conv_source -> TEG Hot Side ..."
# The user input seems to imply radiation from T_source_C to the first layer of the TEG assembly.
# We will model a stack: Adhesive1, Cu1, TEG, Cu2, Adhesive2, HX Wall.
# Radiation occurs from T_source_K to T_s1 (surface of Adhesive1).

R_adhesive1 = thickness_adhesive / (k_adhesive * area)
R_cu1 = thickness_cu / (k_cu * area)
R_teg_thermal = thickness_teg / (k_teg * area)  # Thermal resistance of TEG material
R_cu2 = thickness_cu / (k_cu * area)
R_adhesive2 = thickness_adhesive / (k_adhesive * area)
R_hx_wall = thickness_hx_wall / (k_hx_wall * area)
R_conv_fluid = 1 / (h_fluid * area)

# Sum of conductive and convective resistances from the fluid-side of TEG hot surface to the fluid
R_stack_conductive_conv = R_adhesive1 + R_cu1 + R_teg_thermal + R_cu2 + R_adhesive2 + R_hx_wall + R_conv_fluid
R_stack_conductive_only = R_adhesive1 + R_cu1 + R_teg_thermal + R_cu2 + R_adhesive2 + R_hx_wall


# --- Iterative Solver for Heat Flux and Temperatures ---
# We need to find T_s1 (temperature of the surface receiving radiation)
# Q_rad = emissivity_source * sigma * area * (T_source_K**4 - T_s1**4)
# Q_cond_conv = (T_s1 - T_fluid_avg_K) / R_stack_conductive_conv
# In steady state, Q_rad = Q_cond_conv = Q

def equations_to_solve(p, T_source_K_val, T_fluid_avg_K_val, R_stack_val, area_val, emissivity_val, sigma_val):
    T_s1_K, Q_val = p  # T_s1_K is the temp of the surface receiving radiation

    # Ensure T_s1_K does not exceed T_source_K to avoid physical issues in radiation math or negative Q from source.
    # And T_s1_K must be greater than T_fluid_avg_K_val
    if T_s1_K >= T_source_K_val or T_s1_K <= T_fluid_avg_K_val:
        T_s1_K = (T_source_K_val + T_fluid_avg_K_val) / 2  # Re-guess if out of bounds

    eq1 = Q_val - emissivity_val * sigma_val * area_val * (T_source_K_val ** 4 - T_s1_K ** 4)
    eq2 = Q_val - (T_s1_K - T_fluid_avg_K_val) / R_stack_val
    return (eq1, eq2)

def eta_carnot(T_C,T_H):
    Eta_Carnot = (1 - ((273.15 + T_C) / (273.15 + T_H)))
    return  Eta_Carnot


# Initial guess for T_s1_K (between source and fluid temp) and Q
initial_guess_T_s1 = (T_source_K + T_fluid_avg_K) / 2
initial_guess_Q = emissivity_source * sigma * area * (T_source_K ** 4 - initial_guess_T_s1 ** 4)

T_s1_K_sol, Q_sol = fsolve(equations_to_solve,
                           (initial_guess_T_s1, initial_guess_Q),
                           args=(T_source_K, T_fluid_avg_K, R_stack_conductive_conv, area, emissivity_source, sigma))
eta_teg_to_carnot_ratio = 0.2
# Check if solution is physically reasonable
if Q_sol <= 0 or T_s1_K_sol >= T_source_K or T_s1_K_sol <= T_fluid_avg_K:
    print("Warning: Solver resulted in a non-physical solution. Results might be inaccurate.")
    print(f"  Solved Q: {Q_sol:.2f} W, Solved T_s1: {T_s1_K_sol - 273.15:.2f} °C")
    # Fallback or error handling might be needed here.
    # For now, we'll proceed with potentially problematic values if solver converged to them.

Q_stack_heat_flux = Q_sol  # This is the heat flowing through the entire stack

# Calculate intermediate temperatures based on Q_stack_heat_flux
T_s1_C_sol = T_s1_K_sol - 273.15

# Temperatures at interfaces (T_s1 is surface of adhesive1, i.e. TEG hot side assembly)
T_after_adhesive1_K = T_s1_K_sol - Q_stack_heat_flux * R_adhesive1
T_teg_hot_K = T_after_adhesive1_K - Q_stack_heat_flux * R_cu1
T_teg_cold_K = T_teg_hot_K - Q_stack_heat_flux * R_teg_thermal
T_before_adhesive2_K = T_teg_cold_K - Q_stack_heat_flux * R_cu2
T_hx_wall_hot_K = T_before_adhesive2_K - Q_stack_heat_flux * R_adhesive2
T_hx_wall_cold_K = T_hx_wall_hot_K - Q_stack_heat_flux * R_hx_wall
# T_hx_wall_cold_K should also be Q_stack_heat_flux * R_conv_fluid + T_fluid_avg_K (for verification)
T_check_hx_cold = Q_stack_heat_flux * R_conv_fluid + T_fluid_avg_K
# print(f"Calculated HEX wall cold side temp: {T_hx_wall_cold_K-273.15:.2f} C, Check via fluid: {T_check_hx_cold-273.15:.2f} C")


# --- Energy Recovery Calculations ---
# Heat recovered by fluid in heat exchanger
Q_recovered_fluid = Q_stack_heat_flux  # In steady state, this is the heat passing to the fluid

# Electricity produced by TEG
# Q_in_TEG_hot_side is the heat flowing *into* the hot side of the TEG module itself.
# This is Q_stack_heat_flux, as it's conducted through the TEG.
Q_through_teg = Q_stack_heat_flux
eta_teg = eta_carnot(T_teg_cold_K - 273.15,T_teg_hot_K - 273.15)*eta_teg_to_carnot_ratio  # Assumed TEG efficiency (P_elec / Q_in_TEG_hot_side)

P_electricity_teg = eta_teg * Q_through_teg

# Radiation heat transfer coefficient (for the resistance value)
# This h_rad corresponds to the exchange between T_source_K and T_s1_K_sol
if (T_source_K - T_s1_K_sol) != 0:  # Avoid division by zero if temps are somehow equal
    h_rad_eff = Q_stack_heat_flux / (area * (T_source_K - T_s1_K_sol))
    R_rad_eff = 1 / (h_rad_eff * area)
else:  # Should not happen if Q_sol > 0
    h_rad_eff = float('inf')
    R_rad_eff = 0

# --- Exergy Calculations ---
# Exergy input: Exergy of the heat Q_stack_heat_flux at the source temperature T_source_K
# This is the exergy of the heat successfully transferred to the system.
X_in_q_rad = Q_stack_heat_flux * (1 - T_dead_K / T_source_K) if T_source_K > 0 else 0

# Exergy recovered
X_recovered_electricity = P_electricity_teg  # Electricity is pure exergy

# Exergy of heat recovered by fluid (using average fluid temperature)
# To be more precise, one might use Logarithmic Mean Temperature Difference (LMTD) if calculating exergy transfer to fluid stream
# But for simplicity with T_avg:
X_recovered_fluid_heat = Q_recovered_fluid * (1 - T_dead_K / T_fluid_avg_K) if T_fluid_avg_K > 0 else 0

# Total exergy recovered
X_recovered_total = X_recovered_electricity + X_recovered_fluid_heat

# Exergy destroyed/lost
X_destroyed_total = X_in_q_rad - X_recovered_total
if X_destroyed_total < 0:  # Should not happen if model is consistent
    print(f"Warning: Negative total exergy destruction ({X_destroyed_total:.2f} W). Check model or assumptions.")
    X_destroyed_total = max(0, X_destroyed_total)  # Cap at 0

# Breakdown of exergy destruction
# 1. Exergy destruction in radiation from source to T_s1
X_dest_rad = Q_stack_heat_flux * T_dead_K * (
            1 / T_s1_K_sol - 1 / T_source_K) if T_s1_K_sol > 0 and T_source_K > 0 else 0
X_dest_rad = max(0, X_dest_rad)  # Destruction cannot be negative

# 2. Exergy destruction in conductive layers + TEG thermal path + convection to fluid
# T_s1_K_sol -> T_after_adhesive1_K (Adhesive 1)
X_dest_adh1 = Q_stack_heat_flux * T_dead_K * (
            1 / T_after_adhesive1_K - 1 / T_s1_K_sol) if T_after_adhesive1_K > 0 and T_s1_K_sol > 0 else 0
X_dest_adh1 = max(0, X_dest_adh1)

# T_after_adhesive1_K -> T_teg_hot_K (Cu1)
X_dest_cu1 = Q_stack_heat_flux * T_dead_K * (
            1 / T_teg_hot_K - 1 / T_after_adhesive1_K) if T_teg_hot_K > 0 and T_after_adhesive1_K > 0 else 0
X_dest_cu1 = max(0, X_dest_cu1)

# T_teg_hot_K -> T_teg_cold_K (TEG thermal conduction - part of exergy destruction here)
# The exergy destruction within TEG should also account for electricity generation.
# Simplified: (Q_in_TEG_hot - P_elec) is heat rejected at T_teg_cold_K.
# Entropy generated in TEG: (Q_through_teg - P_electricity_teg)/T_teg_cold_K - Q_through_teg/T_teg_hot_K
S_gen_teg = (
                        Q_through_teg - P_electricity_teg) / T_teg_cold_K - Q_through_teg / T_teg_hot_K if T_teg_cold_K > 0 and T_teg_hot_K > 0 else 0
S_gen_teg = max(0, S_gen_teg)
X_dest_teg = T_dead_K * S_gen_teg

# T_teg_cold_K -> T_before_adhesive2_K (Cu2)
X_dest_cu2 = Q_stack_heat_flux * T_dead_K * (
            1 / T_before_adhesive2_K - 1 / T_teg_cold_K) if T_before_adhesive2_K > 0 and T_teg_cold_K > 0 else 0
X_dest_cu2 = max(0, X_dest_cu2)

# T_before_adhesive2_K -> T_hx_wall_hot_K (Adhesive 2)
X_dest_adh2 = Q_stack_heat_flux * T_dead_K * (
            1 / T_hx_wall_hot_K - 1 / T_before_adhesive2_K) if T_hx_wall_hot_K > 0 and T_before_adhesive2_K > 0 else 0
X_dest_adh2 = max(0, X_dest_adh2)

# T_hx_wall_hot_K -> T_hx_wall_cold_K (HX Wall)
X_dest_hx_wall = Q_stack_heat_flux * T_dead_K * (
            1 / T_hx_wall_cold_K - 1 / T_hx_wall_hot_K) if T_hx_wall_cold_K > 0 and T_hx_wall_hot_K > 0 else 0
X_dest_hx_wall = max(0, X_dest_hx_wall)

# T_hx_wall_cold_K -> T_fluid_avg_K (Convection to fluid)
X_dest_conv_fluid = Q_stack_heat_flux * T_dead_K * (
            1 / T_fluid_avg_K - 1 / T_hx_wall_cold_K) if T_fluid_avg_K > 0 and T_hx_wall_cold_K > 0 else 0
X_dest_conv_fluid = max(0, X_dest_conv_fluid)

# Sum of component exergy destructions
X_dest_sum_components = X_dest_rad + X_dest_adh1 + X_dest_cu1 + X_dest_teg + X_dest_cu2 + X_dest_adh2 + X_dest_hx_wall + X_dest_conv_fluid

# Check: X_destroyed_total should be close to X_dest_sum_components
# Discrepancies can arise from using T_fluid_avg_K for X_recovered_fluid_heat vs. detailed temp drops.
# print(f"Total Exergy Destroyed (Overall Balance): {X_destroyed_total:.2f} W")
# print(f"Sum of Component Exergy Destructions: {X_dest_sum_components:.2f} W")
# For plotting, it's better to use sum of components, and if there's a residual, label it 'unaccounted'
# Or normalize components to match X_destroyed_total from the overall balance.
# Let's use the overall X_destroyed_total and then break it down proportionally if sum is different.
if X_dest_sum_components > 1e-6:  # Avoid division by zero if all destructions are zero
    scale_factor = X_destroyed_total / X_dest_sum_components
    X_dest_rad *= scale_factor
    X_dest_adh1 *= scale_factor
    X_dest_cu1 *= scale_factor
    X_dest_teg *= scale_factor
    X_dest_cu2 *= scale_factor
    X_dest_adh2 *= scale_factor
    X_dest_hx_wall *= scale_factor
    X_dest_conv_fluid *= scale_factor
else:  # if sum_components is zero, and X_destroyed_total is also zero, then all fine. If not, something is wrong.
    if X_destroyed_total > 1e-6:
        print(
            "Warning: Sum of component exergy destructions is zero, but overall exergy destruction is non-zero. Breakdown will be incorrect.")

# --- Plotting ---
# plt.style.use('seaborn-v0_8-whitegrid')

# 1. Energy Recovered vs Lost Pie Chart
# Total energy input to the system is Q_stack_heat_flux (energy transferred from source)
total_energy_input_for_pie = Q_stack_heat_flux
energy_lost_pie = total_energy_input_for_pie - P_electricity_teg - Q_recovered_fluid
if energy_lost_pie < 0:  # This implies P_elec + Q_recovered > Q_input, which is wrong.
    # Q_recovered_fluid is Q_stack_heat_flux. So P_elec must be from Q_stack_heat_flux.
    # This must mean Q_recovered_fluid should be what's left *after* P_elec.
    # No, Q_recovered_fluid is the total heat that reaches the fluid.
    # TEG converts a *portion* of Q_through_teg. The rest of Q_through_teg continues.
    # So, the heat reaching fluid = Q_stack_heat_flux - P_electricity_teg (if TEG is before HEX fluid directly)
    # OR, Q_stack_heat_flux is total, P_elec is extracted from it, Q_recovered_fluid is also that total heat minus electrical.
    # The diagram implies TEG is in series. So heat flows through TEG, some converted to P_elec, rest continues.
    # Thus, heat ultimately transferred to fluid for temperature rise is Q_stack_heat_flux - P_electricity_teg
    Q_net_to_fluid_for_temp_rise = Q_stack_heat_flux - P_electricity_teg
    energy_lost_pie = 0  # Assuming all heat not converted to electricity goes to fluid.
    # This needs clarification from the problem if "losses" beyond system boundaries are considered.
    # For this chart, "Lost" means not utilized as electricity or useful fluid heat.
    # If Q_recovered_fluid is defined as Q_stack_heat_flux (total heat flux),
    # then P_electricity is part of that.
    # Let's use: Recovered_Elec, Recovered_Fluid_Net, and Lost (if any unaccounted)
    # Let Q_useful_fluid_heat = Q_stack_heat_flux - P_electricity_teg (heat that actually heats the fluid after TEG power extraction)
    Q_useful_fluid_heat = Q_stack_heat_flux - P_electricity_teg

    energy_labels = ['Electricity Produced', 'Net Heat to Fluid', 'Losses (Unaccounted)']
    energy_values = [P_electricity_teg, Q_useful_fluid_heat, 0]  # Assuming no other losses for this pie.
    # If Q_stack_heat_flux is the "input", then it's all recovered.

    # Alternative interpretation for "Energy Pie":
    # Input: Q_stack_heat_flux
    # Outputs: P_electricity_teg, Q_heat_to_fluid_AFTER_TEG_conversion
    # The term Q_recovered_fluid = Q_stack_heat_flux seems to mean total heat flow into HX.
    # Let's assume energy "lost" in this context means energy from source that doesn't become P_elec or Q_useful_fluid
    # If T_source emits more than Q_stack_heat_flux (e.g. to surroundings), that's a different loss.
    # Here, let's define "recovered" as P_elec and "remaining thermal energy" as Q_stack_heat_flux - P_elec.

    # Pie chart components:
    pie_energy_recovered_elec = P_electricity_teg
    pie_energy_recovered_thermal_fluid = Q_stack_heat_flux - P_electricity_teg  # Heat that continues past TEG
    pie_energy_lost = 0  # Assuming Q_stack_heat_flux is the input, and all of it is accounted for by elec or thermal.

    if total_energy_input_for_pie > 0:
        sizes_energy = [pie_energy_recovered_elec, pie_energy_recovered_thermal_fluid]
        labels_energy = [f'Electricity\n({pie_energy_recovered_elec:.1f} W)',
                         f'Net Thermal to Fluid\n({pie_energy_recovered_thermal_fluid:.1f} W)']
        if pie_energy_lost > 1e-3:  # Add losses if significant
            sizes_energy.append(pie_energy_lost)
            labels_energy.append(f'Losses\n({pie_energy_lost:.1f} W)')
    else:
        sizes_energy = [1]
        labels_energy = ['No Energy Input']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes_energy, labels=labels_energy, autopct='%1.1f%%', startangle=90,
            colors=['lightblue', 'lightcoral', 'lightgrey'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Energy Utilization of Transferred Heat (Q_stack_heat_flux)')
    plt.tight_layout()

# 2. Breakdown of Lost Exergy Pie Chart
exergy_lost_labels = []
exergy_lost_values = []

if X_dest_rad > 1e-3:
    exergy_lost_labels.append(f'Radiation Source-Surface\n({X_dest_rad:.2f} W)')
    exergy_lost_values.append(X_dest_rad)
if X_dest_adh1 > 1e-3:
    exergy_lost_labels.append(f'Adhesive 1\n({X_dest_adh1:.2f} W)')
    exergy_lost_values.append(X_dest_adh1)
if X_dest_cu1 > 1e-3:
    exergy_lost_labels.append(f'Copper 1\n({X_dest_cu1:.2f} W)')
    exergy_lost_values.append(X_dest_cu1)
if X_dest_teg > 1e-3:
    exergy_lost_labels.append(f'TEG Module (thermal & conversion)\n({X_dest_teg:.2f} W)')
    exergy_lost_values.append(X_dest_teg)
if X_dest_cu2 > 1e-3:
    exergy_lost_labels.append(f'Copper 2\n({X_dest_cu2:.2f} W)')
    exergy_lost_values.append(X_dest_cu2)
if X_dest_adh2 > 1e-3:
    exergy_lost_labels.append(f'Adhesive 2\n({X_dest_adh2:.2f} W)')
    exergy_lost_values.append(X_dest_adh2)
if X_dest_hx_wall > 1e-3:
    exergy_lost_labels.append(f'HX Wall\n({X_dest_hx_wall:.2f} W)')
    exergy_lost_values.append(X_dest_hx_wall)
if X_dest_conv_fluid > 1e-3:
    exergy_lost_labels.append(f'Convection to Fluid\n({X_dest_conv_fluid:.2f} W)')
    exergy_lost_values.append(X_dest_conv_fluid)

# Ensure the sum for pie chart matches X_destroyed_total or is sum of plotted components
total_plotted_exergy_destruction = sum(exergy_lost_values)

# Assuming exergy_lost_labels and exergy_lost_values are defined elsewhere in your code
if abs(total_plotted_exergy_destruction - X_destroyed_total) > 1e-2 and X_destroyed_total > 0:  # If there's a notable difference
    unaccounted_destruction = X_destroyed_total - total_plotted_exergy_destruction
    if unaccounted_destruction > 1e-3:  # Add if significant
        exergy_lost_labels.append(f'Unaccounted / Residual\n({unaccounted_destruction:.2f} W)')
        exergy_lost_values.append(unaccounted_destruction)

if total_plotted_exergy_destruction > 0:
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(exergy_lost_values, labels=exergy_lost_labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title(f'Breakdown of Lost Exergy (Total: {X_destroyed_total:.2f} W)')

    # Adjust the legend to avoid overlap
    for text in texts + autotexts:
        text.set_fontsize(8)
    plt.legend(wedges, exergy_lost_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
else:
    fig2, ax2 = plt.subplots()
    ax2.text(0.5, 0.5, 'No significant exergy destruction calculated.', ha='center', va='center')
    ax2.set_title('Breakdown of Lost Exergy')
plt.tight_layout()

# 3. Resistances Bar Chart
resistance_labels = ['R_rad_eff', 'R_adh1', 'R_cu1', 'R_teg_therm', 'R_cu2', 'R_adh2', 'R_hx_wall', 'R_conv_fluid']
resistance_values = [R_rad_eff, R_adhesive1, R_cu1, R_teg_thermal, R_cu2, R_adhesive2, R_hx_wall, R_conv_fluid]

fig3, ax3 = plt.subplots(figsize=(10, 6))
bars = ax3.bar(resistance_labels, resistance_values,
               color=['skyblue', 'salmon', 'lightgreen', 'gold', 'salmon', 'lightgreen', 'tan', 'plum'])
ax3.set_ylabel('Thermal Resistance (K/W)')
ax3.set_title('Thermal Resistances in the System')
plt.xticks(rotation=45, ha="right")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0005 * max(resistance_values), f'{yval:.2e}', ha='center',
             va='bottom', fontsize=8)
plt.tight_layout()

# --- Outputting Key Calculated Values ---
print("\n--- Key Calculated Values ---")
print(f"Average fluid temperature: {T_fluid_avg_C:.2f} °C ({T_fluid_avg_K:.2f} K)")
print(f"Heat Source Temperature: {T_source_C:.2f} °C ({T_source_K:.2f} K)")
print(f"TEG Assembly First Surface Temperature (T_s1): {T_s1_C_sol:.2f} °C ({T_s1_K_sol:.2f} K)")
print(f"TEG Assembly First Surface Temperature (T_s1): {eta_carnot(T_teg_cold_K - 273.15,T_teg_hot_K - 273.15)*eta_teg_to_carnot_ratio:.2f} [-]")
print(f"TEG Hot Side Temperature: {T_teg_hot_K - 273.15:.2f} °C ({T_teg_hot_K:.2f} K)")
print(f"TEG Cold Side Temperature: {T_teg_cold_K - 273.15:.2f} °C ({T_teg_cold_K:.2f} K)")

print(f"\nHeat recovered by fluid (stack heat flux, Q_stack_heat_flux): {Q_stack_heat_flux:.2f} W")
print(f"Efficiency of TEG (eta_electricity_teg): {eta_teg * 100:.2f} % [-] ")
print(f"Efficiency of TEG based on calc (eta_electricity_teg): {eta_carnot(T_teg_cold_K - 273.15,T_teg_hot_K - 273.15) * 100*eta_teg_to_carnot_ratio:.2f} % [-] ")



print(
    f"Electricity produced by TEG (P_electricity_teg): {P_electricity_teg:.2f} W (assuming {eta_teg * 100}% TEG efficiency)")

print(f"\nExergy Input (from Q_rad at T_source): {X_in_q_rad:.2f} W")
print(f"Exergy Recovered (Electricity): {X_recovered_electricity:.2f} W")
print(f"Exergy Recovered (Fluid Heat at T_fluid_avg): {X_recovered_fluid_heat:.2f} W")
print(f"Total Exergy Recovered: {X_recovered_total:.2f} W")
print(f"Total Exergy Destroyed: {X_destroyed_total:.2f} W")

# Display plots


print("\n--- Assumptions Made ---")
print(f"- Convective heat transfer coefficient for fluid (h_fluid): {h_fluid} W/m^2K")
print(f"- TEG energy conversion efficiency (eta_teg): {eta_teg * 100}% of heat conducted through TEG")
print(
    f"- Emissivity used for radiation source (emissivity_source): {emissivity_source:.2f} (capped at 0.99 if calculated > 1)")
print("- One-dimensional steady-state heat transfer.")
print("- Material properties constant with temperature.")
print("- View factor for radiation F=1.")
print("- No thermal contact resistance at layer interfaces beyond specified 'adhesive' layers.")

plt.show()
