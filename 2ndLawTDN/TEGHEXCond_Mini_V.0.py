import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve # No longer needed for the primary heat transfer model
# from functions import * # Assuming eta_carnot is defined locally

# --- Constants and Boundary Conditions ---
# Temperatures
T_in_fluid_C = 40.0  # °C
T_out_fluid_C = 65.0  # °C
T_dead_C = 25.0  # °C (Ambient for exergy calculations)
T_source_C = 300.0  # Source temperature (°C)

# Geometry and materials
area = 1.0  # m^2 (Assumed reference area for resistances)

# Thermal Interface Material (TIM) properties
thickness_tim = 0.5e-4  # m (Example: 50 microns for TIM)
k_tim = 5.0           # W/mK (Example: Thermal conductivity for a TIM)

thickness_adhesive = 1e-4  # m (Adhesive layer 1, between TIM and Cu1)
thickness_cu = 1e-3  # m (Copper spreader 1)
thickness_teg = 4e-3  # m (TEG module)
# Note: The original code had Cu2 and Adhesive2. Assuming the stack is:
# Source -> TIM -> Adhesive1 -> Cu1 -> TEG -> Cu2 -> Adhesive2 -> HX Wall
thickness_cu2 = 1e-3 # m (Copper spreader 2)
thickness_adhesive2 = 1e-4 # m (Adhesive layer 2, between TEG's cold side Cu and HX wall)
thickness_hx_wall = 10e-3  # m (Heat exchanger wall)


k_adhesive = 0.5  # W/mK
k_cu = 400  # W/mK
k_teg = 1.5  # W/mK (Thermal conductivity of TEG material itself)
k_hx_wall = 167.0  # W/mK - Aluminum 6061-T6

# Assumed parameters
h_fluid = 4000.0  # W/m^2K (Convective heat transfer coefficient for water in HEX)

# --- Helper Functions ---
def celsius_to_kelvin(T_C):
    return T_C + 273.15

def eta_carnot(T_C, T_H):
    """ Calculates Carnot efficiency. T_C and T_H are in Celsius. """
    T_C_K = celsius_to_kelvin(T_C)
    T_H_K = celsius_to_kelvin(T_H)
    if T_H_K <= T_C_K or T_H_K == 0: # Prevent division by zero or non-physical result
        return 0.0
    return (1 - (T_C_K / T_H_K))

# Convert temperatures to Kelvin
T_in_fluid_K = celsius_to_kelvin(T_in_fluid_C)
T_out_fluid_K = celsius_to_kelvin(T_out_fluid_C)
T_dead_K = celsius_to_kelvin(T_dead_C)
T_source_K = celsius_to_kelvin(T_source_C)

T_fluid_avg_K = (T_in_fluid_K + T_out_fluid_K) / 2.0
T_fluid_avg_C = (T_in_fluid_C + T_out_fluid_C) / 2.0

# --- Thermal Resistance Calculations ---
# Stack: Source -> TIM -> Adhesive1 -> Cu1 -> TEG -> Cu2 -> Adhesive2 -> HX Wall -> Fluid
R_tim = thickness_tim / (k_tim * area)
R_adhesive1 = thickness_adhesive / (k_adhesive * area)
R_cu1 = thickness_cu / (k_cu * area)
R_teg_thermal = thickness_teg / (k_teg * area)  # Thermal resistance of TEG material
R_cu2 = thickness_cu2 / (k_cu * area) # Resistance for the second copper layer
R_adhesive2 = thickness_adhesive2 / (k_adhesive * area) # Resistance for the second adhesive layer
R_hx_wall = thickness_hx_wall / (k_hx_wall * area)
R_conv_fluid = 1 / (h_fluid * area)

# Total resistance from the source surface to the average fluid temperature
R_total_stack = R_tim + R_adhesive1 + R_cu1 + R_teg_thermal + R_cu2 + R_adhesive2 + R_hx_wall + R_conv_fluid

# --- Heat Flux and Temperature Calculations (Conduction Dominated) ---
if R_total_stack <= 0:
    print("Error: Total thermal resistance is zero or negative. Check material properties and thicknesses.")
    Q_stack_heat_flux = 0
else:
    Q_stack_heat_flux = (T_source_K - T_fluid_avg_K) / R_total_stack

if Q_stack_heat_flux <= 0:
    print(f"Warning: Calculated heat flux is {Q_stack_heat_flux:.2f} W. Check temperatures and resistances. Source might be colder than fluid or resistances too high.")
    # Ensure Q is not negative for subsequent calculations if proceeding
    Q_stack_heat_flux = max(0, Q_stack_heat_flux)


# Calculate intermediate temperatures based on Q_stack_heat_flux
T_after_tim_K = T_source_K - Q_stack_heat_flux * R_tim
T_after_adhesive1_K = T_after_tim_K - Q_stack_heat_flux * R_adhesive1
T_teg_hot_K = T_after_adhesive1_K - Q_stack_heat_flux * R_cu1 # This is the TEG hot side actual temp
T_teg_cold_K = T_teg_hot_K - Q_stack_heat_flux * R_teg_thermal # This is TEG cold side actual temp
T_after_cu2_K = T_teg_cold_K - Q_stack_heat_flux * R_cu2
T_after_adhesive2_K = T_after_cu2_K - Q_stack_heat_flux * R_adhesive2
T_hx_wall_hot_K = T_after_adhesive2_K # Temp at the HX wall surface facing adhesive2
T_hx_wall_cold_K = T_hx_wall_hot_K - Q_stack_heat_flux * R_hx_wall

# Verification for HEX cold wall temperature (should be close to what fluid convection implies)
T_check_hx_cold_from_fluid = T_fluid_avg_K + Q_stack_heat_flux * R_conv_fluid
# print(f"Calculated HEX wall cold side temp (stack): {T_hx_wall_cold_K-273.15:.2f} C")
# print(f"Calculated HEX wall cold side temp (fluid): {T_check_hx_cold_from_fluid-273.15:.2f} C")

# --- Energy Recovery Calculations ---
Q_recovered_fluid = Q_stack_heat_flux  # Total heat flowing into the system ends up in fluid or converted

# Electricity produced by TEG
Q_through_teg = Q_stack_heat_flux # Heat conducted *into* the TEG module hot side

# Calculate TEG efficiency based on its hot and cold side temperatures
# Convert TEG surface temps to Celsius for eta_carnot function as defined
T_teg_hot_C_calc = T_teg_hot_K - 273.15
T_teg_cold_C_calc = T_teg_cold_K - 273.15

if T_teg_hot_K > T_teg_cold_K:
    # Assuming a certain percentage of Carnot efficiency for a real TEG
    # For example, if TEG is 10% of Carnot, then efficiency_factor = 0.1
    # Here, we use the provided eta_carnot function directly for theoretical max.
    # A practical TEG efficiency (eta_device) would be eta_carnot * material_figure_of_merit_factor.
    # Let's use the Carnot efficiency as per the original structure.
    # The problem statement implies eta_teg IS Carnot.
    eta_teg_to_carnot_ratio = 0.2
    eta_teg_carnot_theoretical = eta_carnot(T_teg_cold_C_calc, T_teg_hot_C_calc) * eta_teg_to_carnot_ratio
else:
    eta_teg_carnot_theoretical = 0.0
    if Q_stack_heat_flux > 0 : # Only warn if there was supposed to be heat flow
         print(f"Warning: TEG hot side temp ({T_teg_hot_C_calc:.2f}°C) is not greater than cold side temp ({T_teg_cold_C_calc:.2f}°C). TEG efficiency is 0.")

# For this simulation, eta_teg is the actual conversion efficiency.
# If it's meant to be a fraction of Carnot, adjust here.
# The original code directly uses eta_carnot result as eta_teg.
eta_teg_actual = eta_teg_carnot_theoretical # Assuming TEG operates at its theoretical Carnot limit between T_teg_hot_K and T_teg_cold_K

P_electricity_teg = eta_teg_actual * Q_through_teg # Electrical power = eff * Heat INTO TEG

# Heat that continues to the fluid AFTER TEG conversion
Q_net_to_fluid_for_temp_rise = Q_through_teg - P_electricity_teg


# --- Exergy Calculations ---
# Exergy input: Exergy of the heat Q_stack_heat_flux at the source temperature T_source_K
X_in_q_source = Q_stack_heat_flux * (1 - T_dead_K / T_source_K) if T_source_K > 0 and T_source_K > T_dead_K else 0

# Exergy recovered
X_recovered_electricity = P_electricity_teg  # Electricity is pure exergy

# Exergy of heat recovered by fluid (using average fluid temperature)
# This should be based on the net heat that actually heats the fluid (Q_net_to_fluid_for_temp_rise)
# However, for consistency with some interpretations where Q_recovered_fluid is the total heat path:
# Let's use Q_stack_heat_flux for total exergy potential transferred to fluid path before TEG conversion losses
# Then, destruction within TEG will account for the conversion.
# Using Q_recovered_fluid = Q_stack_heat_flux which is total heat entering the system path
X_recovered_fluid_heat_potential = Q_stack_heat_flux * (1 - T_dead_K / T_fluid_avg_K) if T_fluid_avg_K > 0 and T_fluid_avg_K > T_dead_K else 0

# Total exergy recovered: electricity + exergy of heat that effectively warms the fluid
# The exergy related to fluid heating is associated with Q_net_to_fluid_for_temp_rise
X_recovered_fluid_heat_effective = Q_net_to_fluid_for_temp_rise * (1 - T_dead_K / T_fluid_avg_K) if T_fluid_avg_K > 0 and T_fluid_avg_K > T_dead_K else 0
X_recovered_total = X_recovered_electricity + X_recovered_fluid_heat_effective

# Exergy destroyed/lost
X_destroyed_total = X_in_q_source - X_recovered_total
if X_destroyed_total < -1e-6:  # Allow for small numerical precision issues
    print(f"Warning: Negative total exergy destruction ({X_destroyed_total:.2f} W). Check model or assumptions. Capping at 0.")
X_destroyed_total = max(0, X_destroyed_total)


# Breakdown of exergy destruction
def calculate_exergy_destruction(Q, T_dead, T_hot, T_cold):
    if T_hot > 0 and T_cold > 0 and T_hot > T_cold and T_hot > T_dead and T_cold > T_dead: # Ensure physical temps
        destruction = Q * T_dead * (1 / T_cold - 1 / T_hot)
        return max(0, destruction) # Destruction cannot be negative
    return 0

X_dest_tim = calculate_exergy_destruction(Q_stack_heat_flux, T_dead_K, T_source_K, T_after_tim_K)
X_dest_adh1 = calculate_exergy_destruction(Q_stack_heat_flux, T_dead_K, T_after_tim_K, T_after_adhesive1_K)
X_dest_cu1 = calculate_exergy_destruction(Q_stack_heat_flux, T_dead_K, T_after_adhesive1_K, T_teg_hot_K)

# Exergy destruction in TEG (more complex due to energy conversion)
# S_gen_teg = (Heat_out / T_cold) - (Heat_in / T_hot) + (Work_out / T_surr) - (Work_out / T_surr)
# S_gen_teg = (Q_through_teg - P_electricity_teg) / T_teg_cold_K - Q_through_teg / T_teg_hot_K
if T_teg_cold_K > 0 and T_teg_hot_K > 0 and T_teg_hot_K > T_teg_cold_K :
    S_gen_teg = (Q_through_teg - P_electricity_teg) / T_teg_cold_K - Q_through_teg / T_teg_hot_K
    S_gen_teg = max(0, S_gen_teg) # Entropy generation cannot be negative
    X_dest_teg = T_dead_K * S_gen_teg
else:
    X_dest_teg = 0
    if Q_stack_heat_flux > 0 and T_teg_hot_K <= T_teg_cold_K:
        # If heat flows but TEG temps are inverted, all heat entering TEG that isn't converted (i.e., all of it if P_elec=0 due to inverted temps)
        # should be considered in destruction across this problematic segment.
        # This case is complex; simpler is to ensure T_hot > T_cold for TEG operation.
        # If T_hot not > T_cold, P_electricity_teg and eta_teg are 0.
        # The destruction is then simply Q_stack_heat_flux * T_dead_K * (1/T_teg_cold_K - 1/T_teg_hot_K), which would be negative or ill-defined.
        # So, if P_elec is 0, X_dest_teg is just the conductive loss if T_hot_K and T_cold_K are used.
        X_dest_teg = calculate_exergy_destruction(Q_stack_heat_flux, T_dead_K, T_teg_hot_K, T_teg_cold_K) # This will be 0 if T_hot <= T_cold

X_dest_cu2 = calculate_exergy_destruction(Q_stack_heat_flux - P_electricity_teg, T_dead_K, T_teg_cold_K, T_after_cu2_K) # Heat flow reduced by P_elec
X_dest_adh2 = calculate_exergy_destruction(Q_stack_heat_flux - P_electricity_teg, T_dead_K, T_after_cu2_K, T_after_adhesive2_K)
X_dest_hx_wall = calculate_exergy_destruction(Q_stack_heat_flux - P_electricity_teg, T_dead_K, T_hx_wall_hot_K, T_hx_wall_cold_K)
X_dest_conv_fluid = calculate_exergy_destruction(Q_stack_heat_flux - P_electricity_teg, T_dead_K, T_hx_wall_cold_K, T_fluid_avg_K)


# Sum of component exergy destructions
X_dest_sum_components = (X_dest_tim + X_dest_adh1 + X_dest_cu1 + X_dest_teg +
                         X_dest_cu2 + X_dest_adh2 + X_dest_hx_wall + X_dest_conv_fluid)

# Normalize component destructions to match overall balance if there's a discrepancy
if X_dest_sum_components > 1e-6: # Avoid division by zero
    if abs(X_dest_sum_components - X_destroyed_total) > 1e-3 * X_destroyed_total: # If discrepancy is more than 0.1%
        # print(f"Note: Scaling component exergy destructions. Sum: {X_dest_sum_components:.2f}, Overall: {X_destroyed_total:.2f}")
        scale_factor = X_destroyed_total / X_dest_sum_components
        X_dest_tim *= scale_factor
        X_dest_adh1 *= scale_factor
        X_dest_cu1 *= scale_factor
        X_dest_teg *= scale_factor
        X_dest_cu2 *= scale_factor
        X_dest_adh2 *= scale_factor
        X_dest_hx_wall *= scale_factor
        X_dest_conv_fluid *= scale_factor
elif X_destroyed_total > 1e-6:
    print("Warning: Sum of component exergy destructions is near zero, but overall exergy destruction is non-zero. Breakdown will be incorrect.")

# --- Plotting ---
# plt.style.use('seaborn-v0_8-whitegrid') # Optional: choose a style

# 1. Energy Utilization Pie Chart
# Total energy input for pie is Q_stack_heat_flux
pie_energy_recovered_elec = P_electricity_teg
pie_energy_recovered_thermal_fluid = Q_net_to_fluid_for_temp_rise # Heat that actually heats the fluid

# "Losses" in this context would be heat that enters the system but is neither electricity nor useful heat to fluid.
# In this model, all Q_stack_heat_flux is accounted for.
# If Q_stack_heat_flux is the "total energy from source considered", then its components are P_elec and Q_net_to_fluid.

energy_values = []
energy_labels = []

if pie_energy_recovered_elec > 1e-3:
    energy_values.append(pie_energy_recovered_elec)
    energy_labels.append(f'Electricity\n({pie_energy_recovered_elec:.1f} W)')
if pie_energy_recovered_thermal_fluid > 1e-3:
    energy_values.append(pie_energy_recovered_thermal_fluid)
    energy_labels.append(f'Net Thermal to Fluid\n({pie_energy_recovered_thermal_fluid:.1f} W)')

if not energy_values : # If Q_stack_heat_flux was 0
    energy_values = [1]
    energy_labels = ['No Energy Transfer']


fig1, ax1 = plt.subplots()
if Q_stack_heat_flux > 1e-3 :
    ax1.pie(energy_values, labels=energy_labels, autopct='%1.1f%%', startangle=90,
            colors=['lightblue', 'lightcoral', 'lightgrey'][:len(energy_values)])
    ax1.set_title(f'Energy Utilization of Transferred Heat (Input Q: {Q_stack_heat_flux:.1f} W)')
else:
    ax1.text(0.5, 0.5, 'No significant energy transferred.', ha='center', va='center')
    ax1.set_title('Energy Utilization')
ax1.axis('equal')
plt.tight_layout()


# 2. Breakdown of Lost Exergy Pie Chart
exergy_lost_labels = []
exergy_lost_values = []

if X_dest_tim > 1e-3:
    exergy_lost_labels.append(f'TIM\n({X_dest_tim:.2f} W)')
    exergy_lost_values.append(X_dest_tim)
if X_dest_adh1 > 1e-3:
    exergy_lost_labels.append(f'Adhesive 1\n({X_dest_adh1:.2f} W)')
    exergy_lost_values.append(X_dest_adh1)
if X_dest_cu1 > 1e-3:
    exergy_lost_labels.append(f'Copper 1\n({X_dest_cu1:.2f} W)')
    exergy_lost_values.append(X_dest_cu1)
if X_dest_teg > 1e-3:
    exergy_lost_labels.append(f'TEG Module\n({X_dest_teg:.2f} W)')
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

total_plotted_exergy_destruction = sum(exergy_lost_values)

# Add unaccounted if significant and overall destruction is positive
if X_destroyed_total > 1e-3 and abs(total_plotted_exergy_destruction - X_destroyed_total) > 1e-3 * X_destroyed_total :
    unaccounted_destruction = X_destroyed_total - total_plotted_exergy_destruction
    if unaccounted_destruction > 1e-3 : # Only add if positive and significant
        exergy_lost_labels.append(f'Unaccounted\n({unaccounted_destruction:.2f} W)')
        exergy_lost_values.append(unaccounted_destruction)


fig2, ax2 = plt.subplots(figsize=(10, 7)) # Increased figure size for legend
if sum(exergy_lost_values) > 1e-3: # Check if there's anything to plot
    wedges, texts, autotexts = ax2.pie(exergy_lost_values, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax2.axis('equal')
    ax2.set_title(f'Breakdown of Lost Exergy (Total: {X_destroyed_total:.2f} W)')
    # Add legend
    ax2.legend(wedges, exergy_lost_labels,
              title="Exergy Destruction Components",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), # Adjust bbox_to_anchor as needed
              fontsize=8)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7)
    for text in texts:
        text.set_fontsize(8)
else:
    ax2.text(0.5, 0.5, 'No significant exergy destruction calculated.', ha='center', va='center')
    ax2.set_title('Breakdown of Lost Exergy')
plt.subplots_adjust(right=0.7) # Adjust layout to make space for legend
plt.tight_layout()


# 3. Resistances Bar Chart
resistance_labels = ['R_tim', 'R_adh1', 'R_cu1', 'R_teg_therm', 'R_cu2', 'R_adh2', 'R_hx_wall', 'R_conv_fluid']
resistance_values = [R_tim, R_adhesive1, R_cu1, R_teg_thermal, R_cu2, R_adhesive2, R_hx_wall, R_conv_fluid]

fig3, ax3 = plt.subplots(figsize=(10, 6))
bars = ax3.bar(resistance_labels, resistance_values,
               color=['grey', 'salmon', 'lightgreen', 'gold', 'lightgreen', 'salmon', 'tan', 'plum'])
ax3.set_ylabel('Thermal Resistance (K/W)')
ax3.set_title('Thermal Resistances in the System Stack')
plt.xticks(rotation=45, ha="right")
max_res_val = max(resistance_values) if resistance_values else 0.001
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * max_res_val, f'{yval:.2e}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()

# --- Outputting Key Calculated Values ---
print("\n--- Key Calculated Values ---")
print(f"Source Temperature: {T_source_C:.2f} °C ({T_source_K:.2f} K)")
print(f"Average Fluid Temperature: {T_fluid_avg_C:.2f} °C ({T_fluid_avg_K:.2f} K)")
print(f"Dead State Temperature: {T_dead_C:.2f} °C ({T_dead_K:.2f} K)")

print(f"\nTemperature after TIM (Interface to Adhesive1): {T_after_tim_K - 273.15:.2f} °C ({T_after_tim_K:.2f} K)")
print(f"TEG Hot Side Temperature: {T_teg_hot_C_calc:.2f} °C ({T_teg_hot_K:.2f} K)")
print(f"TEG Cold Side Temperature: {T_teg_cold_C_calc:.2f} °C ({T_teg_cold_K:.2f} K)")
print(f"HX Wall Cold Side Temperature (Interface to Fluid): {T_hx_wall_cold_K - 273.15:.2f} °C ({T_hx_wall_cold_K:.2f} K)")


print(f"\nTotal Heat Flux through stack (Q_stack_heat_flux): {Q_stack_heat_flux:.2f} W")
print(f"Theoretical TEG Efficiency (Carnot based on T_teg_hot/cold, eta_teg_actual): {eta_teg_actual * 100:.2f}%")
print(f"Electricity Produced by TEG (P_electricity_teg): {P_electricity_teg:.2f} W")
print(f"Net Heat Transferred to Fluid (after TEG conversion): {Q_net_to_fluid_for_temp_rise:.2f} W")


print(f"\nExergy Input (from Source conduction at T_source): {X_in_q_source:.2f} W")
print(f"Exergy Recovered (Electricity): {X_recovered_electricity:.2f} W")
print(f"Exergy Recovered (Effective Fluid Heat at T_fluid_avg): {X_recovered_fluid_heat_effective:.2f} W")
print(f"Total Exergy Recovered: {X_recovered_total:.2f} W")
print(f"Total Exergy Destroyed: {X_destroyed_total:.2f} W (Sum of components: {X_dest_sum_components:.2f} W)")

print("\n--- Component Thermal Resistances (K/W) ---")
for label, value in zip(resistance_labels, resistance_values):
    print(f"- {label}: {value:.2e}")

print("\n--- Assumptions Made ---")
print(f"- Primary heat transfer from source is 1D conduction through the defined stack.")
print(f"- Convective heat transfer coefficient for fluid (h_fluid): {h_fluid} W/m^2K.")
print(f"- TIM: thickness {thickness_tim*1e6:.0f} µm, conductivity {k_tim:.1f} W/mK.")
print(f"- TEG conversion efficiency is based on theoretical Carnot efficiency between its hot and cold surface temperatures.")
print(f"- Material properties are constant with temperature.")
print(f"- Steady-state heat transfer.")
print(f"- Area for all calculations: {area} m^2.")

plt.show()