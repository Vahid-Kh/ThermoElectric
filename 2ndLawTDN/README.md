<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

</head>
<body>
        <h1>Descreption</h1>
    <h2>Boundary condition:</h2>
    <p><strong>Case 1</strong>—you specify the surface temperature of the heat source (<code>T_source_C</code>).</p>
    <p><strong>Case 2</strong>—you specify the total heat flux arriving at that surface (<code>q_source_total</code>) and solve for the surface temperature that makes the energy balance close.</p>
    <h2>Solution strategy:</h2>
    <p><strong>Case 1</strong> is a forward calculation: given <code>T_source</code>, you compute the stack heat flux, then cascade through the TEG and heat‐exchanger resistances to find all temperatures, heat‐fluxes, and exergy streams.</p>
    <p><strong>Case 2</strong> is an inverse (root‐finding) problem: you impose a total input flux, then use <code>fsolve</code> to find the surface temperature <code>Ts</code> at which</p>
    <p><code>q_stack(Ts) + q_rad(Ts) + q_conv(Ts) = q_source_total.</code></p>
    <p>Once you have <code>Ts</code>, you compute the same downstream fluxes and exergy terms.</p>
    <h2>Use cases:</h2>
    <p><strong>Case 1</strong> is ideal when you know the hot‐body’s temperature (e.g. a furnace wall at 500 °C).</p>
    <p><strong>Case 2</strong> is ideal when you know the incident heat flux (e.g. a solar irradiance of 800 W/m²), but the panel’s equilibrium temperature is unknown.</p>
    <h2>Outputs:</h2>
    <p><strong>Case 1</strong> returns <code>q_stack</code>, component exergy flows, and (optionally) radiative losses based on the pre‐set <code>T_source</code>.</p>
    <p><strong>Case 2</strong> returns the solved surface temperature <code>Ts_C</code>, the split of that flux into conduction, radiation, and convection, plus the full exergy‐balance and 2nd‐law efficiency of the system under that load.</p>
    <h1>Computations</h1>
    <h2>Computes the heat transferred to the water:</h2>
    <p>
        <math>
            <msub><mrow>Q</mrow><mrow>.</mrow></msub> = 
            <msub><mrow>m</mrow><mrow>.</mrow></msub> 
            <msub><mrow>c</mrow><mrow>p</mrow></msub> 
            (T<sub>out</sub> − T<sub>in</sub>)
        </math>
    </p>
    <h2>Computes the exergy supplied by the hot source at constant temperature T<sub>s</sub>:</h2>
    <p>
        <math>
            <msub><mrow>E</mrow><mrow>.</mrow></msub><sub>in</sub> = 
            <msub><mrow>Q</mrow><mrow>.</mrow></msub> 
            (1 − T<sub>0</sub>/T<sub>s</sub>)
        </math>
    </p>
    <h2>Computes the exergy gain of the water stream:</h2>
    <p>
        <math>
            <msub><mrow>E</mrow><mrow>.</mrow></msub><sub>out</sub> = 
            <msub><mrow>m</mrow><mrow>.</mrow></msub> 
            [<msub><mrow>c</mrow><mrow>p</mrow></msub> 
            (T<sub>out</sub> − T<sub>in</sub>) − T<sub>0</sub> 
            <msub><mrow>c</mrow><mrow>p</mrow></msub> 
            ln(T<sub>out</sub>/T<sub>in</sub>)]
        </math>
    </p>
    <h2>Finds the exergy destruction:</h2>
    <p>
        <math>
            <msub><mrow>E</mrow><mrow>.</mrow></msub><sub>dest</sub> = 
            <msub><mrow>E</mrow><mrow>.</mrow></msub><sub>in</sub> − 
            <msub><mrow>E</mrow><mrow>.</mrow></msub><sub>out</sub>
        </math>
    </p>
</body>
</html>

