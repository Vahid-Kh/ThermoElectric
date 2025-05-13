import numpy as np

def exergy_destruction(
    T_source_C: float,
    T_water_in_C: float,
    T_water_out_C: float,
    m_dot: float = 1.0,
    cp: float = 4186.0,
    T0_C: float = 25.0
) -> dict:
    """
    Calculate exergy destruction in a heat exchanger.

    Parameters
    ----------
    T_source_C : float
        Temperature of the heat source [°C].
    T_water_in_C : float
        Inlet water temperature [°C].
    T_water_out_C : float
        Outlet water temperature [°C].
    m_dot : float, optional
        Mass flow rate of water [kg/s]. Default is 1 kg/s.
    cp : float, optional
        Specific heat capacity of water [J/(kg·K)]. Default is 4186 J/(kg·K).
    T0_C : float, optional
        Ambient (dead‐state) temperature [°C]. Default is 25 °C.

    Returns
    -------
    dict
        Dictionary containing:
        - Q_dot: Heat transfer rate [W]
        - Ex_in: Exergy input rate [W]
        - Ex_out: Exergy output rate [W]
        - Ex_dest: Exergy destruction rate [W]
    """
    # Convert temperatures to Kelvin
    T_s = T_source_C + 273.15
    T_in = T_water_in_C + 273.15
    T_out = T_water_out_C + 273.15
    T0 = T0_C + 273.15

    # Heat transfer rate (W)
    Q_dot = m_dot * cp * (T_out - T_in)

    # Exergy input from source (W)
    Ex_in = Q_dot * (1.0 - T0 / T_s)

    # Exergy gain of the water (W)
    Ex_out = m_dot * (cp * (T_out - T_in) - T0 * cp * np.log(T_out / T_in))

    # Exergy destruction (W)
    Ex_dest = Ex_in - Ex_out

    return {
        'Q_dot (W)': Q_dot,
        'Ex_in (W)': Ex_in,
        'Ex_out (W)': Ex_out,
        'Ex_dest (W)': Ex_dest
    }

if __name__ == "__main__":
    results = exergy_destruction(
        T_source_C=400.0,
        T_water_in_C=30.0,
        T_water_out_C=50.0,
        m_dot=1.0,       # kg/s
        cp=4186.0,       # J/(kg·K)
        T0_C=25.0        # °C ambient
    )
    for k, v in results.items():
        print(f"{k}: {v:,.2f}")
