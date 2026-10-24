"""
sensor_simulator.py
Simulates an IoT sensor node that sends soil moisture, temperature and humidity.
No physical hardware is used: every call returns NEW random readings,
just like a real sensor would send a new reading every few seconds.
"""
import numpy as np

# One shared random generator (no fixed seed, so every call gives new values)
_rng = np.random.default_rng()


def get_sensor_reading():
    """Return one simulated reading as a dictionary."""
    return {
        "soil_moisture": int(np.clip(_rng.normal(40, 18), 0, 100)),   # %
        "temperature": int(np.clip(_rng.normal(28, 7), 10, 45)),      # degree C
        "humidity": int(np.clip(_rng.normal(60, 18), 10, 100)),       # %
    }


if __name__ == "__main__":
    # Quick test: python sensor_simulator.py
    for _ in range(5):
        print(get_sensor_reading())
