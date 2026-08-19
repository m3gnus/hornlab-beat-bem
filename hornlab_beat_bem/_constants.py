from __future__ import annotations

# These mirror the values the vendored BEAT Engine Julia solver defaults to and
# the values the other HornLab native packages publish. WG asks the package for
# them (``server/solver/acoustics.py``), so they must describe the physics that
# actually ran: ``sweep.py`` passes both into every Julia solve request.
SPEED_OF_SOUND: float = 343.0        # m/s, air at ~20 °C
AIR_DENSITY: float = 1.2041          # kg/m^3, matches hornlab-metal/bempp-bem
REFERENCE_PRESSURE: float = 20e-6    # Pa (20 µPa, standard ref)
