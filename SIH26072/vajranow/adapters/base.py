from abc import ABC, abstractmethod
from datetime import datetime

import xarray as xr


class BaseAdapter(ABC):
    """Abstract Base Class for all VajraNow data adapters.

    All adapters regrid to the common domain grid (~2 km, 10-minute cadence)
    and return an xarray.Dataset containing data variables and availability masks.
    """

    @abstractmethod
    def get(self, t0: datetime, t1: datetime) -> xr.Dataset:
        """Fetch and regrid multi-source observations for the window [t0, t1].

        Returns:
            xarray.Dataset containing:
                - composite_dbz: (time, y, x) in dBZ
                - radar_availability: (radar, time, y, x) binary mask (0 or 1)
                - satellite_ir: (time, band, y, x) in Kelvin
                - lightning_density: (time, y, x) strike density proxy
                - env_cape: (time, y, x) in J/kg (optional/available)
                - env_cin: (time, y, x) in J/kg (optional/available)
                - env_pwat: (time, y, x) in mm (optional/available)
                - env_shear: (time, y, x) in m/s (optional/available)
        """
        pass
