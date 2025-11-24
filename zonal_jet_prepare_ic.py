import numpy as np
import xarray as xr
from Flow2DFFT import Flow2D
import time 

# setup model parameters
Lx = 1 
Ly = 0.5 
dx = 0.01 
dy = 0.01 
dt = 1e-4 

# initialize initial condition arrays
nx, ny = int(Lx / dx) + 1, int(Ly / dy) + 1
zeta0 = np.zeros((ny, nx))


U0 = 1 # scale velocity

# build initial condition corresponding to a zonal jet with initial velocity U0 = 1 
for j in range(ny):
    for i in range(nx):
        zeta0[j,i] = -2* np.pi * U0 / Ly * np.sin(2 *np.pi * j / (ny - 1))

# add small noise to initial condition, causes barotropic instability faster
noise_scale = 0.02
rng = np.random.default_rng(0) # seed so output is reproducible
zeta0 += noise_scale * zeta0 * rng.uniform(-1, 1, (ny, nx)) 

# setup grid
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)

ds = xr.Dataset(data_vars={'vorticity':(['y', 'x'], zeta0),
                            'Lx': ([], Lx),
                            'Ly': ([], Ly),
                            'dx': ([], dx),
                            'dy': ([], dy)},
                coords={
                        'x':('x', x),
                        'y':('y', y)
                            })


ds.to_netcdf('nc_files/zonal_jet_ic.nc')