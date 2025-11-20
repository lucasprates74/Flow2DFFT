import numpy as np
import xarray as xr
from Flow2DFFT import Flow2D
import time 

# setup model parameters
C = 0.1 # courant number
mu = 3e-3 # diffusivity scale
Lx = 1 
Ly = 0.5 
dx = 0.01 
dy = 0.01 
dt = 1e-4 
T = 1 
history_interval = 100 # timestep interval for saving data

# initialize initial condition arrays
nx, ny = int(Lx / dx) + 1, int(Ly / dy) + 1
zeta0 = np.zeros((ny, nx))


U0 = C * dx / dt #0.2 # scale velocity
kappa = mu * dx ** 2 / dt # diffusivity
print(U0, kappa)

# build initial condition corresponding to a zonal jet
for j in range(ny):
    for i in range(nx):
        zeta0[j,i] = -2* np.pi * U0 / Ly * np.sin(2 *np.pi * j / (ny - 1))


# solve
solver = Flow2D(zeta0, dt, dx, dy, T, history_interval, kappa)

start = time.time()
ds = solver.solve()
end = time.time()
print('Seconds Elapsed:', end - start)
ds.to_netcdf('new_model_output.nc')