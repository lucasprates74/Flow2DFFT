import numpy as np
import xarray as xr
from Flow2DFFT import Flow2D
import time 

# setup time step
C = 0.1 # courant number
Re = 2000 # reynolds number
dt = 1e-4 
T = 1 
history_interval = 100 # timestep interval for saving data

# load initial condition with initial velocity zero
ds_in = xr.open_dataset('nc_files/zonal_jet_ic.nc')
zeta0 = ds_in.vorticity

# load grid parameters
Lx = ds_in.Lx.data
Ly = ds_in.Ly.data
dx = ds_in.dx.data 
dy = ds_in.dy.data

# setup initial velocity 
U0 = C * dx / dt # scale velocity
kappa = U0 * Ly / Re # diffusivity
print(U0, kappa)

# rescale voriticity in terms of scale velocity
zeta0 = U0 * zeta0

# setup observation mask 
obsmask = np.zeros_like(zeta0, dtype=np.int64)
# obsmask[:,0:5]=1
obsmask[::10,::10]=1

# solve
solver = Flow2D(zeta0, dt, dx, dy, T, history_interval, kappa)

start = time.time()
# ds = solver.enkf(nens=100,bscale=0.25, rscale=0.001, tobs=10000000, obsmask=obsmask)
ds = solver.enkf(nens=100, bscale=0.25, rscale=0.001, tobs=500, obsmask=obsmask)
end = time.time()
print('Seconds Elapsed:', end - start)
ds.to_netcdf('nc_files/ensemble_output.nc')