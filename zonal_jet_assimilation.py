import numpy as np
import xarray as xr
from Flow2DFFT import Flow2D
import time 

# setup time step
C = 0.1 # courant number
Re = 4000#2200 # reynolds number
dt = 2e-4#1e-4 
T = 1
history_interval = 50 # timestep interval for saving data

# load initial condition with initial velocity zero
ds_in = xr.open_dataset('nc_files/zonal_jet_ic.nc')
zeta0 = ds_in.vorticity

# load grid parameters
Lx = ds_in.Lx.data
Ly = ds_in.Ly.data
dx = ds_in.dx.data 
dy = ds_in.dy.data

# setup initial velocity 
U = C * dx / dt  # max velocity
U0 = 0.5 * U     # scale velocity
ubgd = 0.5 * U   # background u
vbgd = 0         # background v
kappa = U * Ly / Re # U * Ly / Re # diffusivity
print(U0, kappa)

# rescale voriticity in terms of scale velocity
zeta0 = U0 * zeta0

# get solver
solver = Flow2D(zeta0, ubgd, vbgd, dt, dx, dy, T, history_interval, kappa)

# enkf parameters
nens=100
noise_scale=1#0.01
stdr=50#2.0
tobs=1000
o = 1

# setup observation mask 
obsmask = np.zeros_like(zeta0, dtype=np.int64)
ny, nx = obsmask.shape
if o == 1:
    obsmask[::10,::10]=1
elif o == 2:
    obsmask[::10,0:int(nx//2):10]=1
elif o == 3:
    obsmask[:,0:2]=1

print('nobs =', np.sum(obsmask))

start = time.time()
ds = solver.enkf(nens=nens, stdr=stdr, tobs=tobs, obsmask=obsmask, noise_scale=noise_scale)
end = time.time()
print('Seconds Elapsed:', end - start)
if tobs==-1:
    ds.to_netcdf(f'nc_files/ensemble_output_n{nens}_b{noise_scale}_freerunning.nc')
else:
    ds.to_netcdf(f'nc_files/ensemble_output_n{nens}_b{noise_scale}_r{stdr}_t{tobs}_o{o}.nc')