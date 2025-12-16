import numpy as np
import xarray as xr
from Flow2DFFT import Flow2D
import time 
import argparse 

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
parser = argparse.ArgumentParser()
parser.add_argument("nens", type = int)
parser.add_argument("stdr", type = float)
parser.add_argument("tobs", type = int) 
parser.add_argument("o", type = int) 
parser.add_argument("-ns", "--noise_scale", default = 1, type = float) 
args = parser.parse_args()
nens=args.nens
noise_scale=args.noise_scale
stdr=args.stdr
tobs=args.tobs
o = args.o

# setup observation mask 
obsmask = np.zeros_like(zeta0, dtype=np.int64)
ny, nx = obsmask.shape
if tobs != -1:
    if o == 1:
        obsmask[::10,::10]=1
    elif o == 2:
        obsmask[::10,0:int(nx//2):10]=1
    elif o == 3:
        obsmask[:,int(nx//2)-2:int(nx//2)+2]=1

print('nobs =', np.sum(obsmask))

start = time.time()
ds, rms, rmsu, rmsv = solver.enkf(nens=nens, stdr=stdr, tobs=tobs, obsmask=obsmask, noise_scale=noise_scale)
end = time.time()
print('Seconds Elapsed:', end - start)
if tobs==-1:
    ds.to_netcdf(f'nc_files/ensemble_output_n{nens}_b{noise_scale}_freerunning.nc')
    np.save(f'numpy_files/rms_n{nens}_b{noise_scale}_freerunning', rms)
    np.save(f'numpy_files/rmsu_n{nens}_b{noise_scale}_freerunning', rmsu)
    np.save(f'numpy_files/rmsv_n{nens}_b{noise_scale}_freerunning', rmsv)
else:
    ds.to_netcdf(f'nc_files/ensemble_output_n{nens}_b{noise_scale}_r{stdr}_t{tobs}_o{o}.nc')
    np.save(f'numpy_files/rms_n{nens}_b{noise_scale}_r{stdr}_t{tobs}_o{o}', rms)
    np.save(f'numpy_files/rmsu_n{nens}_b{noise_scale}_r{stdr}_t{tobs}_o{o}', rmsu)
    np.save(f'numpy_files/rmsv_n{nens}_b{noise_scale}_r{stdr}_t{tobs}_o{o}', rmsv)