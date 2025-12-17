import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
import seaborn as sns

nens=100
bscale = 1
rscale = 5.0#1000000
tobs = 1000#1000#500
o = 3

if tobs==-1:
    inname = f'ensemble_output_n{nens}_b{bscale}_freerunning.nc'
    outname = f'animation_enkf_n{nens}_b{bscale}_freerunning.mp4'
else:
    inname = f'ensemble_output_n{nens}_b{bscale}_r{rscale}_t{tobs}_o{o}.nc'
    outname = f'ensemble_enkf_n{nens}_b{bscale}_r{rscale}_t{tobs}_o{o}.mp4'

ds = xr.open_dataset(f'nc_files/' + inname)
u = ds.u
v = ds.v
zeta = ds.vorticity
t = ds.time

# get variables
ua = ds.u_mean#ens.mean('ensemble_id')
va = ds.v_mean#ens.mean('ensemble_id')
zetaa = ds.vorticity_mean#ens.mean('ensemble_id')

# compute variance and corr
zeta_var = ds.vorticity_var
zeta_corr = ds.vorticity_corr

# get rmse
mse = zeta_var.sum(('y', 'x'))
rmse = np.sqrt(mse).data

# normalize
zeta_var = zeta_var/mse #zeta_var.max(('x','y'))#/mse  # normalize

# get obsmask
obsmask = ds.obsmask

# setup maxima
MZ=np.ceil(zeta.max().data)
M=np.ceil(u.max().data)
V=zeta_var.max().data

# get grid specs
x, y = ds.x, ds.y
nframes = len(ds.time.values)

sns.set_context('talk')
fig, axes2d = plt.subplots(4, 2, figsize=(8, 12), constrained_layout=True)

axes = axes2d.ravel()
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(r'$t=0.00$')

ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes
c1 = ax1.pcolormesh(x, y, zeta.isel(time=0), cmap="coolwarm", vmin=-MZ, vmax=MZ)
c3 = ax3.pcolormesh(x, y, u.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c5 = ax5.pcolormesh(x, y, v.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c2 = ax2.pcolormesh(x, y, zetaa.isel(time=0), cmap="coolwarm", vmin=-MZ, vmax=MZ)
c4 = ax4.pcolormesh(x, y, ua.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c6 = ax6.pcolormesh(x, y, va.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
# c7 = ax7.pcolormesh(x, y, obsmask, cmap="viridis", vmin=0, vmax=1)
# c8 = ax8.pcolormesh(x, y, zeta_var.isel(time=0), cmap="viridis", vmin=0, vmax=V)
# c8 = ax8.pcolormesh(x, y, zeta_corr.isel(time=0), cmap="PRGn", vmin=-1, vmax=1)

ax1.set_title('Vorticity (Truth)')
ax3.set_title('Horiz. Velocity (Truth)')
ax5.set_title('Vert. Velocity (Truth)')
ax2.set_title('Vorticity (Forecast)')
ax4.set_title('Horiz. Velocity (Forecast)')
ax6.set_title('Vert. Velocity (Forecast)')
# ax7.set_title('Observing Network')
# ax8.set_title(f'Forecast Variance/MSE')
# ax8.set_title(f'Correlation')

fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
fig.colorbar(c3, ax=ax3)
fig.colorbar(c4, ax=ax4)
fig.colorbar(c5, ax=ax5)
fig.colorbar(c6, ax=ax6)
# fig.colorbar(c7, ax=ax7)
# fmt = ticker.ScalarFormatter()
# fmt.set_powerlimits((0, 0)) 
# fig.colorbar(c8, ax=ax8, format=fmt)
plt.show()


def __update(frame):
    global c1, c2, c3, c4, c5, c6, nframes

    print(f'frame {frame}/{nframes}')
    # Remove the data
    c1.remove()
    c2.remove()
    c3.remove()
    c4.remove()
    c5.remove()
    c6.remove()
    
    tf = round(t.data[frame], 2)
    fig.suptitle(r'$t=$'+f'{tf:.2f}')
    # Draw the new frame
    c1 = ax1.pcolormesh(x, y, zeta.isel(time=frame), cmap="coolwarm", vmin=-MZ, vmax=MZ)
    c3 = ax3.pcolormesh(x, y, u.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c5 = ax5.pcolormesh(x, y, v.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c2 = ax2.pcolormesh(x, y, zetaa.isel(time=frame), cmap="coolwarm", vmin=-MZ, vmax=MZ)
    c4 = ax4.pcolormesh(x, y, ua.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c6 = ax6.pcolormesh(x, y, va.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    # c8 = ax8.pcolormesh(x, y, zeta_var.isel(time=frame), cmap="viridis", vmin=0, vmax=V)
    # c8 = ax8.pcolormesh(x, y, zeta_corr.isel(time=frame), cmap="PRGn", vmin=-1, vmax=1)
    # ax8.set_title(f'Variance/MSE (RMSE={round(rmse[frame], 2)})')

    return [c1, c2, c3, c4, c5, c6]

ani = FuncAnimation(fig=fig, func=__update, frames=range(len(ds.time)), interval=100)

ani.save(f'animations/' + outname)
print('done')
