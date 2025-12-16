import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

ds = xr.open_dataset('nc_files/zonal_jet_output.nc')
u = ds.u
v = ds.v
zeta = ds.vorticity
t = ds.time

MZ=np.ceil(zeta.max().data)
M=np.ceil(u.max().data)
x, y = ds.x, ds.y
nframes = len(ds.time.values)

sns.set_context('talk')
fig, axes = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(r'$t=0.00$')

ax1, ax2, ax3 = axes
c1 = ax1.pcolormesh(x, y, zeta.isel(time=0), cmap="coolwarm", vmin=-MZ, vmax=MZ)
c2 = ax2.pcolormesh(x, y, u.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c3 = ax3.pcolormesh(x, y, v.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)

ax1.set_title('Vorticity')
ax2.set_title('Horiz. Velocity')
ax3.set_title('Vert. Velocity')


# set colorbars
fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
fig.colorbar(c3, ax=ax3)

plt.savefig('figures/initial_truth.png')

def __update(frame):
    global c1, c2, c3, nframes

    print(f'frame {frame}/{nframes}')
    # Remove the data
    c1.remove()
    c2.remove()
    c3.remove()
    
    tf = round(t.data[frame], 2)
    fig.suptitle(r'$t=$'+f'{tf:.2f}')
    # Draw the new frame
    c1 = ax1.pcolormesh(x, y, zeta.isel(time=frame), cmap="coolwarm", vmin=-MZ, vmax=MZ)
    c2 = ax2.pcolormesh(x, y, u.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c3 = ax3.pcolormesh(x, y, v.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)

    return [c1, c2, c3]

ani = FuncAnimation(fig=fig, func=__update, frames=range(len(ds.time)), interval=100)

ani.save('animations/animation_truth.mp4')
print('done')
