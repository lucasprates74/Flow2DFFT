import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ds = xr.open_dataset('new_model_output.nc')
u = ds.u
v = ds.v
zeta = ds.vorticity
psi = ds.sf
step=10
M=np.ceil(u.max().data)

x, y = ds.x, ds.y
nframes = len(ds.time.values)
print(nframes)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 18))

c1 = ax1.pcolormesh(x, y, zeta.isel(time=0), cmap="coolwarm", vmin=-M*10, vmax=M*10)
c2 = ax2.pcolormesh(x, y, u.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c3 = ax3.pcolormesh(x, y, v.isel(time=0), cmap="coolwarm", vmin=-M, vmax=M)
c4 = ax4.pcolormesh(x, y, psi.isel(time=0), cmap="coolwarm", vmin=-M/10, vmax=M/10)

ax1.set_title('Vorticity')
ax2.set_title('u')
ax3.set_title('v')
ax4.set_title('Streamfunction')
plt.show()


def __update(frame):
    global c1, c2, c3, c4

    # Remove the data
    c1.remove()
    c2.remove()
    c3.remove()
    c4.remove()
    # Draw the new frame
    c1 = ax1.pcolormesh(x, y, zeta.isel(time=frame), cmap="coolwarm", vmin=-M*10, vmax=M*10)
    c2 = ax2.pcolormesh(x, y, u.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c3 = ax3.pcolormesh(x, y, v.isel(time=frame), cmap="coolwarm", vmin=-M, vmax=M)
    c4 = ax4.pcolormesh(x, y, psi.isel(time=frame), cmap="coolwarm", vmin=-M/10, vmax=M/10)

    return [c1, c2, c3, c4]

ani = FuncAnimation(fig=fig, func=__update, frames=range(len(ds.time)), interval=100)#interval=100 is real time

ani.save('animation.mp4')
plt.show()
print('done')
