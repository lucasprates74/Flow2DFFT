import numpy as np
import xarray as xr
import scipy.fft as fft


class Flow2D():
    """
    Solves the equations of motion for 2D barotropic flow assuming periodic boundary conditions
    """

    def __setup_wavenumbers(self, ny, nx, dy, dx):
        """
        Generate meshgrid of wavenumbers used to solve the Laplace equation
        """
        ell = 2 * np.pi * fft.fftfreq(ny, d=dy)
        kay = 2 * np.pi * fft.rfftfreq(nx, d=dx)
        K, L = np.meshgrid(kay, ell, indexing='xy')
        K2=(K**2+L**2)

        # get rid of K=0 to prevent division by zero later on
        K2[0,0] = (K2[1,0] + K[0,1] + K[1,1])/3

        return K2

    def __init__(self, zeta0, dt, dx, dy, T, history_interval, kappa):
        """
        - zeta0 is the doubly periodic vorticity initial condition
        - dt is the time-step in seconds
        - dx is the resolution in the x direction in meters
        - dy is the resolution in the y direction in meters
        - T is the length of the simulation in seconds
        - history_interval is the interval you wish to output data at, in number of timestamps
        - kappa is the eddy viscosity
        """

        # initial condition 
        self.zeta0 = zeta0

        # resolution 
        self.dt = dt
        self.dx = dx
        self.dy = dy
        
        # grid configuration
        self.T = T
        self.nt = int(self.T // self.dt)
        self.nx = np.shape(zeta0)[1]
        self.ny = np.shape(zeta0)[0]
        self.Lx = self.dx * (self.nx - 1)
        self.Ly = self.dy * (self.ny - 1)

        # history interval
        self.history_interval = history_interval

        # diffusivity
        self.kappa = kappa
        
        # wavenumber array for FFT
        self.K2 = self.__setup_wavenumbers(self.ny, self.nx, self.dy, self.dx)
        



    def __partialx(self, f):
        """
        Compute the partial derivative of f(x,y) w.r.t. x using centered differences. Assumes periodic boundaries
        """
        dx = self.dx
        return (np.roll(f, shift=-1, axis=1) - np.roll(f, shift=1, axis=1)) / (2*dx)

    def __partialy(self, f):
        """
        Compute the partial derivative of f(x,y) w.r.t y using centered differences. Assumes periodic boundaries
        """
        dy = self.dy

        return (np.roll(f, shift=-1, axis=0) - np.roll(f, shift=1, axis=0)) / (2*dy)
    
    def __laplacian(self, f):
        """
        Compute the laplacian of f(x,y) with centered differences. Assumes periodic boundaries
        """
        dx = self.dx
        dy = self.dy

        d2fdx2 = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dx**2
        d2fdy2 = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dy**2

        return d2fdx2 + d2fdy2
        
    def __divergence(self, fx, fy):
        """
        Compute the maximum divergence of the vector field given by fx(x,y) and fy(x,y)
        """
        return np.max(np.abs(self.__partialx(fx) + self.__partialy(fy)))
    
    def __total_energy(self, u, v):
        """
        Compute the total energy of the velocity field (u, v)
        """
        return np.sum(u**2+v**2)/2
        
    def __update_zeta(self, zeta, u, v):
        """
        Time step zeta accounting for advection and diffusion with operator splitting
        """
        arr = np.array(zeta) # retain the boundary conditions

        # advection step
        arr = zeta - (self.dt/2) * (u * self.__partialx(zeta) + v * self.__partialy(zeta))
        
        # time-split diffusion step
        arr += (self.dt/2) * self.kappa * self.__laplacian(zeta)

        return arr
    
    def __get_streamfunction(self, zeta):
        """
        Get stream function from solving the laplace equation Laplacian(psi) = zeta

        Uses FFT method
        """
        nx = self.nx
        ny = self.ny

        zeta_transform = fft.rfft2(zeta, (ny, nx))

        psi_transform = - zeta_transform / self.K2 
        psi = fft.irfft2(psi_transform, s=(ny, nx))

        return psi
    
    def __get_wind(self, psi):
        """
        Get wind components from the streamfunction
        """
        u = - self.__partialy(psi)
        v = self.__partialx(psi)
        return u, v
    
    def __update(self, zeta, u, v):
            zeta_new = self.__update_zeta(zeta, u, v)
            psi_new = self.__get_streamfunction(zeta_new)
            u_new, v_new = self.__get_wind(psi_new)
            return zeta_new, u_new, v_new
        
    def solve(self):
        # initialize arrays
        nt, nx, ny = self.nt, self.nx, self.ny
        history_interval = self.history_interval
        
        nsave = int(nt//history_interval)+1
        t, x, y = np.linspace(0, self.T, nsave), np.linspace(0, self.Lx, nx), np.linspace(0, self.Ly, ny)
        u, v = np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx))
        psi, zeta =  np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx))


        # # setup meshgrid for solving laplace equation
        # ell = 2 * np.pi * fft.fftfreq(ny, d=self.dy)
        # kay = 2 * np.pi * fft.rfftfreq(nx, d=self.dx)
        # K, L = np.meshgrid(kay, ell, indexing='xy')
        # K2=(K**2+L**2)
        # 
        # # get rid of division by wavenumber zero, set K like this for numerical stability
        # K2[0,0] = (K2[1,0] + K[0,1] + K[1,1])/3
        
        # set initial condition
        zeta[0] = self.zeta0

        # add small random noise to the interior, scaled by local zeta
        zeta[0, 1:-1,1:-1] += zeta[0, 1:-1,1:-1] * np.random.uniform(-1, 1, (ny-2, nx-2)) / 50

        # compute initial conditon for other variables
        psi[0] = self.__get_streamfunction(zeta[0])
        u[0], v[0] = self.__get_wind(psi[0])

        # printout 
        print('nstep = ', 0, 
              ',     total energy = ', self.__total_energy(u[0], v[0]),
              ',     max divergence = ', self.__divergence(u[0], v[0]))
        
        # deep copy initial conditions
        uprev, vprev, zetaprev = np.array(u[0]), np.array(v[0]), np.array(zeta[0])

        for k in range(1, nt):
            zetacurr = self.__update_zeta(zetaprev, uprev, vprev)
            psicurr = self.__get_streamfunction(zetacurr)
            ucurr, vcurr = self.__get_wind(psicurr)

            # save output
            if k % history_interval == 0:
                # printout
                print('nstep = ', k, 
                      ',     total energy = ', self.__total_energy(ucurr, vcurr),
                      ',     max divergence = ', self.__divergence(ucurr, vcurr))
                index = int(k//history_interval)
                zeta[index] = zetacurr
                psi[index] = psicurr
                u[index] = ucurr
                v[index] = vcurr

            # deep copy curr data to be the next previous data
            uprev, vprev, zetaprev = np.array(ucurr), np.array(vcurr), np.array(zetacurr)
            
            


        
        ds = xr.Dataset(data_vars={
                                    'u':(['time', 'y', 'x'], u),
                                    'v':(['time', 'y', 'x'], v),
                                    'vorticity':(['time', 'y', 'x'], zeta),
                                    'sf':(['time', 'y', 'x'], psi)
                                },
                        coords={
                                'time':('time', t),
                                'x':('x', x),
                                'y':('y', y)
                            })
    
        return ds



if __name__=='__main__':
    Lx = 1 # m
    Ly = 0.5 # m
    dx = 0.01 # m 
    dy = 0.01 # m
    dt = 1e-3 # s
    T = 180 # s # 180
    history_interval = 100 # timestep interval for saving data
    kappa = 1e-5 # m2/s

    # initialize initial condition arrays
    nx, ny = int(Lx / dx) + 1, int(Ly / dy) + 1
    zeta0 = np.zeros((ny, nx))

    # build initial condition corresponding to a zonal jet
    U0 = 0.2 # m / s
    Ly = (ny - 1) * dy
    for j in range(ny):
        for i in range(nx):
            zeta0[j,i] = -2* np.pi * U0 / Ly * np.sin(2 *np.pi * j / (ny - 1))


    solver = Flow2D(zeta0, dt, dx, dy, T, history_interval, kappa)
    ds = solver.solve()
    ds.to_netcdf('new_model_output.nc')



