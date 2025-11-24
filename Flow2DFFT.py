import numpy as np
import xarray as xr
import scipy.fft as fft
import time


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
        self.nt = int(self.T // self.dt) + 1
        self.nx = np.shape(zeta0)[1]
        self.ny = np.shape(zeta0)[0]
        self.Lx = self.dx * (self.nx - 1)
        self.Ly = self.dy * (self.ny - 1)
        print(self.nt)
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
        return (np.roll(f, shift=-1, axis=-1) - np.roll(f, shift=1, axis=-1)) / (2*dx)

    def __partialy(self, f):
        """
        Compute the partial derivative of f(x,y) w.r.t y using centered differences. Assumes periodic boundaries
        """
        dy = self.dy

        return (np.roll(f, shift=-1, axis=-2) - np.roll(f, shift=1, axis=-2)) / (2*dy)
    
    def __laplacian(self, f):
        """
        Compute the laplacian of f(x,y) with centered differences. Assumes periodic boundaries
        """
        dx = self.dx
        dy = self.dy

        d2fdx2 = (np.roll(f, -1, axis=-1) - 2*f + np.roll(f, 1, axis=-1)) / dx**2
        d2fdy2 = (np.roll(f, -1, axis=-2) - 2*f + np.roll(f, 1, axis=-2)) / dy**2

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
        Time step zeta accounting for advection and diffusion 
        """
        arr = np.array(zeta) # retain the boundary conditions

        # advection step
        arr = zeta - self.dt * (u * self.__partialx(zeta) + v * self.__partialy(zeta)) 
        
        # diffusiion term
        arr += self.dt * self.kappa * self.__laplacian(zeta)

        return arr
    
    def __get_streamfunction(self, zeta):
        """
        Get stream function from solving the laplace equation Laplacian(psi) = zeta

        Uses FFT method
        """
        nx = self.nx
        ny = self.ny

        zeta_transform = fft.rfft2(zeta, (ny, nx), axes=(-2,-1))

        psi_transform = - zeta_transform / self.K2 
        psi = fft.irfft2(psi_transform, s=(ny, nx), axes=(-2,-1))

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
        """
        Integrate the 2D barotropic flow equations
        """

        # rename vars
        nt, nx, ny = self.nt, self.nx, self.ny
        history_interval = self.history_interval
        nsave = int(nt//history_interval)+1 # number of timesteps to save
        print(nt//history_interval, nsave)

        # initialize arrays to save
        t, x, y = np.linspace(0, self.T, nsave), np.linspace(0, self.Lx, nx), np.linspace(0, self.Ly, ny)
        u, v, zeta = np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx))
        
        # set initial condition for voriticity
        zeta[0] = self.zeta0

        # compute and set initial state for winds
        psi = self.__get_streamfunction(zeta[0])
        u[0], v[0] = self.__get_wind(psi)

        # printout 
        print('nstep = ', 0, 
              ',     total energy = ', self.__total_energy(u[0], v[0]),
              ',     max divergence = ', self.__divergence(u[0], v[0]))
        
        # copy ics to 2D arrays for integration
        uprev, vprev, zetaprev = np.array(u[0]), np.array(v[0]), np.array(zeta[0])

        # main loop
        start = time.time()
        for k in range(1, nt + 1):
            # integrate one time step
            zetacurr, ucurr, vcurr = self.__update(zetaprev, uprev, vprev)

            # save output
            if k % history_interval == 0:
                # printout
                end = time.time()
                print('nstep = ', k, 
                      ',     total energy = ', self.__total_energy(ucurr, vcurr),
                      ',     max divergence = ', self.__divergence(ucurr, vcurr),
                      ',     seconds elapsed = ', end - start)
                index = int(k//history_interval)
                zeta[index] = zetacurr
                u[index] = ucurr
                v[index] = vcurr
                start = time.time()

            # copy current state to be the next previous state
            uprev, vprev, zetaprev = np.array(ucurr), np.array(vcurr), np.array(zetacurr)
            

        # dataset to output
        ds = xr.Dataset(data_vars={
                                    'u':(['time', 'y', 'x'], u),
                                    'v':(['time', 'y', 'x'], v),
                                    'vorticity':(['time', 'y', 'x'], zeta)
                                },
                        coords={
                                'time':('time', t),
                                'x':('x', x),
                                'y':('y', y)
                            })

        return ds
    
    def __forward_model(self, obsmask):
        """
        Given an observation mask, obsmask, generate the linear forward model with dimensions nobs by ny*nx
        """
        nobs = np.sum(obsmask)
        hh = np.zeros((nobs, self.ny * self.nx))
        obsmask_flat = np.reshape(obsmask, (self.ny*self.nx))
        
        j = 0
        for i in range(len(obsmask_flat)):
            if obsmask_flat[i] == 1:
                hh[j,i] = 1
                j += 1
                
        return hh
    
    def enkf(self, nens, bscale, rscale, tobs, obsmask, obsstart=0, ens_to_save=[0]):
        """
        Integrate the 2D barotropic flow equations with data assimilation using an 
        Ensemble Kalman Filter
            - nens: number of ensemble members
            - bscale: ensemble of initial conditions calculated using a normal ditribution at 
                      each gridcell, centered on the zeta0, with spread bscale * |zeta0|
            - rscale: ensemble of measurements calculated using a normal ditribution at 
                      each gridcell, centered on the zeta0, with spread rscale * |zeta0|
            - tobs: interval between observations, in number of time steps. For tobs == -1, forecast is freerunning
            - obsmask: mask indicating which gridcells are observed
            - obsstart: the timestep of the first observation. If 0, obs start at obsstart=tobs
            - ens_to_save: list of ensemble members to save
        """
        # maybe add a freerunning forecast to compare to

        # rename vars
        nt, nx, ny = self.nt, self.nx, self.ny
        history_interval = self.history_interval
        nsave = int(nt//history_interval)+1 # number of time steps to save
        nens_to_save = len(ens_to_save) # number of ensemble members to save

 
        # initialize arrays to save
        t, x, y, ensemble_id = np.linspace(0, self.T, nsave), np.linspace(0, self.Lx, nx), np.linspace(0, self.Ly, ny),  np.array(ens_to_save)

        # setup truth arrays
        u, v, zeta = np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx))
        
        # setup ensemble forecast arrays
        uens, vens, zetaens = np.zeros((nsave, nens_to_save, ny, nx)), np.zeros((nsave, nens_to_save, ny, nx)), np.zeros((nsave, nens_to_save, ny, nx))

        # setup arrays for forecast means 
        umean, vmean, zetamean = np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx)), np.zeros((nsave, ny, nx))

        # setup array to store ensemble variance 
        zeta_var = np.zeros((nsave, ny, nx))

        # set initial condition for true vorticity
        zeta[0] = self.zeta0

        # compute initial conditon for true winds
        psi = self.__get_streamfunction(zeta[0])
        u[0], v[0] = self.__get_wind(psi)

        # save ensemble forecast initial condition 
        zetaa = np.random.normal(loc=self.zeta0, scale=bscale * np.where(np.abs(self.zeta0)>1,np.abs(self.zeta0),1), size=(nens,ny,nx))

        # flatten forecasts to compute background error covariance matrix
        xf = np.reshape(zetaa, (nens, ny*nx))
        xfc = xf-np.mean(xf, axis=0)
        bb = xfc.T @ xfc / xfc.shape[0]
        zeta_var[0] = np.reshape(np.diag(bb), (ny, nx)) # save in initial variance
        
        # compute initial conditon for forecast winds
        psia = self.__get_streamfunction(zetaa)
        ua, va = self.__get_wind(psia)
        
        # setup observing network
        hh = self.__forward_model(obsmask)

        # printout 
        print('nstep = ', 0, 
              ',     total energy = ', self.__total_energy(u[0], v[0]),
              ',     max divergence = ', self.__divergence(u[0], v[0]),
              ',     rms = ', np.sqrt(np.trace(bb)))
        

        # save ics for ensemble members we want to save
        uens[0], vens[0], zetaens[0] = np.array(ua[ens_to_save,:,:]), np.array(va[ens_to_save,:,:]), np.array(zetaa[ens_to_save,:,:])
        
        # save ics for means 
        umean[0], vmean[0], zetamean[0] = np.array(zetaa.mean(axis=0)), np.array(ua.mean(axis=0)), np.array(va.mean(axis=0)) 

        # copy ics to 2D arrays for integration
        uprev, vprev, zetaprev = np.array(u[0]), np.array(v[0]), np.array(zeta[0])
        # main loop
        start = time.time()
        for k in range(1, nt + 1):
            # evolve the true state
            zetacurr, ucurr, vcurr = self.__update(zetaprev, uprev, vprev)

            # evolve the forecast
            zetaf, uf, vf = self.__update(zetaa, ua, va)

            if (tobs != -1) and (k-obsstart) % tobs == 0: 
                # flatten truth and forecast into a vector
                xt = np.reshape(zetacurr, (ny*nx))
                xf = np.reshape(zetaf, (nens, ny*nx))
                
                # compute background covariance
                xfc = xf-np.mean(xf, axis=0)
                bb = xfc.T @ xfc / xfc.shape[0]

                # setup observations array
                xo = np.resize((hh @ xt.T).T, (nens, hh.shape[0]))
                
                # setup observation errors
                nu = np.random.normal(loc=0, scale=rscale * np.where(np.abs(xo)>1,np.abs(xo),1), size=(nens,hh.shape[0]))
                nu -= np.mean(nu, axis=0) # nu should not be allowed to be 0
                xo += nu
                rr = nu.T @ nu / nu.shape[0]

                print('nstep = ', k, ',     assimilating obs . . . rms of obs =', np.sqrt(np.trace(rr)))
                
                # get kalman gain 
                rhs = bb @ hh.T
                
                ss = hh @ bb @ hh.T + rr
                kk = np.linalg.solve(ss.T, rhs.T).T

                # do analysis 
                xa = xf + (kk @ (xo.T - hh @ xf.T)).T 
           
                # reshape data 
                zetaa = np.reshape(xa, (nens, ny, nx))

                # update winds based on assimilated data
                psia = self.__get_streamfunction(zetaa)
                ua, va = self.__get_wind(psia)

            else:
                # update fields
                xa = np.reshape(zetaf, (nens, ny*nx))
                zetaa = np.array(zetaf)
                ua = np.array(uf)
                va = np.array(vf)

            # save output
            if k % history_interval == 0:
                
                # compute new covariance for trace
                xac = xa-np.mean(xa, axis=0)
                pa = xac.T @ xac / xac.shape[0]
                
                # printout
                end = time.time()
                print('nstep = ', k, 
                      ',     total energy = ', self.__total_energy(ucurr, vcurr),
                      ',     max divergence = ', self.__divergence(ucurr, vcurr),
                      ',     rms = ', np.sqrt(np.trace(pa)),
                      ',     seconds elapsed = ', end - start)
                start = time.time()
                
                # save data
                index = int(k//history_interval)
                zeta[index] = zetacurr
                u[index] = ucurr
                v[index] = vcurr
                zetaens[index] = zetaa[ens_to_save,:,:]
                uens[index] = ua[ens_to_save,:,:]
                vens[index] = va[ens_to_save,:,:]

                # compute and save variance
                zeta_var[index] = np.reshape(np.diag(pa), (ny, nx))

                # compute and save means
                zetamean[index], ua[index], va[index] = np.array(zetaa.mean(axis=0)), np.array(ua.mean(axis=0)), np.array(va.mean(axis=0)) 

            # deep copy curr data to be the next previous data
            uprev, vprev, zetaprev = np.array(ucurr), np.array(vcurr), np.array(zetacurr)
            
            

        # save data
        ds = xr.Dataset(data_vars={
                                    'u':(['time', 'y', 'x'], u),
                                    'v':(['time', 'y', 'x'], v),
                                    'vorticity':(['time', 'y', 'x'], zeta),
                                    'u_ens':(['time', 'ensemble_id', 'y', 'x'], uens),
                                    'v_ens':(['time', 'ensemble_id', 'y', 'x'], vens),
                                    'vorticity_ens':(['time', 'ensemble_id', 'y', 'x'], zetaens),
                                    'u_mean':(['time', 'y', 'x'], u),
                                    'v_mean':(['time', 'y', 'x'], v),
                                    'vorticity_mean':(['time', 'y', 'x'], zeta),
                                    'vorticity_var':(['time', 'y', 'x'], zeta_var),
                                    'obsmask':(['y', 'x'], obsmask)
                                },
                        coords={
                                'time':('time', t),
                                'ensemble_id':('ensemble_id', ensemble_id),
                                'x':('x', x),
                                'y':('y', y)
                            })
    
        return ds





