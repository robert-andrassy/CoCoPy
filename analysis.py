from __future__ import print_function

import os
import sys
import re
import glob
import mmap
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path

import collections
import operator

class DumpList:
   def __init__(self, run_dir, code):
      if not os.path.isdir(run_dir):
         print("Error, directory does not exist: '{:s}'".format(run_dir))
         return None
    
      self.run_dir = run_dir
    
      file_ext = {'FLASH':'',
                  'MUSIC':'.h5',
                  'PPMSTAR':'.aaa',
                  'PROMPI':'.bindata',
                  'SLH':'.slh'}
    
      cn = code.strip().upper()
      if cn not in file_ext.keys():
         print("Error, code '{:s}' not implemented.".format(code))
         self.dump_list = []
         return
     
      # Regular expressions that should match an integer dump index in the file
      # name. The index is assumed to be an increasing function of time, i.e. it
      # could be the time step number. Actual dump numbers will be assigned to
      # the file names after sorting the matches by the dump index.
      rexp = {'FLASH':'[0-9]{4}',
              'MUSIC':'[0-9]{9}',
              'PPMSTAR':'[0-9]{4}',
              'PROMPI':'[0-9]{5}',
              'SLH':'[0-9]+'}
      crexp = re.compile(rexp[code])

      # Get a list of all files in run_dir.
      fl = [f for f in os.listdir(run_dir) if
            (os.path.isfile(os.path.join(run_dir, f)) and
            f.endswith(file_ext[code]))] 

      # Get a list of all files matching the code's rexp and convert the
      # matching strings to integers.
      ml = [[crexp.search(fn), fn] for fn in fl]
      ml2 = [[int(m.group(0)), fn] for (m, fn) in ml if m is not None]

      # Some codes write dump zero at t = 0, others do not.
      ndump0 = {'FLASH':0,
                'MUSIC':0,
                'PPMSTAR':1,
                'PROMPI':0,
                'SLH':0}[code]

      dl = sorted(ml2, key=operator.itemgetter(0))
      self.dumps = {(ndump0 + ndump):os.path.join(run_dir, dl[ndump][1])
                    for ndump in range(len(dl))}

class CCUnits:
   def __init__(self):
      self.length = 4e8
      self.velocity = 5.050342e8
      self.density  = 1.820940e+06
      self.temperature = 3.401423e+09
      self.volume = self.length**3
      self.mass = self.density*self.volume
      self.time = self.length/self.velocity
      self.acceleration = self.velocity/self.time
      self.pressure = self.density*self.velocity**2
      self.energy = self.mass*self.acceleration*self.length

def get_haxes(ndim, vaxis):
   if vaxis >= ndim:
      print('Error: vaxis = {:d} does not exist with ndim = {:d}.'.\
            format(vaxis, ndim))
      return None

   if ndim not in (2, 3):
      print('Error: ndim = {:d} not supported.'.format(ndim))
      return None

   haxes = list(range(ndim))
   haxes.remove(vaxis)

   return tuple(haxes)

def get_reshape_vec(res, vaxis):
   ndim = len(res)
   if vaxis >= ndim:
      print('Error: vaxis = {:d} does not exist with ndim = {:d}.'.\
            format(vaxis, ndim))
      return None

   reshape_vec= [1,]*ndim
   reshape_vec[vaxis] = res[vaxis]

   return tuple(reshape_vec)

class Analysis(object):

    def __init__(self, **kwargs):
        self.gamma = 5./3.
        self.units = CCUnits()

    @property
    def cspeed(self):
        return np.sqrt(self.gamma * self.pres / self.rho)

    @property
    def mach(self):
        return self.vel / self.cspeed

    @property
    def vol(self):
        return np.ones(self.rho.shape)*np.prod(self.dx)

    @property
    def mass(self):
        return self.vol * self.rho

    @property
    def vorticity(self):
        vort = curl(self.vel, 1, *self.dx)
        return np.sum(vort**2, axis=0)**0.5

    @property
    def velocity_divergence(self):
        return divergence(self.vel, 1, *self.dx)

    @property
    def absvel(self):
        return np.sqrt((self.vel**2).sum(axis=0))

    @property
    def ekin(self):
        return 0.5 * self.rho * (self.vel**2).sum(axis=0)

    @property
    def A(self):
        return self.pres/self.rho**self.gamma

    @property
    def enthalpy(self):
        return self.eps + self.pres

    def enthalpy_flux(self,vaxis):
        enthalpy = self.enthalpy
        vol = self.vol
        vel = self.vel[vaxis]
        mass = self.mass
        enthalpy_flux = self.get_mean(enthalpy*vel, vol, vaxis) - \
                        self.get_mean(enthalpy, vol, vaxis)*\
                        self.get_mean(vel, mass, vaxis)
        return enthalpy_flux

    def kinetic_energy_flux(self,vaxis):
        ek = self.ekin
        vol = self.vol
        vel = self.vel[vaxis]
        mass = self.mass
        ek_flux = self.get_mean(ek*vel, vol, vaxis) - \
                  self.get_mean(ek, vol, vaxis)*\
                  self.get_mean(vel, mass, vaxis) 
        return ek_flux

    def filling_factor_downflow(self, vaxis):
        vel = self.vel[vaxis]
        mass = self.mass
        mean_vel = self.get_mean(vel, mass, vaxis)
        haxes = get_haxes(len(vel.shape), vaxis)
        ffd = np.mean(vel - mean_vel < 0., axis=haxes)
        return ffd

    def get_mean(self, quantity, weight, vaxis):
       haxes = get_haxes(len(quantity.shape), vaxis)
       sum_weight = np.sum(weight, axis=haxes)
       mean = np.sum(quantity*weight, axis=haxes)/sum_weight

       return mean

    def get_stdev(self, quantity, weight, vaxis):
       haxes = get_haxes(len(quantity.shape), vaxis)
       sum_weight = np.sum(weight, axis=haxes)
       mean = np.sum(quantity*weight, axis=haxes)/sum_weight

       reshape_vec = get_reshape_vec(quantity.shape, vaxis)
       tmp = np.reshape(mean, reshape_vec)
       stdev = (np.sum((quantity - tmp)**2*weight, axis=haxes)/sum_weight)**0.5

       return stdev

    def write_Rprof(self, dump_num, rprof_name):
       vaxis = 1
       haxes = get_haxes(3, vaxis)

       cell_volume = self.vol
       cell_mass = self.rho*self.vol

       vars = collections.OrderedDict()
       vars['RHO'] = {'expr':lambda g: g.rho, 'weighting':'v', 'stats':'full'}
       vars['P'] = {'expr':lambda g: g.pres, 'weighting':'v', 'stats':'full'}
       vars['TEMP'] = {'expr':lambda g: g.temp, 'weighting':'v', 'stats':'full'}
       vars['A'] = {'expr':lambda g: g.A, 'weighting':'v', 'stats':'full'}
       vars['X1'] = {'expr':lambda g: g.xnuc[1], 'weighting':'m', 'stats':'full'}
       vars['V'] = {'expr':lambda g: g.absvel, 'weighting':'m', 'stats':'full'}
       vars['VX'] = {'expr':lambda g: g.vel[0], 'weighting':'m', 'stats':'full'}
       vars['VY'] = {'expr':lambda g: g.vel[1], 'weighting':'m', 'stats':'full'}
       vars['VZ'] = {'expr':lambda g: g.vel[2], 'weighting':'m', 'stats':'full'}
       vars['|VY|'] = {'expr':lambda g: np.abs(g.vel[1]), 'weighting':'m', 'stats':'mean'}
       vars['VXZ'] = {'expr':lambda g: (g.vel[haxes[0]]**2 + g.vel[haxes[1]]**2)**0.5, 'weighting':'m', 'stats':'mean'}
       vars['VORT'] = {'expr':lambda g: np.abs(g.vorticity), 'weighting':'m', 'stats':'mean'}

       nx = self.rho.shape[vaxis]
       nbins = nx

       time = self.time
       ir = 1 + np.arange(nbins)
       y = np.mean(self.coords[vaxis], axis=haxes)
       ncols = 1*sum(1 for val in vars.values() if val['stats'] == 'mean') + \
               4*sum(1 for val in vars.values() if val['stats'] == 'full')
       ncols+=4 # FK, FH, FFD, and DISS are computed separately below
       col_names = []
       data_table = np.zeros((nbins, ncols))

       j = 0
       for i, v in enumerate(vars.keys()):
          expr = vars[v]['expr']
          stats = vars[v]['stats']
          weighting=vars[v]['weighting']
          if stats not in ('mean', 'full'):
             print("ERROR: 'stats' must be 'mean' or 'full'.")
             return None

          if weighting == 'v':
             weight = cell_volume
          elif weighting == 'm':
             weight = cell_mass
          else:
             print("ERROR: 'weighting' must be 'v' or 'm'.")
             return None

          # We always want the mean.
          col_names.append(v)
          mean = self.get_mean(expr(self), weight, vaxis)
          data_table[:,j] = mean
          j+=1

          if stats == 'full':
             min = np.min(expr(self), axis=haxes)
             col_names.append('MIN_'+v)
             data_table[:,j] = min
             j+=1

             max = np.max(expr(self), axis=haxes)
             col_names.append('MAX_'+v)
             data_table[:,j] = max
             j+=1

             stdev = self.get_stdev(expr(self), weight, vaxis)
             col_names.append('STDEV_'+v)
             data_table[:,j] = stdev
             j+=1

       col_names.append('FK')
       data_table[:,j] = self.kinetic_energy_flux(vaxis)
       j+=1

       col_names.append('FH')
       data_table[:,j] = self.enthalpy_flux(vaxis)
       j+=1

       col_names.append('FFD')
       data_table[:,j] = self.filling_factor_downflow(vaxis)
       j+=1

       # Dissipation rate to be implemented.
       diss = np.zeros(nbins)
       col_names.append('DISS')
       data_table[:,j] = diss
       j+=1

       try:
          fout = open(rprof_name, "w")
          fout.write('DUMP {:4d}, t = {:.8e}\n'.format(dump_num, time))
          fout.write('Nx = {:d}\n\n\n'.format(nx))

          cols_per_table = 8
          ntables = int(np.ceil(ncols/cols_per_table))
          for i in range(ntables):
             if i < ntables - 1:
                cols_this_table = cols_per_table
             else:
                cols_this_table = ncols - i*cols_per_table
             table = np.zeros((nbins, 2+cols_this_table))
             table[:,0] = ir
             table[:,1] = y
             idx1 = i*cols_per_table
             idx2 = idx1 + cols_this_table

             fmt = 'IR      ' + '{:18s}'*(1+cols_this_table) + '\n\n'
             header = fmt.format('Y', *col_names[idx1:idx2])
             fout.write(header)

             table[:,2:(2+cols_this_table)] = data_table[:,idx1:idx2]
             table = np.flip(table, axis=0)

             fmt = ('%4d',) + ('% .8e',)*(1+cols_this_table)
             np.savetxt(fout, table, fmt=fmt, delimiter='   ')
             fout.write('\n\n')

          fout.close()
       except EnvironmentError:
          print('An environment error has occured!')

    def write_spec(self, dump_num, spec_name):
       vaxis = 1
       haxes = get_haxes(3, vaxis)
       y = np.mean(self.coords[vaxis], axis=haxes)
 
       try:
           fout = open(spec_name, "w")
           fout.write('DUMP {:4d}, t = {:.8e}\n'.format(dump_num, self.time))
 
           nx, ny, nz = self.rho.shape
           for y0 in (1.7, 2.7):
               idx0 = np.argmin(np.abs(y - y0))
               amp = np.array([np.fft.fftshift(np.fft.fft2(self.vel[i, :, idx0, :]))/(nx*nz) \
                      for i in range(3)])
               amp = 0.5*np.sum(np.abs(amp)**2, axis=0)
               kx = np.fft.fftshift(np.fft.fftfreq(nx, 1./nx))
               kxmax = kx[-1]
               kx = np.transpose(np.tile(kx, (nz,1)))
               kz = np.fft.fftshift(np.fft.fftfreq(nz, 1./nz))
               kzmax = kz[-1]
               kz = np.tile(kz, (nx,1))
               kk = (kx**2 + kz**2)**0.5
 
               nbins = 1 + int(np.min((kxmax, kzmax)))
               bin_edges = np.linspace(-0.5, float(nbins) - 0.5, nbins+1)
               tot_amp, bin_edges = np.histogram(kk, weights=amp, bins=bin_edges)
               k = 0.5*(bin_edges[:-1] + bin_edges[1:])

               header = '\n\nk       H2\n\n'
               fout.write(header)
               arr = np.transpose(np.array((k, tot_amp)))
               fmt = ('%4d', '%.8e')
               np.savetxt(fout, arr, fmt=fmt, delimiter='   ')
 
           fout.close()
       except EnvironmentError:
           print('An environment error has occured!')

    def write_slices(self, dump_num, run_id):
        vaxis = 1
        haxes = get_haxes(3, vaxis)
        x = self.coords[0][:, 0, 0]
        y = self.coords[1][0, :, 0]
        z = self.coords[2][0, 0, :]

        vars = {} 
        vars['VX'] = self.vel[0]
        vars['VY'] = self.vel[1]
        vars['VZ'] = self.vel[2]
        vars['RHO'] = self.rho
        vars['A'] = self.A
        vars['X1'] = self.xnuc[1]
        vars['VORT'] = self.vorticity
        vars['DIVV'] = self.velocity_divergence

        nx, ny, nz = self.rho.shape
        x0 = 0.
        y0 = 1.7
        y1 = 2.7
        z0 = 0.
        ix0 = np.argmin(np.abs(x - x0))
        iy0 = np.argmin(np.abs(y - y0))
        iy1 = np.argmin(np.abs(y - y1))
        iz0 = np.argmin(np.abs(z - z0))
        slices = {'X-{:.3f}'.format(x0):np.s_[ix0, :, :],
                  'Y-{:.3f}'.format(y0):np.s_[:, iy0, :],
                  'Y-{:.3f}'.format(y1):np.s_[:, iy1, :],
                  'Z-{:.3f}'.format(z0):np.s_[:, :, iz0]}

        dir_name = './slices/{:04d}/'.format(dump_num)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        for var_name in vars.keys():
            for sl_name in slices.keys():
                data = vars[var_name][slices[sl_name]].astype(np.float32)
                file_name = '{:s}-{:04d}-{:s}-{:s}.npy'.format(\
                            run_id, dump_num, var_name, sl_name)
                np.save(dir_name+file_name, data, allow_pickle=False)

class SLH(Analysis):
    def __init__(self, filename, **kwargs):
        import slhoutput
        super(SLH, self).__init__(**kwargs)
        g = slhoutput.slhgrid(filename, mode='f')
        self.rho = np.copy(g.rho())
        self.pres = np.copy(g.pres())
        self.temp = np.copy(g.temp())
        self.vel = np.copy(g.vel())
        self.eps = self.pres/(self.gamma - 1.)
        self.xnuc = np.copy(g.xnuc())
        self.coords = np.copy(g.coords())

        self.dx = g.geometry.dx
        self.time = g.time

        if g.qref.length == 1.0:
            self.rho /= self.units.density
            self.pres /= self.units.pressure
            self.temp /= self.units.temperature
            self.vel /= self.units.velocity
            self.eps /= (self.units.energy/self.units.volume)
            self.coords /= self.units.length
            self.dx /= self.units.length
            self.time /= self.units.time

class Flash(Analysis):
    def __init__(self,filename,**kwargs):
        import yt
        from yt.frontends.flash.data_structures import FLASHDataset

        #conversion factors from cgs to code units
        rhofac = 1.820940e6
        pfac = 4.644481e+23
        tempfac = 3.401423e9
        efac = 2.972468e+49
        rfac = 4e8
        volfac = rfac**3
        tfac = 0.7920256
        velfac = rfac/tfac
        
        if isinstance(filename,FLASHDataset):
            d = filename
        else:
            d = yt.load(filename)
        shape = [complex(0,d.domain_dimensions[i]) for i in range(3)]
        grid = d.r[::shape[0],::shape[1],::shape[2]]

        self.rho = grid['dens'].to_ndarray() / rhofac
        self.pres = grid['pres'].to_ndarray() / pfac
        self.temp = grid['temp'].to_ndarray() / tempfac
        self.vel = np.array([grid['velx'].to_ndarray(),grid['vely'].to_ndarray(),grid['velz'].to_ndarray()])/velfac
        #FLASH stores the internal energy in units of energy/mass
        #For the analysis we want energy/volume 
        eintvol = grid['eint'] * grid['dens']
        #non dimensionalization requires volfac/efac
        self.eps =  eintvol.to_ndarray() * volfac/efac
        self.xnuc = np.array([grid['conv'].to_ndarray(),grid['stab'].to_ndarray()])
        self.coords = np.array([grid['x'].to_ndarray()-2*rfac,grid['y'].to_ndarray(),grid['z'].to_ndarray()-2*rfac])/rfac

        self.dx = np.array([grid['dx'].to_ndarray()[0,0,0],grid['dy'].to_ndarray()[0,0,0],grid['dz'].to_ndarray()[0,0,0]])/rfac
        self.time = d.current_time.to_ndarray() / tfac

        super(Flash,self).__init__(**kwargs)

class PROMPI(Analysis):
    def __init__(self, filename, **kwargs):
        import UTILS.PROMPI.PROMPI_data as prd

        # conversion factors from cgs to code units
        rhofac = 1.820940e6
        pfac = 4.644481e+23
        tempfac = 3.401423e9
        efac = 2.972468e+49
        rfac = 4e8
        tfac = 0.7920256
        velfac = rfac / tfac

        dat = ['density', 'velx', 'vely', 'velz', 'energy',
               'press', 'temp', 'gam1', 'gam2', 'enuc1', 'enuc2',
               '0001','0002']

        block = prd.PROMPI_bindata(filename, dat)

        # PROMPI data vertical direction is X #
        # hence swap X for Y axis everywhere  #
        dens =  np.swapaxes(block.datadict['density'],0,1)
        self.rho = dens / rhofac
        self.pres = np.swapaxes(block.datadict['press'],0,1) / pfac
        self.temp = np.swapaxes(block.datadict['temp'],0,1) / tempfac

        velx = np.swapaxes(block.datadict['vely'],0,1)
        vely = np.swapaxes(block.datadict['velx'],0,1)
        velz = np.swapaxes(block.datadict['velz'],0,1)
        etot = np.swapaxes(block.datadict['energy'],0,1)

        ekin = 0.5*(velx**2.+vely**2.+velz**2.)
        eint = etot - ekin

        self.vel = np.array([velx,vely,velz]) / velfac
        self.eps = np.array(dens * eint) / (efac/(rfac**3.))
        self.xnuc = np.array([np.swapaxes(block.datadict['0001'],0,1), np.swapaxes(block.datadict['0002'],0,1)])

        nx = np.array(block.datadict['qqx'])
        ny = np.array(block.datadict['qqy'])
        nz = np.array(block.datadict['qqz'])

        xzn0 = np.array(block.datadict['xzn0'])
        yzn0 = np.array(block.datadict['yzn0'])
        zzn0 = np.array(block.datadict['zzn0'])

        # hope this is right #
        gridx = np.empty((nx, ny, nz))
        for k, v in enumerate(yzn0/rfac - 1.): gridx[k, :, :] = v

        gridy = np.empty((nx, ny, nz))
        for k, v in enumerate(xzn0/rfac): gridy[:, k, :] = v

        gridz = np.empty((nx, ny, nz))
        for k, v in enumerate(zzn0/rfac - 1.): gridz[:, :, k] = v

        self.coords = np.array([gridx,gridy,gridz])

        deltax = np.asarray(block.datadict['yznr']) - np.asarray(block.datadict['yznl'])
        deltay = np.asarray(block.datadict['xznr']) - np.asarray(block.datadict['xznl'])
        deltaz = np.asarray(block.datadict['zznr']) - np.asarray(block.datadict['yznl'])

        dx = np.empty((nx, ny, nz))
        for k, v in enumerate(deltax/rfac): dx[k, :, :] = v

        dy = np.empty((nx, ny, nz))
        for k, v in enumerate(deltay/rfac): dy[:, k, :] = v

        dz = np.empty((nx, ny, nz))
        for k, v in enumerate(deltaz/rfac): dz[:, :, k] = v

        self.dx = np.array([dx[0,0,0], dy[0,0,0], dz[0,0,0]])

        self.time = block.datadict['time'] / tfac

        super(PROMPI, self).__init__(**kwargs)

class MUSIC(Analysis):
    def __init__(self, filename, ndim=3, num_scalars=1, **kwargs):
        import h5py

        f = h5py.File(filename, mode='r')
        #relevant parameters
        self.gamma = f['parameters/gamma'][0]
        self.mu0 = f['parameters/mu0'][0]
        self.mu1 = f['parameters/mu1'][0]
        self.Rgasconst = f['parameters/Rgasconst'][0]
        self.time = f['parameters/time'][0]

        fields = f['fields']
        # MUSIC data vertical direction is 0
        # hence swap 0 for 1 axis everywhere
        self.rho = np.swapaxes(fields['rho'][...], 0, 1)
        self.eps = np.swapaxes(fields['rho'][...]*fields['e_int'][...], 0, 1)

        vel = [np.swapaxes(fields['vel_{}'.format(i)][...], 0, 1) for i in (2,1)]
        if ndim == 3:
            vel.append(np.swapaxes(fields['vel_3'][...], 0, 1))
        self.vel = np.array(vel)

        #MUSIC evolves the mu0 scalar
        if 'scalar_2' in fields:
            scal0 = np.swapaxes(fields['scalar_1'][...], 0, 1)
            scal1 = np.swapaxes(fields['scalar_2'][...], 0, 1)
        else:
            scal0 = np.swapaxes(fields['scalar_1'][...], 0, 1)
            scal1 = 1 - scal0
        self.xnuc = np.stack((scal0, scal1), axis=0)
        #re-order vertical direction form index 0 to index 1 and
        #reset the horizontal extent from [0,2](MUSIC) to [-1,1](SLH)
        ax = [fields['y'][...] - 1, fields['x'][...]]
        if ndim == 3:
            ax.append(fields['z'][...] - 1)

        self.coords = np.array(np.meshgrid(*ax, indexing='ij'))
        self.dx = [np.diff(ax[0]).mean(),
                   np.diff(ax[1]).mean()]
        if ndim == 3:
            self.dx.append(np.diff(ax[2]).mean())
        self.dx = np.array(self.dx)

    @property
    def pres(self):
        return (self.gamma -1)*self.eps

    def mu_ave(self):
        return self.mu1*self.xnuc[1]+self.mu0*self.xnuc[0]

    @property
    def temp(self):
        return (self.gamma - 1)*self.eps/self.rho*self.mu_ave()/self.Rgasconst

class PPMSTAR(Analysis):
    def __init__(self, filename):
        import PPMstardata
        # code units == problem units

        dump = PPMstardata.fdmp(filename)
        self.time=dump.time
        self.gamma = dump.gamma
        self.mu0 = dump.airmu
        self.mu1 = dump.cldmu
        self.Rgasconst = dump.Rgasconst
        self.rho = dump.rho0 + dump.rho1
        self.pres = dump.p0 + dump.p1
        self.vel=dump.vel
        self.mu=dump.airmu*dump.fvair+dump.cldmu*dump.fv
        ix,iy,iz=dump.rho0.shape
        xnuc=np.zeros((2,ix,iy,iz))
        xnuc[1]=dump.cldmu*dump.fv/self.mu
        xnuc[0]=dump.airmu*dump.fvair/self.mu
        self.xnuc=xnuc
        self.temp=self.pres*self.mu/(self.rho*self.Rgasconst)
        self.dx=[dump.deex,]*3
        self.coords=dump.coords
        self.eps=self.pres/(self.gamma-1.)

def divergence(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return dx[0] + dy[1] + dz[2]

def curl(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return np.array([dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]])
