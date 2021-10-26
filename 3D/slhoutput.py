from __future__ import print_function

import numpy as np
import struct
import time
import sys
import os
import re
import glob
import fcntl
import collections
import shutil
import tempfile
import base64
import subprocess
import types
import mmap
import warnings
import logging
import copy

# We do not need a cryptographic hash, so this should be enough.
from hashlib import md5 as slhhash

from decimal import Decimal

import matplotlib.pyplot as plt
import matplotlib._cm
#import matplotlib.animation
import mpl_toolkits.axes_grid1

import functools

gmean = (1+np.sqrt(5))/2

try:
    from progress.bar import Bar
    use_progressbar = True
except ImportError:
    use_progressbar = False

try:
    from parallel_decorators import vectorize_parallel
except ImportError:
    def vectorize_parallel(f, *args, **kwargs):
        return f(*args, **kwargs)

# module level logger
# The default log level is at WARN. To see more messages use:
# slhoutput.logger.setLevel(slhoutput.logging.INFO)
# or
# slhoutput.logger.setLevel(slhoutput.logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

if sys.maxsize < 2**63 - 1:
    # We don't want mmap on 32-bit systems where virtual memory is limited.
    use_mmap = False
else:
    use_mmap = True

pcolor = plt.pcolormesh

ptypes = {'SS': 'S64', 'SL': 'S1024', 'I4': 'i4', 'I8': 'i8',
          'FS': 'f4', 'FD': 'f8', 'OB': 'i8', 'CS': 'c8', 'CD': 'c16', 'FQ': 'S16' }

iptypes = {'S64': 'SS', 'S1024': 'SL', 'i4': 'I4', 'i8': 'I8',
          'f4': 'FS', 'f8': 'FD', 'c8': 'CS', 'c16': 'CD' }
iptypes = dict([(str(np.dtype(k)), v) for k, v in iptypes.items()])

ibm_quad = False # set this to True to use IBM-style quad precision numbers (two doubles in a row)

plt.register_cmap(name='slh',
                  cmap=matplotlib.colors.LinearSegmentedColormap('slh',
                     matplotlib._cm.cubehelix(h=1.0,r=0.7,gamma=1.2,s=-0.6),
                     plt.rcParams['image.lut']))
plt.rc('image', cmap='slh')

def divergence(f, *varargs):
    D = np.array(np.gradient(f, *varargs)[1:])
    return np.trace(D)

def curl(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return np.array([dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]])

def parse_nonibm_quad(s, return_float=True, endian='='):
    """parse a floating point number in binary128 format as specified by IEEE 754-2008
    Arguments:
        s:            a string containing the binary representation of the floating point number (length: 16)
        return_float: return result as a Python float (default) otherwise Decimal is used
        endian:       can be <, >, or = (default)
    """
    if not isinstance(s, (str, bytes)):
        raise TypeError("argument must be a string")
    if len(s) != 16:
        raise ValueError("length of input must be 16")

    if endian == '=':
        endian = '<' if np.little_endian else '>'

    if endian == '<':
        swap = slice(None,None,-1)
    elif endian == '>':
        swap = slice(None)
    else:
        raise ValueError("unknown endianness")

    s = np.fromstring(s, dtype='u1')
    bits = ''.join([np.binary_repr(i,width=8)[swap] for i in s])
    bits = bits[swap]

    sign = int(bits[0], 2)

    exponent = int(bits[1:16],2)

    significand = int(bits[16:],2)

    if exponent == 0x7fff:
        if significand == 0:
            ret = Decimal(float('inf') * (-1)**sign)
        else:
            ret = Decimal(float('nan'))
    elif exponent == 0:
        ret = Decimal((sign, (1,), 0)) * Decimal(2)**(-16382) * (0 + Decimal(2)**(1-113) * significand)
    else:
        ret = Decimal((sign, (1,), 0)) * Decimal(2)**(exponent-16383) * (1+Decimal(2)**(1-113) * significand)

    if return_float:
        return float(ret)
    else:
        return ret

def parse_quad(x, endian='='):
    global ibm_quad
    if ibm_quad:
        return unpack(endian+"d",x[:8])[0] + unpack(endian+"d",x[8:])[0]
    else:
        return parse_nonibm_quad(x, endian=endian)

def wrap_mproc(i):
    """wrapper function seemingly needed for multiprocessing"""
    global process_one
    return process_one(i)

def fallback(f):
    """tries to get the variable from the current output file and falls back to
    grid_n00000.slh otherwise"""

    @functools.wraps(f)
    def ret(self, *args, **kwargs):
        try:
            return f(self,*args, **kwargs)
        except KeyError as e:
            if self.step ==0:
                raise e
            g = slhgrid(0,prefix=os.path.dirname(self.filename),memmap=self.memmap)
            g._relative_coordinates = self._relative_coordinates
            g.origin = self.origin
            g._origin_old = self._origin_old
            return g.__getattribute__(f.__name__)(*args,**kwargs)
    ret.__name__ = f.__name__
    return ret

def vector(f):
    """add vector specific options like radial projection

    optional keyword arguments:
        - idir: project vector along idir direction
        - orthogonal: if True, return the *norm* of orthogonal component
        - origin: list [x,y,z] defining the origin for the projections,
                  e.g. for cartesian grids not
          centered at (0,0,0). Defaults to [0,0,0].

        - radial (deprecated): same as idir == 0
    """

    if f.__doc__ is not None:
        f.__doc__ = f.__doc__ + '\n\n' + vector.__doc__
    else:
        f.__doc__ = vector.__doc__

    @functools.wraps(f)
    def vector_func(self, *args, **kwargs):
        orthogonal = kwargs.pop('orthogonal', False)
        radial = kwargs.pop('radial', False)
        idir = kwargs.pop('idir', -1)
        along = kwargs.pop('along', False)
        slicer = kwargs.pop('slicer', tuple([slice(None)]))


        # to ensure backward compatibility
        if (radial and idir > 0):
            raise RuntimeError(
                '%s: radial == True and iv set at the same time' % (
                    f.__name__))
        elif radial:
            idir = 0

        # because idir=1 is considered as phi component in 2D worlds (?)
        if (idir >= 1 and self.sdim == 2):
            idir += 1  # +1 to trigger IndexError when idir=2 selected for 2D

        if idir >= 0:
            if self.sdim == 1:
                # We cannot project 1D data. Just leave it as is.
                return f(self, *args, **kwargs)

            # radial
            elif idir == 0:
                return f(self, *args, **kwargs).project(
                    self.coords()[slicer], orthogonal=orthogonal, along=along,
                    slicer=slicer)

            # horizontal (theta)
            elif idir == 1:
                coords = self.coords()[slicer]
                phi = np.arctan2(coords[1], coords[0])
                theta = np.arccos(coords[2]/np.sqrt((coords**2).sum(axis=0)))
                e_theta = np.array([
                    np.cos(theta)*np.cos(phi),
                    np.cos(theta)*np.sin(phi),
                    -np.sin(theta)])
                return f(self, *args, **kwargs).project(
                    e_theta, orthogonal=orthogonal, along=along, slicer=slicer)

            # azimuthal (phi)
            elif idir == 2:
                coords = self.coords()[slicer]
                e_phi = np.zeros_like(coords)
                phi = np.arctan2(coords[1], coords[0])
                e_phi[:2] = -np.sin(phi), np.cos(phi)
                return f(self, *args, **kwargs).project(
                    e_phi, orthogonal=orthogonal, along=along, slicer=slicer)

            elif idir > 2:
                raise IndexError(
                    'idir does not match geometry of dim %d' % self.sdim)
        else:
            if orthogonal:
                raise RuntimeError("orthogonal needs 'idir' keyword.")
            return f(self, *args, **kwargs)
    return vector_func

class slhoutput(collections.Mapping):

    def __init__(self,filename, memmap=use_mmap):
        """opens a slhoutput file
           memmap: Use memory mapping instead of normal read calls"""

        self.sslen = 64
        self.sllen = 1024

        self.filename = os.path.abspath(filename)

        self.fh = open(self.filename, "rb")
        self.memmap = memmap
        if self.memmap:
            self.buf = mmap.mmap(self.fh.fileno(), 0, mmap.MAP_SHARED, mmap.ACCESS_READ)
            self.fh.close()
            self.fh = self.buf

        self.dict = []
        self.read()

        # keys to ignore for hashing
        self.hashignore = set()

    @staticmethod
    def FindFiles(i1=0,i2=10000, path='./',pattern=r"^grid_n([0-9]{5})\.slh$"):

        files = []

        allfiles = os.listdir(path)

        for filename in allfiles:
            if re.match(pattern, filename):
                i = int(re.sub(pattern,r"\1",filename))
                if i>=i1 and i<=i2:
                    files.append(filename)

        files.sort()
        return files

    def read(self):
        from struct import unpack

        s = self.fh.read(3)
        if s != b'SLH' and s != b'LHC':
            raise IOError('file {} does not start with SLH'.format(
                self.filename))

        self.fileformat = self.fh.read(self.sslen).decode('ascii')
        self.endianstr  = self.fh.read(self.sslen).decode('ascii')

        self.endianint = int(self.endianstr)

        self.fh.seek(-24,2)

        if (self.fh.read(8) != b'LASTDICT'):
            raise IOError('LASTDICT not found in file {}'.format(self.filename))

        endian = self.readendian()

        self.lastdict = unpack(endian+"q",self.fh.read(8))[0]

        self.fh.seek(self.lastdict,0)
        self.readdict()
        self.readallheads()

    def readdict(self):
        from struct import unpack

        while True:

            if self.fh.read(3) != b'DIC':
                raise IOError('DIC not found')

            endian = self.readendian()

            prevdict = unpack(endian+"q",self.fh.read(8))[0]
            nentries = unpack(endian+"q",self.fh.read(8))[0]

            for i in range(nentries):
                self.dict.append(unpack(endian+"q",self.fh.read(8))[0])

            if prevdict >= 0:
                self.fh.seek(prevdict,0)
                continue
            else:
                break

    def readendian(self):
        from struct import unpack

        s = self.fh.read(8)

        if unpack("<q",s)[0] == self.endianint:
            return "<"

        if unpack(">q",s)[0] == self.endianint:
            return ">"

        raise IOError('readendian')

    def readallheads(self):

        self.ident = []
        self.identdic = {}

        for off in self.dict:

            self.fh.seek(off + 3 + 8) # skip type and endianness

            ident = self.fh.read(self.sslen).decode('ascii').rstrip()
            self.ident.append(ident)
            if ident not in self.identdic:
                # The first that we read value was added last. We use that one.
                self.identdic[ident] = off

    def readitembyident(self,ident):
        self.fh.seek(self.identdic[ident],os.SEEK_SET)
        return self.readitem()

    def readitematpos(self,pos):
        self.fh.seek(pos,0)
        return self.readitem()

    def readitem(self):

        oldoff = self.fh.tell()
        t = self.fh.read(3).decode('ascii')
        self.fh.seek(oldoff,os.SEEK_SET)

        if t == 'TAB':
            return slhoutput_tab(self)
        elif t == 'VEC':
            return slhoutput_vec(self)
        elif t == 'ARR':
            return slhoutput_arr(self)
        else:
            raise IOError('unknown type '+t)

    def compare(self, other, dataonly=False):
        """compares this object with another object
        returns a (possibly empty) list of keys with differing contents"""
        if not isinstance(other, slhoutput):
            # different subclasses are tolerated
            raise TypeError("argument must be a subclass of slhoutput")
        mykeys = set(self.keys())
        okeys  = set(other.keys())
        keys = [k for k in mykeys.intersection(okeys) if np.any(self[k] != other[k])]
        keys += list(mykeys.symmetric_difference(okeys))
        if dataonly:
            keys = [x for x in keys if x.endswith('%data')]
        return keys

    def compilercmd(self, modname):
        """return compiler command used to compile 'modname.f90'
        """
        import base64

        try:
            cmd = self['compilercmd'][modname]
            return base64.b64decode(cmd)
        except KeyError:
            raise KeyError('{} not found. Valid file names are '.format(modname),
                           list(self['compilercmd'].keys()))

    def add_vector_data(self, name, value):
        """Add the key "name" containing vector data "value" to this grid file."""
        if not isinstance(value, np.ndarray):
            raise TypeError('value must be of type numpy.ndarray')
        if value.ndim > 4:
            raise IndexError('value must be 4-dimensional at most')
        if value.ndim < 4:
            newshape = (4 - value.ndim) * [1] + list(value.shape)
            value = value.reshape(newshape, order='A')

        lw = slhoutput_writer(self.filename, mode='a')
        try:
            lw.write_vec(name, 'FD', value)
        finally:
            lw.close()

    def __eq__(self, other):
        try:
            return len(self.compare(other)) == 0
        except TypeError:
            return super(slhoutput,self).__eq__(other)

    def __ne__(self, other):
        return not self == other

    def __getitem__(self,ident):
        return self.readitembyident(ident)

    def __len__(self):
        return len(self.ident)

    def __iter__(self):
        for i in self.ident:
            yield i

    def __contains__(self, value):
        return value in self.ident

    def __hash__(self):
        kv = set(self.keys())
        kv = list(kv.difference(self.hashignore))
        kv.sort()
        return hash(tuple(hash(self[a]) for a in kv))

    @property
    def revisionid(self):
        return self['repositoryinfo']['revisionid']


def slhoutput_arr(slhout):
    from struct import unpack
    global ptypes

    fh = slhout.fh

    if fh.read(3) != b'ARR':
        raise IOError('ARR not found')

    endian = slhout.readendian()

    # skip ident
    fh.seek(slhout.sslen, 1)

    type_str = fh.read(2).decode('ascii')
    try:
        ptype = ptypes[type_str]
    except KeyError:
        raise IOError('unknown type %s' % type_str)

    dims = unpack(endian+"q", fh.read(8))[0]

    if slhout.memmap:
        vals_per_dim = np.ndarray([dims], dtype=endian+'i8', buffer=slhout.buf, offset=slhout.fh.tell(), order='F')
        slhout.fh.seek(dims*8,1)
        out = np.ndarray(vals_per_dim, dtype=endian+ptype, buffer=slhout.buf, offset=slhout.fh.tell(), order='F')
    else:
        vals_per_dim = np.fromfile(fh, dtype=endian+'i8', count=dims)
        out = np.fromfile(fh, dtype=endian+ptype, count=vals_per_dim.prod()).reshape(vals_per_dim,order="f")

    if type_str == 'FQ':
        out = np.vectorize(lambda x: parse_quad(x, endian=endian))(out)

    return out

class slhoutput_tab(collections.Mapping):

    def __init__(self,slhout):
        self.slhout = slhout
        self.fh = self.slhout.fh
        self.read()

    def read(self):
        from struct import unpack

        if self.fh.read(3) != b'TAB':
            raise IOError('TAB not found')

        endian = self.slhout.readendian()
        self.endian = endian

        self.ident = self.fh.read(self.slhout.sslen).rstrip()
        self.nentries = unpack(endian+"q", self.fh.read(8))[0]

        self.type  = []
        self.ident = []
        self.value = []

        for i in range(self.nentries):

            t = self.fh.read(2).decode('ascii')
            self.type.append(t)
            self.ident.append(self.fh.read(self.slhout.sslen).decode('ascii').rstrip())

            if t == 'SS':
                self.value.append(self.fh.read(self.slhout.sslen).decode('ascii').rstrip())
            elif t == 'SL':
                self.value.append(self.fh.read(self.slhout.sllen).decode('ascii').rstrip())
            elif t == 'SV':
                svlen = unpack(endian+"q",self.fh.read(8))[0]
                self.value.append(self.fh.read(svlen).decode('ascii').rstrip())
            elif t == 'I4':
                self.value.append(unpack(endian+"l",self.fh.read(4))[0])
            elif t == 'I8':
                self.value.append(unpack(endian+"q",self.fh.read(8))[0])
            elif t == 'OB':
                self.value.append(unpack(endian+"q",self.fh.read(8))[0])
            elif t == 'FS':
                self.value.append(unpack(endian+"f",self.fh.read(4))[0])
            elif t == 'FD':
                self.value.append(unpack(endian+"d",self.fh.read(8))[0])
            elif t == 'FQ':
                self.value.append(parse_quad(self.fh.read(16), endian=endian))
            else:
                raise IOError('unknown type')
    def foroutput(self):
        """returns data in a format compatible with the slhoutput_writer"""
        return dict([(i,(t,v)) for i, t, v in zip(self.ident, self.type, self.value)])

    def __getitem__(self,key):
        try:
            return self.value[self.ident.index(key)]
        except ValueError:
            raise KeyError('%s not in slhoutput file' % key)

    def __len__(self):
        return len(self.ident)

    def __iter__(self):
        for i in self.ident:
            yield i

    def __contains__(self, value):
        return value in self.ident

    def __str__(self):
        return '\n'.join(["%s: %s" % (i, x) for i, x in zip(self.ident, self.value)])

    def __hash__(self):
        m = slhhash()
        m.update(str(self).encode('ascii'))
        return int(m.hexdigest(),16)

def shift_array(a,dim,c):

    b=zeros(shape(a))

    s=shape(a)

    if dim==0:
        b[0:c,:,:] = a[s[0]-c:,:,:]
        b[c:,:,:] = a[0:s[0]-c,:,:]
    if dim==1:
        b[:,0:c,:] = a[:,s[1]-c:,:]
        b[:,c:,:] = a[:,0:s[1]-c,:]
    if dim==2:
        b[:,:,0:c] = a[:,:,s[2]-c:]
        b[:,:,c:] = a[:,:,0:s[2]-c]

    return b

class slhoutput_vec(np.ndarray):

    def __new__(cls, slhout):
        from struct import unpack
        global ptypes

        fh = slhout.fh

        if fh.read(3) != b'VEC':
            raise ValueError('VEC not found')

        endian = slhout.readendian()


        ident = fh.read(slhout.sslen).decode('ascii').rstrip()

        type = fh.read(2).decode('ascii')

        try:
            ptype = ptypes[type]
        except KeyError:
            raise TypeError('unknown type %s' % type)

        gnc = unpack(endian+"4q",fh.read(4*8))

        nvals = gnc[0]*gnc[1]*gnc[2]*gnc[3]

        if type == 'FQ':
            # this only works for IBM's quad precision numbers
            #
            self = super(slhoutput_vec, cls).__new__(cls, gnc)
            for iz in             range(self.shape[3]):
                for iy in         range(self.shape[2]):
                    for ix in     range(self.shape[1]):
                        for iv in range(self.shape[0]):
                            buf = fh.read(16)
                            self[iv,ix,iy,iz] = parse_quad(buf, endian=endian)
        else:
            if slhout.memmap:
                self = super(slhoutput_vec, cls).__new__(
                        cls, gnc, dtype=endian+ptype,
                        buffer=slhout.buf, offset=fh.tell(), order='F')
            else:
                self = np.fromfile(fh,dtype=endian+ptype,count=nvals) \
                        .reshape(gnc[0],gnc[1],gnc[2],gnc[3],order="F") \
                        .view(type=cls)

        self.slhout = slhout
        self.fh = self.slhout.fh
        self.endian = endian
        self.type = type
        self.ident = ident
        self.gnc = gnc

        return self

    def __hash__(self):
        return hash(self.tobytes())

    def project(self, a, orthogonal=False, along=False, slicer=slice(None)):
        """returns component of this vector in the direction of a

        orthogonal: returns the *norm* of the orthogonal component instead
        along: return projection muith vector """

        if len(a) == len(slicer):
            a = np.squeeze(a[slicer])

        d = np.squeeze(self[slicer])
        if self.shape[0] != a.shape[0]:
            raise ValueError("vectors must have same dimensionality (0th dimension)")
        if self.shape[0] <= 1:
            raise ValueError("vectors must be at least two-dimensional")
        norm = np.linalg.norm(a, axis=0)
        v = (d*a).sum(axis=0) / norm
        if orthogonal:
            if along:
                return (d-a*v)
            else:
                return np.linalg.norm(d - a*v[None,:]/norm[None,:], axis=0)
        else:
            if along:
                return v * a
            else:
                return v

    def rotate(self, angles, axes=None):
        """rotate vector around axes by angles

            Input arguments:
                angles: array of values of rotation angles
                axes: axes of rotate, defaults to [0,1,2]

            Example:
                rotation around z, then x then z by
                pi pi/2, and -pi
                    x,y,z = vec.rotate([pi,pi/2,-pi],[2,0,2])
        """

        if axes is None:
            axes = [0, 1, 2]

        if np.shape(axes)[0] != np.shape(angles)[0]:
            raise ValueError('rotate: number of rot. axes does \
                        not match number of angles')

        x, y, z = self[:, :, :, :]
        xo, yo, zo = x, y, z

        for axis, angle in zip(axes, angles):
            #rotate around x axis
            if axis == 0:
                y = yo*np.cos(angle) - zo*np.sin(angle)
                z = yo*np.sin(angle) + zo*np.cos(angle)
                xo, yo, zo = x, y, z

            #rotate around y axis
            elif axis == 1:
                x = xo*np.cos(angle) + zo*np.sin(angle)
                z = -xo*np.sin(angle) + zo*np.cos(angle)
                xo, yo, zo = x, y, z

            #rotate around z axis
            elif axis == 2:
                x = xo*np.cos(angle) - yo*np.sin(angle)
                y = xo*np.sin(angle) + yo*np.cos(angle)
                xo, yo, zo = x, y, z
            else:
                raise ValueError("rotate: invalid axis %d" % axis)
        return np.array([x, y, z])

    @property
    def length(self):
        return np.sqrt((self**2).sum(axis=0))

def repair_slhoutput(filename,force=False):
    fh = open(filename,"rb+")

    fh.seek(-1,os.SEEK_END)
    if force:
        fh.seek(-24,os.SEEK_END)
    fp = fh.tell()

    ss=b""

    while fp > 0:
        fp2 = max(fp - 1024,0)
        nb = fp-fp2
        fp = fp2

        fh.seek(fp)

        ss2 = fh.read(nb)
        ss3 = ss2 + ss

        i = ss3.rfind(b"LASTDICT")

        if (i>=0):
            logger.info("Found LASTDICT at %d, truncating file" % (fp+i))
            fh.seek(fp+i+24)
            fh.truncate()
            break

        ss = ss2

    if (fp==0):
        logger.warning("no LASTDICT found")

    fh.close()

class slhoutput_writer(object):

    def __init__(self,filename, mode="a", version=""):
        """creates a new slhoutput_writer object
        mode == w: truncates the old file if it exists, otherwise creates a new file
        mode == a: appends to the file if it exists, otherwise creates a new file
        version: version string, ignored if the file already exists"""

        mode = mode.lower()
        if not mode in ("a", "w"):
            raise ValueError("unknown mode %s" % mode)

        if os.access(filename, os.F_OK) and mode == "a":
            old = slhoutput(filename)
            self.endianstr = old.endianstr
            self.fh = open(filename, 'ab')
            self.prevdict = old.lastdict
        else:
            self.fh = open(filename, 'wb')
            self.endianstr = "%64s" % 1
            self.fh.write(b"SLH")
            self.fh.write(("{:<64}".format(version)).encode('ascii'))
            self.fh.write(("{:<64}".format(self.endianstr)).encode('ascii'))
            self.prevdict = -1

        self.endianint = int(self.endianstr)

    def existing(self, f, ident=None):
        """writes data from an existing slhoutput object f to this file"""
        global iptypes

        if not isinstance(f, slhoutput):
            raise TypeError("f must be an slhoutput object")
        if ident is None:
            ident = f.ident
        for i in ident:
            if isinstance(f[i], slhoutput_tab):
                self.write_tab(i, f[i].foroutput())
            elif isinstance(f[i], slhoutput_vec):
                self.write_vec(i, f[i].type, f[i])
            elif isinstance(f[i], np.ndarray):
                self.write_arr(i, iptypes(str(f[i].dtype)),  f[i])
            else:
                raise ValueError("identifier '%s' contains an unknown object" % i)

    def write_endian(self):
        self.write_I8(self.endianint)

    def write_prevdict(self):
        self.write_I8(self.prevdict)

    def write_I8(self, i):
        self.fh.write(struct.pack("q", i))

    def write_dic_header(self):
        """writes the header for a DIC with one entry immediately after the header"""

        off = self.fh.tell()
        self.fh.write(b"DIC")
        self.write_endian()
        self.write_prevdict()
        self.prevdict = off
        # only one entry per DIC
        self.write_I8(1)
        self.write_I8(self.fh.tell() + struct.calcsize("q"))


    def write_tab(self, identifier, tab):
        """writes a table to the file
        tab must be a dictionary containing (type, value) tuples"""
        global ptypes

        if not isinstance(tab,dict):
            raise TypeError("tab must be a dictionary")

        for x in tab.values():
            try:
                if x[0] not in ptypes and x[0] != 'SV':
                    raise ValueError("unknown type %s" % x[0])
            except IndexError:
                raise ValueError("dictionary elements must be (type, value) tuples")

        self.write_dic_header()
        self.fh.write(b"TAB")
        self.write_endian()
        self.fh.write(("%-64s" % identifier).encode('ascii'))
        self.write_I8(len(tab))

        for i, x in tab.items():
            self.fh.write(x[0].encode('ascii'))
            self.fh.write(("%-64s" % i).encode('ascii'))
            if x[0] == 'SV':
                s = x[1].encode('ascii')
                self.write_I8(len(s))
                self.fh.write(s)
            else:
                v = x[1]
                if x[0] == 'SS':
                    v = "%-64s" % v
                elif x[0] == 'SL':
                    v = "%-1024s" % v
                dat = np.array(v, dtype=ptypes[x[0]])
                dat.tofile(self.fh)

    def write_arr(self, identifier, typename, arr):
        """writes an ARR entry to the file
        typename must be a valid SLH type specifier
        arr is converted to a numpy array"""
        global  ptypes

        if not typename in ptypes:
            raise ValueError("%s is not a valid type specifier" % typename)

        arr = np.array(arr, dtype=ptypes[typename])

        self.write_dic_header()
        self.fh.write(b"ARR")
        self.write_endian()
        self.fh.write(("%-64s" % identifier).encode('ascii'))
        self.fh.write(typename.encode('ascii'))
        self.write_I8(len(arr.shape))
        for s in arr.shape:
            self.write_I8(s)
        # tofile always uses C ordering => transpose first
        arr.T.tofile(self.fh)

    def write_vec(self, identifier, typename, vec):
        """writes a VEC entry to the file
        typename must be a valid SLH type specifier
        vec is converted to a numpy array"""
        global  ptypes

        if not typename in ptypes:
            raise ValueError("%s is not a valid type specifier" % typename)

        vec = np.array(vec, dtype=ptypes[typename])
        if vec.ndim != 4:
            raise ValueError("vec must be four-dimensional not %d-dimensional" % vec.ndim)

        self.write_dic_header()
        self.fh.write(b"VEC")
        self.write_endian()
        self.fh.write(("%-64s" % identifier).encode('ascii'))
        self.fh.write(typename.encode('ascii'))
        for s in vec.shape:
            self.write_I8(s)
        # tofile always uses C ordering => transpose first
        vec.T.tofile(self.fh)

    def close(self):
        if not self.fh.closed:
            self.fh.seek(0, 2)
            self.fh.write(b"LASTDICT")
            self.write_endian()
            self.write_prevdict()
            self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

class slh_geometry(object):

    def __init__(self,geo):

        self.type = geo['type']
        self.alpha = geo['alpha']
        self.x1 = np.array([ geo['x1(1)'],geo['x1(2)'],geo['x1(3)'] ])
        self.x2 = np.array([ geo['x2(1)'],geo['x2(2)'],geo['x2(3)'] ])
        self.dx = np.array([ geo['dx(1)'],geo['dx(2)'],geo['dx(3)'] ])

class slh_eos(object):

    def __init__(self,eos):

        self.type = eos['type']
        self.gamma = eos['gamma']
        if 'mu' in eos.ident:
            self.mu = eos['mu']
        if 'Z' in eos.ident:
            self.Z = eos['Z']
        self.aybar_mode = eos['aybar_mode']
        try:
            self.passive_species = eos['passive_species']
        except KeyError:
            self.passive_species = False
        self.as1 = eos['as1']
        self.as2 = eos['as2']
        self.ps1 = eos['ps1']
        self.ps2 = eos['ps2']
        if 'nspecies' in eos.ident:
            self.nspecies = eos['nspecies']
        else:
            self.nspecies = max(0,self.as2-self.as1+1) + max(0,self.ps2-self.ps1+1)

        self.ionized    = eos['ionized']
        self.radiation  = eos['radiation']
        self.degenerate = eos['degenerate']
        self.coulomb    = eos['coulomb']
        self.gas        = eos.get('gas',True)

        self.strong_ps  = eos['strong_ps']

        self.cargo_potential = eos.get('cargo_potential',False)

        try:
            self.nucdata = np.zeros((self.nspecies,), dtype=([ ('name', '5str'), ('A', 'f8'), ('Z', 'f8')]))
            for i in range(1,self.nspecies+1):
                self.nucdata[i-1] = (eos['isoname(%d)'%i] if 'isoname(%d)'%i in eos else '', eos['molmass(%d)'%i], eos['charge(%d)'%i])
        except:
            del self.nucdata

class slh_sd(object):

    def __init__(self,sd):

        self.hydro_flux = sd['hydro_flux']
        self.diffusive_flux = sd['diffusive_flux']
        self.neutrino_losses = sd.get('neutrino_losses', 'none')
        self.kappac = sd['kappac']
        self.diffc = sd['diffc']
        if 'cond_ei' in sd.ident:
            self.cond_ei = sd['cond_ei']
        if 'cond_ee' in sd.ident:
            self.cond_ee = sd['cond_ee']
        if 'cond_rad_compton' in sd.ident:
            self.cond_rad_compton = sd['cond_rad_compton']
        if 'cond_rad_free_free' in sd.ident:
            self.cond_rad_free_free = sd['cond_rad_free_free']
        self.gravity = sd['gravity']
        self.lowlim = sd['lowlim']
        self.Uprojection = sd['Uprojection']
        self.species_cma = sd['species_cma']
        self.relax_residual = sd['relax_residual']
        self.relax_jacobian = sd['relax_jacobian']
        self.relax_sigma = sd['relax_sigma']
        self.ausmpup_const = sd.get('ausmpup_const',np.nan)  # new outputs don't have this const any more
        self.ausmpup_Kp = sd.get('ausmpup_Kp', 0.25)  # The default is the previously hardcoded value.
        self.ausmpup_Ku = sd.get('ausmpup_Ku', 0.75)  # The default is the previously hardcoded value.

        self.fict_rot = sd.get('fict_rot', 0)
        if 'rot_omega_x' in sd:
            self.rot_omega = np.array([sd['rot_omega_' + k] for k in ('x', 'y', 'z')])
        else:
            self.rot_omega = np.zeros(3)

        try:
            self.ausmpup_lowmach = sd['ausmpup_lowmach']
        except KeyError:
            pass
        self.Mref = sd['Mref']
        self.Mref_pdiff = sd.get('Mref_pdiff', self.Mref)

        self.rc_hydro = slh_reconst()
        self.rc_hydro.type = sd['rc_hydro%type']
        self.rc_hydro.vars = sd['rc_hydro%vars']
        self.rc_hydro.slip_lim_mu = sd['rc_hydro%slip_lim_mu']
        self.rc_hydro.muscl_kappa = sd['rc_hydro%muscl_kappa']

        self.rc_ps = slh_reconst()
        self.rc_ps.type = sd['rc_ps%type']
        self.rc_ps.vars = sd['rc_ps%vars']
        self.rc_ps.slip_lim_mu = sd['rc_ps%slip_lim_mu']
        self.rc_ps.muscl_kappa = sd['rc_ps%muscl_kappa']

class slh_reconst(object):

    def __init__(self):
        self.type = ""
        self.vars = ""
        self.slip_lim_mu = -1
        self.muscl_kappa = -1000


class slh_restart(object):

    def __init__(self,r):
        self.state = r['state']
        self.ps = r['ps']
        self.eosvars = r['eosvars']
        self.grav = r['grav']
        self.ivol = r['ivol']
        self.coords = r['coords']
        self.metric = r['metric']

class slh_qref(object):

    def __init__(self,q=None):

        keys = ['density','velocity','soundspeed','length','temperature',\
                    'entropy','gravity','gpot','thermcond','time','pressure',\
                    'Etot','Eint','Ekin','Mach','Froude','Peclet']

        if q is None:
            for key in keys:
                self.__dict__[key] = 1.0
        else:
            for key in keys:
                self.__dict__[key] = q.get(key, 1.0)

    def isdimensional(self):
        return all(x == 1.0 for x in self.__dict__.values())


class slh_modconfig_globals(object):

    def __init__(self,g):
        self.VDIM = g['VDIM']
        self.ip = g['ip']
        self.rp = g['rp']
        self.ip_mpi = g['ip_mpi']
        self.ip_out = g['ip_out']
        self.ip_lapack = g['ip_lapack']
        self.rp_lapack = g['rp_lapack']
        self.ip_pardiso = g['ip_pardiso']
        self.ip_pardiso8 = g['ip_pardiso8']
        self.rp_pardiso = g['rp_pardiso']
        self.SS_size = g['SS_size']
        self.SL_size = g['SL_size']
        self.cfg_omp_num_threads = g['cfg_omp_num_threads']
        self.idxc_rho  = g['idxc_rho']
        self.idxc_rhou = g['idxc_rhou']
        try:
            self.idxc_rhov = g['idxc_rhov']
            self.idxc_rhow = g['idxc_rhow']
        except:
            pass
        self.idxc_rhoE = g['idxc_rhoE']
        self.idxc_as1 = g['idxc_as1']
        self.idxc_momfirst = g['idxc_momfirst']
        self.idxc_momlast = g['idxc_momlast']
        self.idxeos_T = g['idxeos_T']
        self.idxeos_p = g['idxeos_p']
        self.idxeos_dp_drho = g['idxeos_dp_drho']
        self.idxeos_dp_deps = g['idxeos_dp_deps']
        self.idxeos_deps_drho = g['idxeos_deps_drho']
        self.idxeos_deps_dT = g['idxeos_deps_dT']
        self.idxeos_d_dX = g['idxeos_d_dX']
        self.idxeos_dp_dX = g['idxeos_dp_dX']
        self.idxeos_deps_dX = g['idxeos_deps_dX']
        self.idxeos_ndX = g['idxeos_ndX']

class slhgrid(slhoutput):

    @classmethod
    def file_list(cls, prefix=""):
        filelist = glob.glob(os.path.join(prefix,'*.slh')) + glob.glob(os.path.join(prefix,'*.lhc'))

        files = []

        for file in filelist:
            f = os.path.basename(file)
            m = re.search('grid_n([0-9]{5,})\\.(slh|lhc)$', f)
            if m:
                files.append((int(m.group(1)), f))

        files.sort(key=lambda x: x[0])

        return files

    def __init__(self,filename,mode='n',prefix="",**kwargs):
        """creates a new slhgrid object

        filename may have different meanings, depending on the mode:

        - 'n': filename is an integer containing the step number, e.g. filename=10 opens "grid_n00010.slh"

        - 'i': filename is an integer containing the number of the grid file from the directory listing
               e.g. filename=5 opens the 5th grid file present in the current directory
               negative values are also possible, e.g. filename=-1 opens the last grid file

        - 'f': filename is interpreted as the actual file name, e.g. filename='myspecialgrid.slh'

        - 't': filename is a floating point value describing the physical time of interest
               e.g. filename=2.5 opens the grid file which is closest to t=2.5

               WARNING: this mode is very inefficient. consider using a (cached) slhsequence if you need
                        grids at multiple points in time

        """

        if mode=='f':
            pass

        elif mode=='i':
            filename = self.file_list(prefix=prefix)[filename][1]

        elif mode=='n':
            num = filename
            for ext in ('slh', 'lhc'):
                filename = "grid_n%05d.%s" % (num, ext)
                if  os.path.exists(os.path.join(prefix, filename)):
                    break
            else:
                # To show the error message with slh extension if no files are present.
                filename = "grid_n%05d.slh" % num


        elif mode=='t':
            if os.path.isfile(os.path.join(prefix,'slhsequence.cache')):
                s = slhsequence(dir=prefix)
                times = s.get_time()
                it = np.abs(times-filename).argmin()

                filename = s.filenames[it]

            else:
                filelist = self.file_list(prefix=prefix)
                times = [slhgrid(os.path.basename(f), prefix=prefix, mode='f').time for i, f in filelist]

                times = np.array(times)
                it = np.abs(times-filename).argmin()

                filename = filelist[it][1]

        else:
            raise ValueError('unknown mode ' + str(mode))

        super(slhgrid,self).__init__(os.path.join(prefix,filename), **kwargs)

        self.init_grid()
        if 'grid%geometry' in self.ident:
            self.geometry = slh_geometry(self['grid%geometry'])
            self.vaxis = 1 if self.geometry.type=='cartesian' else 0
        else:
            logger.warning("grid%geometry not present")
        if 'grid%eos' in self.ident:
            self.eos = slh_eos(self['grid%eos'])
        else:
            logger.warning("grid%eos not present")
        if 'grid%sd' in self.ident:
            self.sd = slh_sd(self['grid%sd'])
        else:
            logger.warning("grid%sd not present")
        if 'grid%qref' in self.ident:
            self.qref = slh_qref(self['grid%qref'])
        else:
            self.qref = slh_qref()
            logger.warning("grid%qref not present")
        if 'grid%qref_calc' in self.ident:
            self.qref_calc = slh_qref(self['grid%qref_calc'])
        else:
            self.qref_calc = slh_qref()
            logger.warning("grid%qref_calc not present")
        self.globals = slh_modconfig_globals(self['modconfig_globals'])

        self.nodes = None
        self.centers = None

        self.origin = [0, 0, 0]
        self._origin_old = self.origin
        # coordinates relative to origin
        self._relative_coordinates = None

        try:
            self.wellbalancing = self['grid']['wellbalancing']
        except KeyError:
            self.wellbalancing = 'Not loaded'

        self.lab = grid_label(self)

    def init_grid(self):

        self.sdim = self['grid']['sdim']
        self.vdim = self['grid']['vdim']
        self.nas  = self['grid']['nas']
        self.nps  = self['grid']['nps']
        self.ngc  = self['grid']['ngc']
        self.step = self['grid']['step']
        self.time = self['grid']['time']

        self.i1 = np.array([ \
                self['grid']['i1(1)'], \
                self['grid']['i1(2)'], \
                self['grid']['i1(3)']])
        self.i2 = np.array([ \
                self['grid']['i2(1)'], \
                self['grid']['i2(2)'], \
                self['grid']['i2(3)']])
        self.gnc = np.array([ \
                self['grid']['gnc(1)'], \
                self['grid']['gnc(2)'], \
                self['grid']['gnc(3)']])
        self.bc1 = [ \
                self['grid']['bc1(1)'], \
                self['grid']['bc1(2)'], \
                self['grid']['bc1(3)']]
        self.bc2 = [ \
                self['grid']['bc2(1)'], \
                self['grid']['bc2(2)'], \
                self['grid']['bc2(3)']]

        self.dd = [self.gnc[i]//(self.i2[i] - self.i1[i] + 1) for i in range(3)]

    @property
    def relative_coordinates(self):
        if self._relative_coordinates is None or (
                self.origin != self._origin_old):
            self._origin_old = self.origin
            self._relative_coordinates = np.array([
                self.coords()[i] - self.origin[i] for i in range(self.sdim)])
            return self._relative_coordinates
        else:
            return self._relative_coordinates

    def get_com(self, ix, iy):
        """calculate center-of-mass for node coordinate at index ix, iy

        This routines is a copy of what is done in modgeometry.F90
        Not the most pythonic implementation but the idea to keep it easily
        comparable to the F90 implementation

        Currently only implemented in 2D
        """

        if self.sdim != 2:
            raise RuntimeError('slhoutput.py, get_com_at, only impl. in 2D')
        nodes = self.getnodes()

        triangles = np.array([[0, 1, 3], [1, 2, 3]]).T
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T

        def tarea(a, b, c):
            cross = np.cross(a-c, b-c)
            return 0.5 * np.sqrt(np.sum(cross**2))

        pcoords = np.zeros([2, 3])
        tcom = np.zeros([2])

        arsum = 0
        for it in range(0,2):
            for iv in range(0,3):
                cx, cy = points[:, triangles[iv, it]]
                pcoords[:, iv] = nodes[:, ix+cx, iy+cy, 0]
            ar = tarea(pcoords[:, 0], pcoords[:, 1], pcoords[:, 2])

            tcom += np.sum(pcoords, axis=1) / 3.0 * ar
            arsum += ar

        return tcom / arsum


    def getnodes(self,icoord=-1):
        """get an array with the cartesian coordinates of the nodes

        node .slh file is assumed to be in the same path as the grid file
        icoord>=0 return only the icoord'th component"""

        if self.nodes is None:
            nfile     = os.path.join( os.path.dirname(self.filename), 'nodes.slh')
            nfile_lhc = os.path.join( os.path.dirname(self.filename), 'nodes.lhc')
            if os.path.exists(nfile):
                self.nodesfile = slhoutput( nfile )
                self.nodes = self.nodesfile['tvector%data']
            elif os.path.exists(nfile_lhc):
                self.nodesfile = slhoutput( nfile_lhc )
                self.nodes = self.nodesfile['tvector%data']
            elif self.geometry.type == 'cartesian':
                self.nodes = np.array(np.meshgrid(*[np.linspace(self.geometry.x1[i],self.geometry.x2[i],self.gnc[i]+1) for i in range(self.sdim)], indexing='ij'))
            else:
                RuntimeError("nodes information not available")

        if icoord<0:
            return self.nodes

        sdim = self['grid']['sdim']

        if sdim == 1:
            return self.nodes[0,:,0,0]
        elif sdim == 2:
            return self.nodes[icoord,:,:,0]
        elif sdim == 3:
            return self.nodes[icoord,:,:,:]

        raise NotImplementedError


    def cell_centers(self):
        if self.geometry.type=='cartesian' and self.sdim==1:
            x1d = np.linspace( \
                self.geometry.x1[0]+0.5*self.geometry.dx[0], \
                self.geometry.x2[0]-0.5*self.geometry.dx[0], \
                self.gnc[0] )
            return x1d
        elif self.geometry.type=='cartesian' and self.sdim==2:

            x1d = np.linspace( \
                self.geometry.x1[0]+0.5*self.geometry.dx[0], \
                self.geometry.x2[0]-0.5*self.geometry.dx[0], \
                self.gnc[0] )

            y1d = np.linspace( \
                self.geometry.x1[1]+0.5*self.geometry.dx[1], \
                self.geometry.x2[1]-0.5*self.geometry.dx[1], \
                self.gnc[1] )

            x2d,y2d=np.meshgrid(x1d,y1d)
            return x2d.T,y2d.T

            pass
        else:
            raise NotImplementedError


    def radius(self):
        return np.sqrt((self.coords()**2).sum(axis=0))

    def phi(self, slicer=slice(None)):
        return np.arctan2(self.coords(1)[slicer], self.coords(0)[slicer])

    def theta(self, slicer=slice(None)):
        if self.sdim < 3:
            return 0.5 * np.pi * np.ones_like(self.coords(0)[slicer])
        else:
            varpi = np.sqrt(self.coords(0)[slicer]**2 + self.coords(1)[slicer]**2)
            return np.arctan2(varpi, self.coords(2)[slicer])

    @property
    def r(self):
        if self.sdim == 1:
            return np.squeeze(self.coords(0))
        else:
            return self.Havgp(self.radius())

    @property
    def nr(self):
        shape = self.coords().shape[1:]
        return shape[self.vaxis]

    def radial_bin(self, q, **kwargs):
        val, bin_edges = np.histogram(self.radius(), weights=q, **kwargs)
        num, bin_edges = np.histogram(self.radius(), **kwargs)
        return 0.5 * (bin_edges[:-1] + bin_edges[1:]), val / num

    def metric_divergence(self,vec,pad=True):
        '''calculate divergence of vec using grid's metric terms
        '''
        if self.sdim != 2:
            raise NotImplementedError("metric divergence only implemented in 2D so far"
                                      +"\n"
                                      +"extension to 3D straight forward and appreciated")

        idxm_area = 0
        idxm_nx   = 1
        idxm_ny   = 2
        idxm_nz   = 3
        idxm_Ji   = 4

        def diffpart(self,vec,sl,im):
            m = self.metric_ctr(im)[:,sl[0],sl[1]]
            return (m[idxm_nx]*vec[0,sl[0],sl[1]] + m[idxm_ny]*vec[1,sl[0],sl[1]])*m[idxm_area]

        a = diffpart(self,vec,[slice(2,None ),slice(1,-1)],1) \
          - diffpart(self,vec,[slice(None,-2),slice(1,-1)],1)

        b = diffpart(self,vec,[slice(1,-1),slice(2,None )],2) \
          - diffpart(self,vec,[slice(1,-1),slice(None,-2)],2)

        res = (a+b)/2/self.metric_ctr(1)[idxm_Ji,1:-1,1:-1]

        if pad:
            # extrapolate values at the edges to get a shape
            # dimension which is compatible with self.pcolor
            pad = np.zeros_like(self.rho())

            pad[1:-1,1:-1] = res

            pad[0,1:-1] = res[0,:]
            pad[-1,1:-1] = res[-1,:]
            pad[1:-1,0] = res[:,0]
            pad[1:-1,-1] = res[:,-1]

            # fill corners
            pad[0,0] = pad[0,1]
            pad[0,-1] = pad[0,-2]
            pad[-1,0] = pad[-1,-2]
            pad[-1,-1] = pad[-1,-2]


            res = pad
        return res

    def divergence(self, vec):
        from stelo.model_reader import h

        def deriv(f, x, axis, periodic=False):
            dx = np.diff(x, axis=axis)
            df = np.diff(f, axis=axis)

            dfdx = np.zeros_like(f)
            sl  = [slice(None)] * self.sdim
            sl2 = [slice(None)] * self.sdim
            sl[axis] = slice(1,None)
            sl2[axis] = 0
            sl = tuple(sl)
            sl2 = tuple(sl2)

            dfdx[sl] = df/dx
            if periodic:
                dfdx[sl2] = -np.sum(dfdx[sl], axis=axis)
            else:
                sl = np.array(sl)
                sl[axis] = 1
                dfdx[sl2] = dfdx[tuple(sl)]

            return dfdx


        if self.geometry.type == 'polar':
            vr = vec(idir=0)
            vphi = vec(idir=1)
            r = self.r[:,np.newaxis]
            phi = self.phi()

            return 1/r * h.grad(self.r,r*vr) \
                   +1/r * h.grad(self.phi()[0,:],vphi,axis=1)

        elif self.geometry.type == 'spherical':
            vrad = vec(idir=0)
            vthe = vec(idir=1)
            vphi = vec(idir=2)

            r = self.r[:, np.newaxis, np.newaxis]
            the = self.theta()
            phi = self.phi()
            vol = self.vol()

            frad = 1 / (r**2) * deriv(r**2 * vrad, r, axis=0, periodic=False)
            fthe = 1 / (r * np.sin(the)) * deriv(np.sin(the) * vthe, the, axis=1)
            fphi = 1 / (r * np.sin(the)) * deriv(vphi, phi, axis=2)

            return frad + fthe + fphi

        elif self.geometry.type == 'cartesian':

            def cartdiv(vfield, coords):
                num_dims = len(vfield)
                return np.ufunc.reduce(np.add, [np.gradient(vfield[i], coords[i], axis=i)
                    for i in range(num_dims)])

            co = self.coords()
            if self.sdim == 3:
                return cartdiv(vec(), [co[0,:,0,0], co[1,0,:,0], co[2,0,0,:]])
            else:
                return cartdiv(vec(), [co[0,:,0], co[1,0,:]])
        else:
            raise NotImplementedError(
            'slhouput_vec: divergence not implemented for {}'.format(self.geometry.type))

    def getcenters(self):
        """calculate coordinates of the cell centers from the cell nodes"""

        if self.centers is not None:
            return self.centers

        self.getnodes()

        sdim = self['grid']['sdim']

        nc = np.array(self.nodes.shape)
        nc[:] = nc[:]-1

        if sdim == 1:
            self.centers = 0.5 * (self.nodes[:,:nc[1],:,:]+self.nodes[:,1:,:,:])
        if sdim == 2:
            self.centers = 0.25 * ( \
                self.nodes[:,:nc[1],:nc[2],:] + \
                self.nodes[:,1:,:nc[2],:] + \
                self.nodes[:,:nc[1],1:,:] + \
                self.nodes[:,1:,1:,:] )
        if sdim == 3:
            self.centers = 0.125 * ( \
                self.nodes[:,:nc[1],:nc[2],:nc[3]] + \
                self.nodes[:,1:,:nc[2],:nc[3]] + \
                self.nodes[:,:nc[1],1:,:nc[3]] + \
                self.nodes[:,1:,1:,:nc[3]] + \
                self.nodes[:,:nc[1],:nc[2],1:] + \
                self.nodes[:,1:,:nc[2],1:] + \
                self.nodes[:,:nc[1],1:,1:] + \
                self.nodes[:,1:,1:,1:] )

        return self.centers

    def polar_vel(self):
        """return the velocity components v_(r,phi,[z]) in the polar coordinate base"""

        self.getcenters()

        rad = np.sqrt((self.centers**2).sum(axis=0))
        phi = np.sign(self.centers[1,:,:,:])*np.arccos(self.centers[0,:,:,:]/rad)

        vel = self['grid%vel%data']

        pvel = np.zeros(vel.shape)

        pvel[0,:,:,:] =  vel[0,:,:,:]*np.cos(phi) + vel[1,:,:,:]*np.sin(phi)
        pvel[1,:,:,:] = -vel[0,:,:,:]*np.sin(phi) + vel[1,:,:,:]*np.cos(phi)
        if vel.shape[0]>2:
            pvel[2,:,:,:] = vel[2,:,:,:]

        return pvel

    def plot(self,data,ix=-1,iy=-1,iz=-1,xunit=1.,Clear=False,NoGeo=False,Title=True,ax=None,**kwargs):
        if Clear:
            plt.clf()
        if ax is None:
            ax = plt.gca()

        if data.ndim==1:
            pdata = data
            ic = 0
        elif data.ndim==2:
            if ix>=0:
                pdata = data[ix,:]
                ic = 1
            elif iy>=0:
                pdata = data[:,iy]
                ic = 0
            else:
                raise IndexError('one of ix,iy must be greater than zero for a 1D slice')
        elif data.ndim==3:
            i1 = np.array([0,0,0])
            i2 = np.array(data.shape)

            ic = list(range(3))
            if ix>=0:
                i1[0] = ix
                i2[0] = ix
                ic.remove(0)
            if iy>=0:
                i1[1] = iy
                i2[1] = iy
                ic.remove(1)
            if iz>=0:
                i1[2] = iz
                i2[2] = iz
                ic.remove(2)
            if len(ic) != 1:
                raise IndexError('two of ix,iy,iz must be greater than zero for a 1D slice')
            ic = ic[0]
            pdata = np.squeeze(data[i1[0]:i2[0]+1,i1[1]:i2[1]+1,i1[2]:i2[2]+1])
        else:
            raise IndexError('invalid dimensionality of data (%d)' % self.ndim)

        if NoGeo:
            geo = 'none'
        else:
            geo = self.geometry.type

        if geo in {'cartesian', 'polar'}:
            x = np.linspace(self.geometry.x1[ic]+0.5*self.geometry.dx[ic], \
                                self.geometry.x2[ic]-0.5*self.geometry.dx[ic],self.gnc[ic])
            ret = ax.plot(x / xunit,pdata,**kwargs)
        else:
            ret = ax.plot(pdata,**kwargs)

        if Title:
            plt.title("t = %13.6e" % self.time)

        plt.draw_if_interactive()

        return ret

    def ddlines_toax(self,ax):

        if self.geometry.type != 'cartesian' or self.sdim !=2:
            raise NotImplementedError(
            'ddlines not implemented for given geometry or dimension')

        if self.geometry.type == 'cartesian':
            x = self.coords(0)[:,0]
            y = self.coords(1)[0,:]
            for i in range(1,self.dd[0]):
                ax.axvline(x[0]+(x[-1]-x[0])/self.dd[0] *i)
            for i in range(1,self.dd[1]):
                ax.axhline(y[0]+(y[-1]-y[0])/self.dd[1] *i)

    def plot_multi(self, data, vaxis=0, fig=None,
                   nrows_cols=None, cmap=None, vmin=None, vmax=None,
                   Clear=True, Title=True, data_titles=None, dt_pos='top',
                   dt_style={}, tickcolor='black', tunit='s', locator=None,
                   xunit=1., yunit=1., xprune=None, yprune=None,
                   xlabel=None, ylabel=None, cbar_title=None,
                   cbar_locator=None, xlim=None, ylim=None,grid=False,**kwargs):

        import stelo.model_reader as mr
        import itertools
        if not isinstance(data, list):
            raise TypeError("data must be a list of arrays to plot")

        N = len(data)
        if nrows_cols is None:
            nrows_cols = 2*[int(np.ceil(np.sqrt(N)))]
            while (nrows_cols[0] - 1) * nrows_cols[1] >= N:
                nrows_cols[0] -= 1



        if xlim is None:
            xlim = N * [None]

        if vmin is None:
            vmin = N * [None]

        if vmax is None:
            vmax = N * [None]

        if ylabel is None:
            ylabel = N * ['']

        if data_titles is None:
            data_titles = N * [None]
        elif len(data_titles) != N:
            raise RuntimeError('plot_multi, data_titles list not matching len(data)')


        if fig is None:
            fig = plt.gcf()

        if Clear:
            fig.clf()

        try:
            species = kwargs.pop('species')
        except KeyError:
            species = None

        try:
            abund0 = kwargs.pop('abund0')
        except KeyError:
            abund0 = None


        grkwargs = {}
        for k in ("direction", "axes_pad", "add_all", "share_all", "aspect", "label_mode", "cbar_mode", "cbar_location", "cbar_pad", "cbar_size", "cbar_set_cax", "axes_class"):
            if k in kwargs:
                grkwargs[k] = kwargs.pop(k)

        for i in range(len(data)):
            if isinstance(data[i],str):
                data[i] = data[i]
            else:
                data[i] = self.Havg(data[i],vaxis=vaxis,repeat=False).flatten()


        if self.geometry.type == 'polar':
            r = self.r
            #if xlabel is None: xlabel = 'r in cm'
            xlim = [self.r[0],self.r[-1]]
        elif self.geometry.type == 'cartesian':
            r = self.coords(1)[0,:]
            xlim = [self.r[0],self.r[-1]]
            xlim = None
        else:
            raise NotImplementedError(
                'plot_multi: geometry type %s not implemented'
                    %self.geometry.type)

        ims = []
        ctd = 0
        #for i in range(nrows_cols[0]):
        #    for j in range(nrows_cols[1]):
        inds = [range(nrows_cols[0]),range(nrows_cols[1])]
        for i,j in itertools.product(*inds):

            if ctd==N:
                break
            d = data[ctd]
            dt = data_titles[ctd]

            yli = [vmin[ctd],vmax[ctd]]
            ylab = ylabel[ctd]


            ax = plt.subplot2grid(nrows_cols, (i, j))
            if grid: ax.grid()
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)



            if d=='abund':
                l = mr.slh_model(self, radial=True, vaxis=0)
                l.plot_abundances(l.r, ax=ax, species=species, legend=False)
                if abund0 is not None:
                    abund0.plot_abundances(abund0.r, ax=ax, species=species,
                                           legend=False,ls=':')
                del(l)

            elif d=='dedt_nuc':
                l = mr.slh_model(self, radial=True, vaxis=0)
                dedt = l.dedt_nuc()
                try:
                    dedt = dedt * self['grid']['networkboost']
                except KeyError:
                    pass
                del(l)

                ims.append(ax.plot(r,dedt,**kwargs))
                ax.set_ylim(yli[0],yli[1])
            else:
                if isinstance(d,str):
                    print('plot_multi, not recognized plottype %s'%d)
                if(r'$N^2$' in dt):
                    ax.axhline(0)
                ims.append(ax.plot(r, d, **kwargs))
                ax.set_ylim(yli[0],yli[1])

            if i < ((nrows_cols[0]-1)*nrows_cols[1]): #true if not lowest row
               ax.xaxis.offsetText.set_visible(False) #removes offset xlabel
            #if i%(nrows_cols[1]) != 0: #true if not left column
            #   ax.yaxis.offsetText.set_visible(False) #removes offset ylabel
            if i != nrows_cols[1] * (nrows_cols[0] - 1): # all but lower left panel
                if xprune:
                    ax.xaxis.get_major_locator().set_params(prune=xprune)
                if yprune:
                    ax.yaxis.get_major_locator().set_params(prune=yprune)

            if dt:
                if dt_pos == 'top':
                    ax.set_title(dt)
                else:
                    ax.text(dt_pos[0], dt_pos[1], dt, transform=ax.transAxes, **dt_style)

            ax.set_xlim(xlim)
            ax.set_ylabel(ylab)
            if xlabel is not None:
                ax.set_xlabel(xlabel)

            ctd += 1

        if Title:
            if isinstance(Title, str):
                fig.suptitle(Title)
            else:
                if plt.rcParams['text.usetex']:
                    fig.suptitle('$t$ = ' + format_time(self.time, tunit))
                else:
                    fig.suptitle('t = '  + format_time(self.time, tunit),y=1.002)

        plt.tight_layout(pad=0.4)
        plt.draw_if_interactive()
        return ims

    def add_inlet(
        self, ax, xlim, ylim, zoom=3, loc='lower right',
        add_marker=True, loc1=None, loc2=None,
        remove_alabels=True):
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        axins = zoomed_inset_axes(ax, zoom, loc=loc)
        axins.set_xlim(xlim)
        axins.set_ylim(ylim)
        if remove_alabels:
            axins.axes.get_xaxis().set_visible(False)
            axins.axes.get_yaxis().set_visible(False)
        if add_marker:
            if loc1 is None or loc2 is None:
                raise RuntimeError(
                    'add_inlet, add_marker is True,' \
                    'please pass marker location1 as loc1' \
                    'and marker location2 as loc2')
            mark_inset(ax, axins, loc1=loc1, loc2=loc2)

        return axins

    def pcolor_multi(self, data, ix=None, iy=None, iz=None, fig=None,
        nrows_cols=None, vmin=None, vmax=None, cmap=None, Clear=True,
        Title=True, data_titles=None, dt_pos='top', dt_style={},
        tickcolor='black', tunit='s', locator=None, xunit=1., yunit=1.,
        xprune=None, yprune=None, xlabel=None, ylabel=None, cbar_title=None,
        cbar_locator=None, xlim=None, ylim=None, addboundary=None,
        add_inlet=False, inlet_kwargs=None, rotate_coords=None,
        **kwargs):

        if not isinstance(data, list):
            raise TypeError("data must be a list of arrays to plot")

        if 'rasterized' not in kwargs:
            kwargs['rasterized'] = True

        N = len(data)
        if nrows_cols is None:
            nrows_cols = 2*[int(np.ceil(np.sqrt(N)))]
            while (nrows_cols[0] - 1) * nrows_cols[1] >= N:
                nrows_cols[0] -= 1

        if vmin is None:
            vmin = N * [None]
        if vmax is None:
            vmax = N * [None]
        if cmap is None:
            cmap = N * [None]

        if data_titles is None:
            data_titles = N * [None]

        if fig is None:
            fig = plt.gcf()

        if Clear:
            fig.clf()

        grkwargs = {}
        for k in ("direction", "axes_pad", "add_all", "share_all", "aspect", "label_mode", "cbar_mode", "cbar_location", "cbar_pad", "cbar_size", "cbar_set_cax", "axes_class"):
            if k in kwargs:
                grkwargs[k] = kwargs.pop(k)

        gr = mpl_toolkits.axes_grid1.ImageGrid(fig, 111, nrows_cols, **grkwargs)
        if self.sdim==3:
            if ix is not None:
               data = [dat[i,:,:] for dat,i in zip(data,ensure_list(ix,len(data),True))]
            elif iy is not None:
               data = [dat[:,i,:] for dat,i in zip(data,ensure_list(iy,len(data),True))]
            elif iz is not None:
               data = [dat[:,:,i] for dat,i in zip(data,ensure_list(iz,len(data),True))]
            else:
               raise Exception('one of ix,iy,iz must be set for a 2D slice')
        try:
            n = self.getnodes()
        except IOError:
            n = self.coords()
        if n is None:
            n = self.coords()


        if self.sdim == 3:
            #for spherical grids, rotate coordinate to xy or xz plane
            #to avoid artifacts due to projection
            if self.geometry.type == 'spherical':
                phi = np.arctan2(n[1],n[0])
                alt = np.pi/2. - np.arccos(n[2]/n.length)
                f = self.getnodes()

                if ix is not None:
                    x_list = [n[1,ix,:,:] for i in ensure_list(ix,len(data),True)]
                    y_list = [n[2,ix,:,:] for i in ensure_list(ix,len(data),True)]

                elif iy is not None:
                    axes = [2,1,2]
                    angles = [-phi,alt,phi]
                    if rotate_coords is not None:
                        angles[0] += rotate_coords[0]
                        angles[1] += rotate_coords[1]
                        angles[2] += rotate_coords[2]
                    rotated = f.rotate(angles,axes)
                    x_list = [rotated[0,:,i,:] for i in ensure_list(iy,len(data),True)]
                    y_list = [rotated[1,:,i,:] for i in ensure_list(iy,len(data),True)]
                    #x,y,_ = f.rotate(angles,axes)[:,:,iy,:]

                elif iz is not None:
                    axes = [0,1,2]
                    angles = [0,0,-phi]
                    if rotate_coords is not None:
                        angles = np.array(angles) + np.array(rotate_coords)
                    rotated = f.rotate(angles,axes)
                    x_list = [rotated[0,:,:,i] for i in ensure_list(iz,len(data),True)]
                    y_list = [rotated[2,:,:,i] for i in ensure_list(iz,len(data),True)]
                    #x,_,y = f.rotate(angles,axes)[:,:,:,iz]
                else:
                    raise ValueError('one of ix,iy,iz must be greater than zero for a 2D slice')
            else:
                if ix is not None:
                    x_list = [n[1,i,:,:] for i in ensure_list(ix,len(data),True)]
                    y_list = [n[2,i,:,:] for i in ensure_list(ix,len(data),True)]
                elif iy is not None:
                    x_list = [n[0,:,i,:] for i in ensure_list(iy,len(data),True)]
                    y_list = [n[2,:,i,:] for i in ensure_list(iy,len(data),True)]
                elif iz is not None:
                    x_list = [n[0,:,:,i] for i in ensure_list(iz,len(data),True)]
                    y_list = [n[1,:,:,i] for i in ensure_list(iz,len(data),True)]
                else:
                    raise ValueError('one of ix,iy,iz must be greater than zero for a 2D slice')

        elif self.sdim == 2:
            x_list = [n[0]] * len(data)
            y_list = [n[1]] * len(data)

        x_list = [np.squeeze(x)/xunit for x in x_list]
        y_list = [np.squeeze(y)/yunit for y in y_list]

        if cbar_title is not None:
            cbar_title = ensure_list(cbar_title,len(data),True)
        else:
            cbar_title = [None] * len(data)

        ims = []
        for i, g, d, dt, vmi, vma, cm,x,y in zip(range(N), gr, data, data_titles, vmin, vmax, cmap,x_list,y_list):
            #if self.sdim==3:
            #    if ix is not None:
            #        d = d[ix,:,:]
            #    elif iy is not None:
            #        d = d[:,iy,:]
            #    elif iz is not None:
            #        d = d[:,:,iz]
            #    else:
            #        raise Exception('one of ix,iy,iz must be set for a 2D slice')

            ims.append(g.pcolormesh(x.T, y.T, d.T, vmin=vmi, vmax=vma, cmap=cm, **kwargs))
            if add_inlet:
                if inlet_kwargs is None:
                    raise RuntimeError(
                        'pcolor_multi, '
                        'add_inlet is True but no inlet_kwargs given')

                defaults = {
                    'loc': 'lower right',
                    'add_marker': False,
                    'remove_alabels': True
                }
                inlet_kwargs.update(
                    defaults,
                    **inlet_kwargs.copy())

                axin = self.add_inlet(
                    ax=g,
                    **inlet_kwargs
                    )
                axin.pcolormesh(
                    x.T, y.T, d.T, vmin=vmi, vmax=vma, cmap=cm, **kwargs)

            if locator:
                g.xaxis.set_major_locator(copy.copy(locator))
                g.yaxis.set_major_locator(copy.copy(locator))
            plt.setp([g.get_xticklines(), g.get_yticklines()], color=tickcolor)
            cbar = g.cax.colorbar(ims[-1])
            cbar.solids.set_edgecolor('face')
            if cbar_title[i]:
                if grkwargs.get('cbar_location',"right") in ('top', 'bottom'):
                    g.cax.set_xlabel(cbar_title[i])
                else:
                    g.cax.set_ylabel(cbar_title[i])
            plt.setp([g.cax.get_xticklines(), g.cax.get_yticklines()], color=tickcolor)
            if cbar_locator:
                g.cax.xaxis.set_major_locator(copy.copy(cbar_locator))
                g.cax.yaxis.set_major_locator(copy.copy(cbar_locator))
            if i < ((nrows_cols[0]-1)*nrows_cols[1]): #true if not lowest row
               g.xaxis.offsetText.set_visible(False) #removes offset xlabel
            if i%(nrows_cols[1]) != 0: #true if not left column
               g.yaxis.offsetText.set_visible(False) #removes offset ylabel
            if i != nrows_cols[1] * (nrows_cols[0] - 1): # all but lower left panel
                if xprune:
                    g.xaxis.get_major_locator().set_params(prune=xprune)
                if yprune:
                    g.yaxis.get_major_locator().set_params(prune=yprune)
            if xlabel:
                g.set_xlabel(xlabel)
            if ylabel:
                g.set_ylabel(ylabel)
            if xlim:
                    g.set_xlim(xlim)
            if ylim:
                    g.set_ylim(ylim)
            if dt:
                if dt_pos == 'top':
                    g.set_title(dt)
                else:
                    g.text(dt_pos[0], dt_pos[1], dt, transform=g.transAxes,
                    **dt_style)

            if iy is not None: idir = 1; idx=iy
            if iz is not None: idir = 2; idx=iz
            if i == addboundary:
                #self.plot_boundaries(g, mode='grad_abar', idir=idir, mean=idx,
                #    ls=':', color='white', lw=0.75)
                self.plot_boundaries(g, mode='grad_ps', ls=':', idir=idir, mean=idx,
                    color='white', lw=1.75)

        if Title:
            if isinstance(Title, str):
                fig.suptitle(Title)
            else:
                if plt.rcParams['text.usetex']:
                    fig.suptitle('$t$ = ' + format_time(self.time, tunit))
                else:
                    fig.suptitle('t = '  + format_time(self.time, tunit))

        plt.draw_if_interactive()
        return ims

    def pcolor(self,data,ix=-1,iy=-1,iz=-1,vsymm=False,Clear=True,AxisImage=True,CenterXAxis=False,Colorbar=True,NoGeo=False,Title=True,postprocess=None,tickcolor='black',XScale=1., YScale=1., tunit='s', **kwargs):
        if Clear:
            plt.clf()

        if 'rasterized' not in kwargs:
            kwargs['rasterized'] = True

        for k in ('vmin', 'vmax'):
            if k in kwargs:
                try:
                    kwargs[k] = ensure_scalar(kwargs[k])
                except ValueError:
                    raise ValueError('{0} must be scalar'.format(k))

        if isinstance(data,str):
            data = self['grid%{0}%data'.format(data)]

        if self.sdim==3:
            if ix>=0:
                pdata = data[ix,:,:]
            elif iy>=0:
                pdata = data[:,iy,:]
            elif iz>=0:
                pdata = data[:,:,iz]
            else:
                raise Exception('one of ix,iy,iz must be greater than zero for a 2D slice')
        else:
            pdata = data

        if vsymm:
            vmin=pdata.min()
            vmax=pdata.max()
            vabsmax = max(abs(vmin),abs(vmax))

            kwargs['vmin'] = -vabsmax
            kwargs['vmax'] = +vabsmax

        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanmin(pdata)
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanmax(pdata)


        if NoGeo:
            geo = 'none'
        else:
            geo = self.geometry.type

        if geo=='cartesian':
            if self.sdim==2:
                if CenterXAxis:
                    x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0]+1)-self.geometry.x2[0]/2
                    y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1]+1)
                else:
                    x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0]+1)
                    y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1]+1)
            else:
                if ix>=0:
                    x = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1]+1)
                    y = np.linspace(self.geometry.x1[2],self.geometry.x2[2],self.gnc[2]+1)
                elif iy>=0:
                    x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0]+1)
                    y = np.linspace(self.geometry.x1[2],self.geometry.x2[2],self.gnc[2]+1)
                elif iz>=0:
                    x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0]+1)
                    y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1]+1)


            im = pcolor(x*XScale,y*YScale,pdata.T, **kwargs)
        elif geo=='none':
            im = pcolor(pdata.T, **kwargs)
        else:
            try:
                n = self.getnodes()
            except IOError:
                n = self.coords()
            if n is None:
                n = self.coords()

            if self.sdim == 3:

            #for spherical grids, rotate coordinate to xy or xz plane
            #to avoid artifacts due to projection
                if geo == 'spherical':
                    phi = np.arctan2(n[1],n[0])
                    alt = np.pi/2. - np.arccos(n[2]/n.length)
                    f = self.getnodes()

                    if ix >= 0:
                        x = n[1,ix,:,:]
                        y = n[2,ix,:,:]

                    elif iy >= 0:
                        axes = [2,1,2]
                        angles = [-phi,alt,phi]
                        x,y,_ = f.rotate(angles,axes)[:,:,iy,:]

                    elif iz >= 0:
                        axes = [2]
                        angles = [-phi]
                        x,_,y = f.rotate(angles,axes)[:,:,:,iz]
                    else:
                        raise ValueError('one of ix,iy,iz must be greater than zero for a 2D slice')
                else:
                    if ix >= 0:
                        x = n[1,ix,:,:]
                        y = n[2,ix,:,:]
                    elif iy >= 0:
                        x = n[0,:,iy,:]
                        y = n[2,:,iy,:]
                    elif iz >= 0:
                        x = n[0,:,:,iz]
                        y = n[1,:,:,iz]
                    else:
                        raise ValueError('one of ix,iy,iz must be greater than zero for a 2D slice')
            elif self.sdim == 2:
                x = n[0]
                y = n[1]
                if geo == 'spherical':
                    x = -x
            else:
                raise ValueError("invalid spatial dimensionality (%d)" % grid.sdim)
            x = np.squeeze(x)
            y = np.squeeze(y)

            im = pcolor(x.T*XScale,y.T*YScale,pdata.T, **kwargs)

        ax = plt.gca()
#        plt.setp(ax.spines.values(), color=tickcolor)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=tickcolor)

        if AxisImage:
            plt.axis('image')
        if Title:
            if plt.rcParams['text.usetex']:
                plt.suptitle('$t$ = ' + format_time(self.time, tunit))
            else:
                plt.suptitle('t = '  + format_time(self.time, tunit))
        if Colorbar:
            ax = plt.gca()
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.solids.set_edgecolor('face')
#            plt.setp(cax.spines.values(), color=tickcolor)
            plt.setp([cax.get_xticklines(), cax.get_yticklines()], color=tickcolor)
            plt.sca(ax)
        if postprocess:
            postprocess(self)

        plt.draw_if_interactive()

        return im

    def grid(self, edgecolor='k', alpha=0.1, antialiased=True, **kwargs):
        self.pcolor(self.rho(), edgecolor=edgecolor, facecolor=(0,0,0,0), alpha=alpha, antialiased=antialiased, Colorbar=False, **kwargs)

    def contour(self,data,ix=-1,iy=-1,iz=-1,Clear=True,AxisImage=True,Colorbar=True,NoGeo=False,Title=True,XScale=1.0,YScale=1.0,**kwargs):
        if Clear:
            plt.clf()

        if isinstance(data,str):
            data = self['grid%{0}%data'.format(data)]

        if self.sdim==3:
            if ix>=0:
                pdata = data[:,ix,:,:]
            elif iy>=0:
                pdata = data[:,:,iy,:]
            elif iz>=0:
                pdata = data[:,:,:,iz]
            else:
                raise Exception('one of ix,iy,iz must be greater than zero for a 2D slice')
        else:
            pdata = data

        if NoGeo:
            geo = 'none'
        else:
            geo = self.geometry.type

        if geo=='cartesian':
            x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0])*XScale
            y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1])*YScale

            plt.contour(x,y,pdata, **kwargs)
        elif geo=='polar':
            #n = self.getnodes()
            #x = n[0,:-1,:-1,0]
            #y = n[1,:-1,:-1,0]

            plt.contour(self.coords(0)*XScale,self.coords(1)*YScale,pdata, **kwargs)

        else:
            plt.contour(pdata, **kwargs)


        if Colorbar:
            plt.colorbar();
        if AxisImage:
            plt.axis('image');
        if Title:
            plt.title("t = %13.6e" % self.time)

        plt.draw_if_interactive()

    def quiver(self,data,ix=-1,iy=-1,iz=-1,Clear=True,AxisImage=True,NoGeo=False,Title=True,Trim=None,cdata=None,**kwargs):
        if Clear:
            plt.clf()

        if isinstance(data, str):
            data = self['grid%{0}%data'.format(data)]
        if cdata is not None:
            if cdata.ndim != (data.ndim-1):
                raise Exception('cdata must have the same dimension as data')
            else:
                setc = True
        else:
            setc = False
            codata = np.full(np.shape(data)[1:],0,dtype='int64')

        if self.sdim==3:
            if ix>=0:
                pdata = data[:,ix,:,:]
                if setc: codata = cdata[ix,:,:]
            elif iy>=0:
                pdata = data[:,:,iy,:]
                if setc: codata = cdata[:,iy,:]
            elif iz>=0:
                pdata = data[:,:,:,iz]
                if setc: codata = cdata[:,:,iz]
            else:
                raise Exception('one of ix,iy,iz must be greater than zero for a 2D slice')
        else:
            pdata = data


        if NoGeo:
            geo = 'none'
        else:
            geo = self.geometry.type

        if geo=='cartesian':
            if ix>=0:
                x = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1])
                y = np.linspace(self.geometry.x1[2],self.geometry.x2[2],self.gnc[2])
                pdata = pdata[1:,:,:]
            elif iy>=0:
                x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0])
                y = np.linspace(self.geometry.x1[2],self.geometry.x2[2],self.gnc[2])
                pdata = pdata[::2,:,:]
            elif iz>=0:
                x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0])
                y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1])
                pdata = pdata[:2,:,:]
            else: # 2D case
                x = np.linspace(self.geometry.x1[0],self.geometry.x2[0],self.gnc[0])
                y = np.linspace(self.geometry.x1[1],self.geometry.x2[1],self.gnc[1])

            #plt.quiver(x[::Trim,::Trim],y[::Trim,::Trim],pdata[0,::Trim,::Trim].T,pdata[1,::Trim,::Trim].T, **kwargs)
            qvr = plt.quiver(x[::Trim],y[::Trim],pdata[0,::Trim,::Trim].T,pdata[1,::Trim,::Trim].T,codata.T[::Trim,::Trim], **kwargs)
        elif geo=='polar':
            x = self.coords(0)
            y = self.coords(1)
            if Trim is not None:
                x = x[::Trim, ::Trim]
                y = y[::Trim, ::Trim]
                pdata = pdata[:, ::Trim, ::Trim]
                codata = codata[::Trim,::Trim]
            qvr = plt.quiver(x,y,pdata[0,:,:],pdata[1,:,:],codata.T, **kwargs)


        elif geo=='spherical':
            n = self.coords()
            phi = self.phi()
            alt = np.pi/2. - self.theta()

            if   ix >= 0:
                raise NotImplementedError('only iy,iz>0 supported so far')
            elif iy >= 0:
                axes = [2,1,2]
                angles = [-phi,alt,phi]

                x,y,z = n.rotate(angles,axes)[:,::Trim,iy,::Trim]
                x_d,y_d,z_d = data.rotate(angles,axes)[:,::Trim,iy,::Trim]
                codata = codata[::Trim,::Trim]
                qvr = plt.quiver(x.T,y.T,x_d.T,y_d.T,codata.T,**kwargs)

            elif iz >= 0:
                axes = [2]
                angles = [-phi]

                x,y,z = n.rotate(angles,axes)[:,::Trim,::Trim,iz]
                x_d,y_d,z_d = data.rotate(angles,axes)[:,::Trim,::Trim,iz]
                codata = codata[::Trim,::Trim]

                qvr = plt.quiver(x.T,z.T,x_d.T,z_d.T,codata.T,**kwargs)

            else:
                raise Exception('one of ix,iy,iz must be greater than zero for 2D quiver')

        else:
            qvr = plt.quiver(pdata[0,:,:].T,pdata[1,:,:].T,codata.T, **kwargs)


        if AxisImage:
            plt.axis('image');
        if Title:
            plt.title("t = %13.6e" % self.time)

        plt.draw_if_interactive()
        return qvr

    def Havg(self,data,vaxis=1,**kwargs):
        """horizontal average
        vaxis is the vertical axis"""
        return Havg(data,vaxis,**kwargs)

    def Havgp(self, data):
        """ horizontal average"""
        h = Havg(data, self.vaxis, repeat=False)
        h.shape = (self.nr,)
        return h

    def HavgPlot(self, data, Clear=False, XScale=1., **kwargs):
        """ plots horizontally averaged profile of *data*"""

        #if self.geometry.type=='cartesian':
        #    pdata = self.Havg(data, vaxis=self.vaxis)[0,:]
        #else:
        #    pdata = self.Havg(data, vaxis=self.vaxis)[:,0]

        pdata = self.Havgp(data)
        if Clear:
            plt.clf()

        r = self.r*XScale

        plt.plot(r, pdata, **kwargs)
        plt.draw_if_interactive()

    def yt_dataset(self, add_fields=None):
        import yt

        if self.geometry.type != 'cartesian':
            raise NotImplementedError('only Cartesian geometry supported so far')

        data = dict()
        v = self.vel()
        for a, i in zip('xyz', range(v.shape[0])):
            data['velocity_' + a] = v[i]
        data['density'] =self.rho()
        data['temperature'] =self.temp()
        data['pressure'] =self.temp()

        if add_fields:
            data.update(add_fields)

        bbox = np.vstack([self.geometry.x1, self.geometry.x2]).T

        ds = yt.load_uniform_grid(data, self.gnc, sim_time=self.time, periodicity=3*[False], bbox=bbox, length_unit='cm', velocity_unit='cm/s')

        return ds

    def rho(self):
        return np.squeeze(self['grid%rho%data'])

    @vector
    def vel(self,ivel=-1,inertialframe=False):
        """velocity, ivel >= 0 selects specific axis"""
        v = np.squeeze(self['grid%vel%data'])
        if inertialframe and self.sd.fict_rot:
            v = v + np.cross(self.sd.rot_omega, self.coords(), axis=0)[:v.shape[0]]
        if ivel>=0:
            return v[ivel]
        else:
            return v

    def pres(self, cargo=False):
        """set cargo to True to see the deviation from hydrostatic background only
        otherwise the total physical pressure is returned"""
        p = np.squeeze(self['grid%pres%data'])
        if cargo or not self.eos.cargo_potential:
            return p
        else:
            return p + self.cargo_pot()

    def omega(self):
       """angular velocity assuming a rotation around the z axis"""
       e_phi = np.zeros_like(self.coords())
       phi = np.arctan2(self.coords()[1],self.coords()[0])
       e_phi[:2] = -np.sin(phi),np.cos(phi)#unit vector in phi direction (polar plane)
       v_phi = np.abs(self.vel().project(e_phi)) #velocity in phi direction
       polarrad = np.sqrt(self.coords()[0]**2+self.coords()[1]**2) #radius in the polar plane
       return v_phi / polarrad #return omega = vphi/rpol

    def L(self):
        """angular momentum per volume"""
        return np.cross(self.coords(), self.vel(), axis=0) * self.rho()

    def temp(self):
        if not 'grid%temp%data' in self.ident and self.eos.type == 'ideal':
            t = self.pres() * self.eos.mu / (constants.Rgas * self.rho())
            return t * self.qref.pressure/self.qref.rho/self.qref.temperature
        return np.squeeze(self['grid%temp%data'])

    @fallback
    def cargo_pot(self):
        return np.squeeze(self['grid%cargo_pot%data'])

    @fallback
    def rho0(self):
        return np.squeeze(self['grid%rho0%data'])

    def entropy(self):
        """computes specific entropy"""
        import c_bindings
        eos_mode = self.eos_mode()
        fun = np.vectorize(lambda r,t,a,z: c_bindings.helm_eos_calc_tgiven(r, t, a, z, mode=eos_mode)[2])
        return fun(self.rho(),self.temp(),self.abar(),self.zbar())

    def enthalpy(self):
        """computes specific enthalpy"""
        return (self.eps() + self.pres()) / self.rho()

    def cspeed(self):
        if not 'grid%cspeed%data' in self.ident and self.eos.type=='ideal':
                return np.sqrt(self.pres()*self.eos.gamma/self.rho())
        return np.squeeze(self['grid%cspeed%data'])

    @fallback
    def vol(self):
        return np.squeeze(self['grid%vol%data'])

    def eps(self, cargo=False):
        """internal energy per volume
        cargo: return modified energy according to Cargo-LeRoux WB method (default: False)"""
        if not 'grid%eps%data' in self.ident and self.eos.type in ('ideal', 'ideal_species'):
            eps = self.pres()/(self.eos.gamma-1.0)
            if self.eos.cargo_potential and cargo:
                eps += self.cargo_pot()
        else:
            eps = np.copy(np.squeeze(self['grid%eps%data']))
            if self.eos.cargo_potential and not cargo:
                eps -= self.cargo_pot()

        return eps

    def state(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%state%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%state%data'])

    def eosvars(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%eosvars%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%eosvars%data'])

    def eos_mode(self):
        return list(filter(lambda x: self['grid%eos'][x], ['ionized', 'radiation', 'degenerate', 'coulomb', 'gas']))

    @fallback
    @vector
    def grav(self,i=-1):
        """gravity, i >= 0 selects specific axis"""

        if i>=0:
            return np.squeeze(self['grid%grav%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%grav%data'])

    def gpot0(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%gpot0%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%gpot0%data'])

    def gpot1(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%gpot1%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%gpot1%data'])

    def gpot2(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%gpot2%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%gpot2%data'])

    def gpot3(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%gpot3%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%gpot3%data'])

    @fallback
    def coords(self, i=-1):
        if i>=0:
            return np.squeeze(self['grid%coords%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%coords%data'])

    @fallback
    def com(self,i=-1):
        if i>=0:
            return np.squeeze(self['grid%com%data'][i,:,:,:])
        else:
            return np.squeeze(self['grid%com%data'])

    @fallback
    def metric_ctr(self,idir,im=-1):
        if im>=0:
            return np.squeeze(self['grid%%metric_ctr%d%%data' % idir][im,:,:,:])
        else:
            return np.squeeze(self['grid%%metric_ctr%d%%data' % idir])

    @fallback
    def metric(self,idir,im=-1):
        if im>=0:
            return np.squeeze(self['grid%%metric%d%%data' % idir][im,:,:,:])
        else:
            return np.squeeze(self['grid%%metric%d%%data' % idir])

    def asc(self,ias=-1):
        if ias>=0:
            return np.squeeze(self['grid%as%data'][ias,:,:,:])
        else:
            return np.squeeze(self['grid%as%data'])

    def ps(self,ips=-1):
        if ips>=0:
            return np.squeeze(self['grid%ps%data'][ips,:,:,:])
        else:
            return np.squeeze(self['grid%ps%data'])

    def vorticity(self, iv=None):
        vort = curl(self.vel())
        if iv is not None:
            return vort[iv]
        else:
            return np.sqrt((vort**2).sum(axis=0))

    def opacity(self):
        if self.sd.diffusive_flux == 'const_kappa':
            return self.sd.kappac * np.ones_like(self.rho())
        if self.sd.diffusive_flux == 'var_kappa':
            return self.kappa()
        elif self.sd.diffusive_flux == 'const':
            return 16. * constants.sigma / (3. * self.sd.diffc) * self.temp()**3 / self.rho()
        elif self.sd.diffusive_flux == 'none':
            return 0.
        else:
            raise NotImplementedError("%s mode not supported" % self.sd.diffusive_flux)

    @fallback
    def kappa(self):
        '''kappa aka opacity - not to be confused with diffusivity
        '''
        return np.squeeze(self['grid%kappa%data'])

    def thermcond(self):
        if self.sd.diffusive_flux == 'const_kappa':
            return 16. * constants.sigma / (3. * self.sd.kappac) * self.temp()**3 / self.rho()
        elif self.sd.diffusive_flux == 'var_kappa':
            return 16. * constants.sigma / (3. * self.kappa()) * self.temp()**3 / self.rho()
        elif self.sd.diffusive_flux == 'const':
            return self.sd.diffc * np.ones_like(g.rho())
        elif self.sd.diffusive_flux == 'none':
            return 0.
        else:
            raise NotImplementedError("%s mode not supported" % self.sd.diffusive_flux)

    def mach(self,iv=-1,dim=True, radial=False, inertialframe=False, project=False, orthogonal=False):
        """default(project==False): if radial is True iv == 0 selects the radial component
        and iv >= 1 selects the orthogonal component

        projection==True enables additional options (note: the radial keyword will be ignored)
           iv sets direction for projection in spherical coordinates (independent of grid.geometry.type)
           orthogonal==True: return *norm* of orthogonal component instead

           2D:
            radial: iv == 0
            azimuthal (phi): iv == 1

           3D:
            radial: iv == 0
            horizontal (theta): iv == 1
            azimuthal (phi): iv == 2
        """

        if not project:
            if iv>=0:
                if radial:
                    if iv == 0:
                        m = self.vel(idir=0,inertialframe=inertialframe)
                    else:
                        m = self.vel(idir=0,orthogonal=True,inertialframe=inertialframe)
                else:
                    m = self.vel(ivel=iv,inertialframe=inertialframe)
            else:
                if radial:
                    m = np.abs(self.vel(idir=0,inertialframe=inertialframe))
                else:
                    m = self.absvel(inertialframe=inertialframe)
        else:
            m = self.vel(idir=iv,orthogonal=orthogonal,inertialframe=inertialframe)

        m = m / self.cspeed()

        if dim:
            m *= self.qref.Mach
        return m

    def absvel(self, **kwargs):
        if self.sdim == 1:
            return np.abs(self.vel(**kwargs))
        else:
            return np.linalg.norm(self.vel(**kwargs),axis=0)

    def absgrav(self):
        res = self.grav(0)**2
        for i in range(1,self.sdim):
            res = res + self.grav(i)**2
        return np.sqrt(res)

    def ekin(self):
        """kinetic energy per volume"""
        ek = self.vel(0)**2
        for i in range(1,self.sdim):
            ek += self.vel(i)**2
        return 0.5*self.rho()*ek

    def etot(self):
        """total energy per volume"""
        return self.ekin() + self.eps()

    def get_ekin_tot_dir(self,i):
        return self.get_ekin_tot(iv=i)

    def get_ekin_tot(self,iv=None,idir=None):
        """kinetic energy summed along specific direction

        iv:   gives total kinetic energy along iv axis
        idir: selects kinetic energy along (iv must be None):

           2D grid:
            radial: idir == 0
            azimuthal (phi): idir == 1

           3D grid:
            radial: idir == 0
            horizontal (theta): idir == 1
            azimuthal (phi): idir == 2

        Setting both to None will return total kinetic energy
        """
        if iv is not None and idir is not None :
            raise IndexError('get_ekin_etot: iv and idir set at the same time')
        elif idir is not None:
            #projects along idir in spherical coordinates
            vel = self.vel(idir=idir)
        elif iv is not None:
            #selects velcocity of axis iv
            vel = self.vel(ivel=iv)

        else:
            #do not project at all
            vel = self.vel()

        return 0.5 * (self.rho() * self.vol() * vel**2).sum()

    def Hfluc(self,val,axis=1):
        """relative horizontal fluctuations of val

        horizontal lines must be aligned with the given axis
        """
        avg = self.Havg(val,axis)
        return (val-avg)/avg

    def Hrhofluc(self, **kwargs):
        """relative horizontal density fluctuations

        horizontal lines must be aligned with the given axis
        """
        return self.Hfluc(self.rho(), **kwargs)

    def RMSfluc(self,val):
        """rms fluctuations of val

        horizontal lines must be aligned with the given axis
        """
        avg = self.Havg(val,self.vaxis)
        return np.sqrt(self.Havgp( (val - avg)**2 ))

    def energy_flux(self, axis=1):
        return self.vel(axis)*(self.eps() + self.pres())

    def dEdt_nuc(self):
        logger.warning("This function has been merged into dedt_nuc and will be removed at some point in the future. Please update your scripts.")
        if 'postprocess%dedt_nuc' in self.keys():
            return np.squeeze(self['postprocess%dedt_nuc'])
        else:
            logger.warning('gridfile does not contain dedt_nuc data and must be postprocessed')

    def fft2d(self,data):
        import numpy.fft.fftpack as f
        warnings.warn("fft2d is deprecated")
        return f.fft(f.fft(data,axis=0),axis=1)

    def fftspec(self,data):
        """multi-dimensional discrete Fourier transform of data"""
        import numpy.fft.fftpack as f

        res=data.copy()
        for i in range(data.ndim):
            res=f.fft(res,axis=i)

        return res

    def diff1d(self,data,dx):

        d = np.zeros(data.shape)
        d[1:-1] = 0.5*(data[2:]-data[:-2])/dx

        d[0] = (-1.5*data[0]+2.0*data[1]-0.5*data[2])/dx
        d[-1] = (1.5*data[-1]-2.0*data[-2]+0.5*data[-3])/dx

        return d

    def diff(self,data,axis):
        """compute the spatial derivative of data along a given axis"""

        if self.geometry.type!='cartesian':
            raise Exception('only Cartesian geometry supported yet')

        dx = self.geometry.dx[axis]
        return np.apply_along_axis(self.diff1d,axis,data,dx)


    def powerspec(self,data):
        """multi-dimensional power spectrum of data"""

        if data.ndim<1 or data.ndim>3:
            raise Exception("dimension of data must be 1,2 or 3")

        n = data.shape
        for i in range(data.ndim):
            if data.shape[i]%2 != 0:
                raise Exception("length of each dimension must be even")

        pdata0 = np.abs(self.fftspec(data))**2

        if data.ndim==1:
            pdata1 = pdata0[:n[0]/2+1].copy()
            pdata1[1:-1] += pdata0[-1:-n[0]/2:-1]

            return pdata1

        elif data.ndim==2:
            pdata1 = pdata0[:n[0]/2+1,:].copy()
            pdata1[1:-1,:] += pdata0[-1:-n[0]/2:-1,:]

            pdata2 = pdata1[:,:n[1]/2+1].copy()
            pdata2[:,1:-1] += pdata1[:,-1:-n[1]/2:-1]

            return pdata2

        elif data.ndim==3:
            pdata1 = pdata0[:n[0]/2+1,:,:].copy()
            pdata1[1:-1,:,:] += pdata0[-1:-n[0]/2:-1,:,:]

            pdata2 = pdata1[:,:n[1]/2+1,:].copy()
            pdata2[:,1:-1,:] += pdata1[:,-1:-n[1]/2:-1,:]

            pdata3 = pdata1[:,:,:n[1]/2+1].copy()
            pdata3[:,:,1:-1] += pdata2[:,:,-1:-n[1]/2:-1]

            return pdata3


    def HFrhofluc(self,dx=1,dy=1,**kwargs):
        """high frequency horizontal density fluctuations"""
        if self.sdim!=2:
            raise Exception("wrong spatial dimension")

        d = np.abs(self.fftspec(self.Hrhofluc(**kwargs)))

        nx=self.gnc[0]
        ny=self.gnc[1]

        return d[nx/2-dx:nx/2+dx+1,ny/2-dy:ny/2+dy+1].sum() + \
            d[nx/2-dx:nx/2+dx+1,0:dy+1].sum() + \
            d[0:dx+1,ny/2-dy:ny/2+dy+1].sum()

    def xnuc(self, species=None):
        adsp = False
        if self.eos.nucdata.shape[0] == 0:
            # no species information
            return np.array([])
        elif self.eos.nucdata.shape[0] < self.nps:
            adsp = True
        if self.eos.passive_species:
            if adsp:
                xnuc = self.ps()[:-1,:]
            else:
                xnuc = self.ps()[:,:]
        elif self.eos.as2 < self.eos.as1:
            # Treat this case explicitly so we can directly use the mmaped array.
            # This is much faster and more memory efficient.
            xnuc = self.ps()[self.eos.ps1-1:self.eos.ps2,:]
        elif self.eos.ps2 < self.eos.ps1:
            # see above
            xnuc = self.asc()[self.eos.as1 - (self.vdim - self.nas) - 1:self.eos.as2 - (self.vdim - self.nas) + 1,:]
        else:
            # most general case
            xnuc = np.zeros([self.eos.nspecies] + list(self.rho().shape), dtype=self.rho().dtype)
            if self.eos.as2 >= self.eos.as1:
                xnuc[:self.nas,:] = self.asc()[self.eos.as1 - (self.vdim - self.nas) - 1:self.eos.as2 - (self.vdim - self.nas) + 1,:]
            if self.eos.ps2 >= self.eos.ps1:
                xnuc[self.eos.as2-self.eos.as1+1:,:] = self.ps()[self.eos.ps1-1:self.eos.ps2,:]
        if species is not None:
            selection = self.eos.nucdata['name'] == species
            if not np.any(selection):
                raise ValueError("species (%s) not found" % species)
            return np.squeeze(xnuc[selection, :])
        else:
            return np.squeeze(xnuc)

    def metallicity(self):
        #if 'p' in g.eos.nucdata['name']:
            #TODO make if clause and account for other helium isotopes
        return self.xnuc().sum(axis=0)-self.xnuc('p')-self.xnuc('he4')

    def abar(self):
        if self.eos.type == 'ideal' or len(self.eos.nucdata['A']) == 0:
            return np.ones_like(self.rho()) * self.eos.mu
        elif self.eos.aybar_mode == 1:
            if self.eos.as2 < self.eos.as1:
                return 1.0 / self.ps(self.eos.ps1-1)
            elif self.eos.ps2 < self.eos.ps1:
                return 1.0 / self.asc(0)
            else:
                raise ValueError("invalid species configuration for aybar_mode")
        else:
            return 1.0 / np.tensordot(1.0 / self.eos.nucdata['A'], self.xnuc(), axes=[0,0])

    def zbar(self):
        if self.eos.type == 'ideal' or len(self.eos.nucdata['A']) == 0:
            return np.zeros_like(self.rho())
        elif self.eos.aybar_mode == 1:
            if self.eos.as2 < self.eos.as1:
                if self.eos.ps2 == self.eos.ps1 + 1:
                    return self.ps(self.eos.ps2-1) * self.abar()
                else:
                    return np.zeros_like(self.rho())
            elif self.eos.ps2 < self.eos.ps1:
                if self.eos.as2 == self.eos.as1 + 1:
                    return self.asc(1) * self.abar()
                else:
                    return np.zeros_like(self.rho())
            else:
                raise ValueError("invalid species configuration for aybar_mode")
        else:
            return self.abar() *  np.tensordot(self.eos.nucdata['Z'] / self.eos.nucdata['A'], self.xnuc(), axes=[0,0])

    def qnuc(self, n):
        """returns the binding energy of all nuclei
        needs an instance of a pyyann network with initialed mass and partfile
        e.g. n = pyyann.network('/dev/null', '/dev/null', partfile='data/part.txt', massfile='data/mass.txt')"""
        import pyyann
        if not isinstance(n, pyyann.Network):
            raise TypeError("Argument must be an instance of pyyann.Network.")
        n.set_species(self.eos.nucdata.astype([('name', '5str'), ('A', '<i4'), ('Z', '<i4')]))
        xnuc = self.xnuc()
        qnuc = np.zeros_like(xnuc)
        names = self.eos.nucdata['name']
        conv = 1.6021765e-09 * 6.0221418e+23 # mass excess is in keV per atom
        for i, name in enumerate(names):
            qnuc[i,:] = n.species_dic[name].exm * xnuc[i,:] / self.eos.nucdata[i]['A']
        qnuc *= conv

        return qnuc

    def getnetwork(self, c_mod=True):
        """returns a pyyann network instance using the species from the grid file"""
        if c_mod:
            nucdata = self.eos.nucdata
            name    = nucdata['name']
            A       = nucdata['A']
            Z       = nucdata['Z']
            species = [(str(n), int(a), int(z)) for n, a, z in zip(name, A, Z)]
            import _pyyann
            return _pyyann.Network(species, 'data/reaclib20090121', 'data/part.txt', 'data/mass.txt', 'data/lmp_weak_rates.txt')
        else:
            import pyyann
            network = pyyann.network('data/reaclib20090121', 'data/lmp_weak_rates.txt', partfile='data/part.txt', massfile='data/mass.txt')
            network.set_species(self.eos.nucdata)
            return network


    def dedt_nuc(self, reaction_string=None, network=None, dt=0.01, xnucmin=1e-20, havg=False, xnuc_opt=None):
        """ Nuclear energy release rate

            The energy rate is computed with the nuclear reaction network YANN, via the python interface pyyann.
            Units are erg/s/cm**3. This function has two modes:

            * Standard : reaction_string :: `None`
                Returns the total energy rate if present in the gridfile
                Else the reaction network is integrated over the time dt

            * Specific reaction : reaction_string :: `a_valid_reaction_string`
                The energy rate for a specified reaction in the network

            Parameters
            ----------
            reaction_string : string, optional
                If `None`, the 'standard' mode is used
                The string must be a pyyann.Rate.reaction_string
            network : a pyyann network object, optional
            dt : float, optional
                The time over which the network is integrated
            xnucmin : float, optional
                lower boundary for the species abundances to remove negative values

            See also
            --------

        """
        if reaction_string is None:

            if 'postprocess%dedt_nuc' in self.keys():
                return np.squeeze(self['postprocess%dedt_nuc'])
            else:
                if network is None:
                    # The network instance is of type _pyyann.Network
                    network = self.getnetwork()
                if xnuc_opt is None:
                    xnuc = np.fmin(1.0, np.fmax(xnucmin, self.xnuc()))
                else:
                    xnuc = np.fmin(1.0, np.fmax(xnucmin, xnuc_opt))
                xnuc /= xnuc.sum(axis=0)[None]

                if not havg:
                    rho     = self.rho()
                    temp    = self.temp()
                else:
                    rho     = self.Havgp(self.rho())
                    temp    = self.Havgp(self.temp())
                    xnuc    = np.array([self.Havgp(x) for x in xnuc])
                arr = np.concatenate((rho[None], temp[None], xnuc), axis=0)

                def integrate(arg):
                    # network.integrate returns rate in erg/g/s. Multiplication by rho makes erg/cm**3/s
                    return network.integrate(arg[0], arg[1], arg[2:], dt)[0] * arg[0]

                return np.apply_along_axis(integrate, 0, arr)

        else:

            if 'postprocess%dedt_nuc%'+reaction_string in self.keys():
                return np.squeeze(self['postprocess%dedt_nuc%'+reaction_string])
            elif 'postprocess%dedt_nuc:'+reaction_string in self.keys():
                return np.squeeze(self['postprocess%dedt_nuc:'+reaction_string])
            else:
                if network is None:
                    # network.integrate returns rate in erg/g/s. Multiplication by rho makes erg/cm**3/s
                    network = self.getnetwork(c_mod=False)
                dedt = network.dedt_nuc(reaction_string, self.rho(), self.temp(), self.xnuc()) * self.rho()
                return dedt


    def dxdt_nuc(self, species, network=None, dt=0.01, havg=False, xnucmin=1e-20, xnuc_opt=None):
        """compute rate of change in mass fraction of species per volume in each cell by running the nuclear network for the time dt. xnucmin is a lower boundary for the species abundances to remove negative values."""

        speciesind = np.nonzero(self.eos.nucdata['name'] == species)[0]
        if len(speciesind) != 1:
            raise ValueError('species "%s" not found' % species)
        speciesind = speciesind[0]
        if network is None:
            network = self.getnetwork()
        if xnuc_opt is None:
            xnuc = np.fmin(1.0, np.fmax(xnucmin, self.xnuc()))
        else:
            xnuc = np.fmin(1.0, np.fmax(xnucmin, xnuc_opt))
        xnuc /= xnuc.sum(axis=0)[None]

        def integrate(arg):
            return network.integrate(arg[0], arg[1], arg[2:], dt)[1][speciesind]

        if not havg:
            rho     = self.rho()
            temp    = self.temp()
        else:
            rho     = self.Havgp(self.rho())
            temp    = self.Havgp(self.temp())
            xnuc    = np.array([self.Havgp(x) for x in xnuc])
        arr = np.concatenate((rho[None], temp[None], xnuc), axis=0)
        return np.apply_along_axis(integrate, 0, arr)

    def hdf5_export(self,filename=None):
        import h5py as h
        if filename is None:
            fn=self.filename.replace('.slh','.h5')
        else:
            fn=filename
        f=h.File(fn,mode='w')
        g=f.create_group('grid')

        g.attrs.create('time',self.time)
        g.attrs.create('step',self.step)

        geo=g.create_group('geometry')
        geo.attrs.create('type', str(self.geometry.type))
        geo.attrs.create('alpha', self.geometry.alpha)
        geo.create_dataset('x1',data=self.geometry.x1)
        geo.create_dataset('x2',data=self.geometry.x2)
        geo.create_dataset('dx',data=self.geometry.dx)

        g.create_dataset('gnc',data=self.gnc)

        fields = {'grid%rho%data':'rho', 'grid%pres%data':'pressure', 'grid%vel%data':'velocity',\
                      'grid%eps%data':'eps','grid%cspeed%data':'cspeed','grid%temp%data':'temperature'}

        for key in fields:
            if key in self.ident:
                g.create_dataset(fields[key],data=self[key].T)

        key = 'grid%ps%data'
        if key in self.ident:
            for i in range(self[key].shape[0]):
                g.create_dataset('ps.%d' % i, data=self[key][i:i+1,:,:,:].T)
        key = 'grid%as%data'
        if key in self.ident:
            for i in range(self[key].shape[0]):
                g.create_dataset('ps.%d' % i, data=self[key][i:i+1,:,:,:].T)


        f.close()

    def griddata(self, vals, xi, **kwargs):
        """interpolates vals at the points xi
        kwargs are passed to scipy.interpolate.griddata"""
        import scipy.interpolate

        points = np.array([self.coords(i).flatten() for i in range(self.sdim)]).T
        xi = np.rollaxis(xi,0,xi.ndim)
        return scipy.interpolate.griddata(points, vals.flatten(), xi, **kwargs)

    def take(self, data, rmin=None, rmax=None):
        """ Like np.take but takes *physical* radial values instead of indices
            Useful for analysis of values in specific regions """

        if rmin is None:
            raise ValueError('rmin missing')
        imin = abs(self.Havgp(self.radius())-rmin).argmin()

        if rmax is not None:
            imax = abs(self.Havgp(self.radius())-rmax).argmin()

        datadim = len(data.shape)
        if datadim == 1:
            imax = len(data)
            return data.take(range(imin,imax))
        else:
            imax = data.shape[self.vaxis]
            return data.take(range(imin,imax), axis=self.vaxis)
            #vshape = data.shape[self.vaxis]

    def ind_twopeaks(self, data, mindex=None, vaxis=0):
        '''index of highest total peaks in two regions of data
        The two regions result from splitting data at mindex in vertical
        direction

        Example for spherical grid:
            sdata = stelo.grad(g.r, g.abar())

            val_at_twopeaks(val, sdata, len(val)//2, vaxis=0) returns
            two arrays of shape (self.gnc[1], self.gnc[2]) with all
            indexes aling vaxis for the highest value of the abundance
            gradient

            if lowest value is wanted, just multiply by -1.
            if absolute value is wanted, just pass the absolute values
            as input

            can be used to find lower and upper boundaries, e.g. of a
            convection zone by composition or velocity gradients
        '''

        if isinstance(data, str):
            r = self.r
            if mindex is None:
                mindex = self.gnc[0]//2
            if vaxis is None:
                vaxis = 0

            axis = list(range(self.sdim))
            del(axis[0])

            if data == 'grad_abar' or data is None:
                data = np.abs(np.gradient(self.abar(), r, axis=0))

            elif data == 'grad_vhor':
                data = np.abs(np.mean(np.gradient(
                        self.vel(idir=0, orthogonal=True), r, axis=0),
                            axis=tuple(axis)))
            elif data == 'grad_ps':
                data = np.abs(np.gradient(self.ps()[-1], r, axis=0))
            else:
                raise RuntimeError(
                    'ind_twopeaks, {} mode not recognizes' % data)

        sl1 = [slice(None)] * data.ndim
        sl1[vaxis] = slice(None, mindex)
        sl2 = [slice(None)] * data.ndim
        sl2[vaxis] = slice(mindex, None)

        p1 = np.argmax(data[tuple(sl1)], axis=vaxis)
        p2 = np.argmax(data[tuple(sl2)], axis=vaxis) + mindex

        return p1, p2

    def plot_boundaries(self, ax, mode=None, idir=1, mean=True, **kwargs):
        r = self.r

        if mode is None:
            mode = 'grad_abar'

        data = self.ind_twopeaks(mode, mindex=None, vaxis=0)

        lb = r[data[0]]
        ub = r[data[1]]

        if self.sdim == 2:
            ph = self.phi()[0, :]
        else:
            if idir == 1:
                ph = self.phi()[0, 0, :]
                axis = 1
            elif idir == 2:
                ph = self.theta()[0, :, 0]
                axis = 0
            else:
                raise RuntimeError('idir %d not valid' % idir)

            if mean is True:
                lb = np.mean(lb, axis=axis)
                ub = np.mean(ub, axis=axis)
            else:
                if idir == 1:
                    lb = lb[mean, :]
                    ub = ub[mean, :]
                else:
                    lb = lb[:, mean]
                    ub = ub[:, mean]

        if idir == 1:
            ax.plot(lb*np.cos(ph), lb*np.sin(ph), **kwargs)
            ax.plot(ub*np.cos(ph), ub*np.sin(ph), **kwargs)
        else:
            ax.plot(lb*np.sin(ph), lb*np.cos(ph), **kwargs)
            ax.plot(ub*np.sin(ph), ub*np.cos(ph), **kwargs)

    def mean_std_twopeakspos(self, data, coords, mindex=None, vaxis=None):
        '''return mean and standard deviation of peaks at
        region divided by mindex
        '''

        p1, p2 = self.ind_twopeaks(data, mindex, vaxis)

        mean1 = np.mean(coords[p1])
        std1 = np.std(coords[p1])

        mean2 = np.mean(coords[p2])
        std2 = np.std(coords[p2])

        return (mean1, std1), (mean2, std2)

    def get_bulk_richardson(self, mode=None, **kwargs):
        import stelo.model_reader as mr

        r = self.r
        gx, gy, gz = self.grav()[:, :,  0, 0]
        x, y, z = self.coords()[:, :, 0, 0]
        grav = (gx*x + gy*y + gz*z) / np.sqrt(x**2 + y**2 + z**2)
        l = mr.slh_model(self, radial=True, vaxis=0, grav=grav)
        data = self.mean_std_twopeakspos(data=mode, coords=r)

        lmean, lstd = data[0]
        umean, ustd = data[1]

        inds = [np.argmin(np.abs(self.r - r)) for r in [lmean, umean]]
        sl = slice(*inds)

        vsq = np.mean(self.vel(idir=0)[sl]**2)
        #vsq = np.sum((self.vol()*self.vel(idir=0)**2)[sl])/np.sum(self.vol()[sl])

        ubulk = l.richardson_bulk(umean, vsq=vsq, **kwargs)
        lbulk = l.richardson_bulk(lmean, vsq=vsq, **kwargs)

        return lbulk, ubulk, vsq



class slhdefect(slhoutput):
    def __init__(self, step, stage, it, name='defect'):
        """opens the defect for given timestep, stage, and iteration"""
        filename = "%s_n%05d_s%02d_i%03d.slh" % (name, step, stage, it)
        super(slhdefect,self).__init__(filename)

        self.v = self['tvector%data']

class slhhistory(slhoutput):

    def __init__(self,filename):

        super(slhhistory,self).__init__(filename)

        self.stepmin = 2000000000
        self.stepmax = 0

        for ident in self.ident:
            step = re.match(r'n([0-9]+)_*',ident)
            if step is not None:
                step = int(step.group(1))
                self.stepmin = min(step,self.stepmin)
                self.stepmax = max(step,self.stepmax)
            else:
                raise AttributeError('slhhistory, unknown ident %s'%(ident))

        self._maxtry = {}

    def analyze(self,stalled=True):

        for ident in self.ident:

            #step = int(ident[1:6])

            if '_newton_' in ident or '_newtonps_' in ident:
                crit = self[ident]['stoppingcrit']
                #if crit!='tolest' and (crit!='stalled' and stalled):
                if crit!='tolest' and ((crit!='stalled') ^ stalled):
                    nident = ident.replace('_newton_','_newtondef_')
                    nident = nident.replace('_newtonps_','_newtonpsdef_')
                    print('%s: stopping criterion: %s' % (ident,crit))
                    print(self[nident][:,-1])


    def find_max_try(self,ident,istep):
        try:
            return self._maxtry[ident][istep]
        except KeyError:
            self._maxtry[ident] = {}
            expr = re.compile(r'n([0-9]+)_%s_t([0-9]{2})' % ident)
            for i in self.keys():
                m = expr.match(i)
                if m:
                    ind = int(m.group(1))
                    self._maxtry[ident][ind] = max(int(m.group(2)),self._maxtry[ident].get(ind,-1))
            return self._maxtry[ident][istep]

    def get_newton_def(self,ident='newtondef_s01',iv=0,nv=1,iter=-1):

        defect = np.zeros([nv,self.stepmax-self.stepmin])

        for istep in range(self.stepmin,self.stepmax):
            defect[:,istep-self.stepmin] = self['n%0.5d_%s' % (istep,ident)][iv:iv+nv,iter]

        return defect

    def get_stepper_time(self):
        time = np.zeros([self.stepmax-self.stepmin+1])

        for istep in range(self.stepmin,self.stepmax+1):

            maxtry = self.find_max_try('stepper',istep)
            stepper = self['n%0.5d_stepper_t%0.2d' % (istep,maxtry)]

            time[istep-self.stepmin] = stepper['time']

        return time


    def get_stepper_err(self,ident):
        """get array of error estimates from time stepper
        ident may be state_rel,state_max,ps_rel,ps_max
        """

        maxtry0=self.find_max_try('stepper_err_'+ident,self.stepmin)
        err0 = self['n%0.5d_stepper_err_%s_t%0.2d' % (self.stepmin,ident,maxtry0)]
        vdim, = np.shape(err0)

        err = np.zeros([vdim,self.stepmax-self.stepmin+1])

        for istep in range(self.stepmin,self.stepmax+1):

            maxtry = self.find_max_try('stepper_err_'+ident,istep)
            erristep = self['n%0.5d_stepper_err_%s_t%0.2d' % (istep,ident,maxtry)]

            err[:,istep-self.stepmin] = erristep[:]

        return err

    def plot_stepper_err(self,ident,iv,nv=1,**kwargs):
        self.time = self.get_stepper_time()
        self.err = self.get_stepper_err(ident)

        plt.clf()
        for i in range(iv,iv+nv):
            plt.plot(self.time,np.log10(self.err[i,:]),label=('%d' % i),**kwargs)
        plt.legend()

    def get_stepper_timestep(self):
        time = np.zeros([self.stepmax-self.stepmin+1])

        for istep in range(self.stepmin,self.stepmax+1):

            maxtry = self.find_max_try('stepper',istep)
            stepper = self['n%0.5d_stepper_t%0.2d' % (istep,maxtry)]
            try:
                time[istep-self.stepmin] = stepper['timestep']
            except Keyerror:
                pass


        return time

    def integral(self):
        try:
            return self._integral
        except AttributeError:
            keys = self['n%0.5d_integral_quantities' % self.stepmin].keys()
            self._integral = dict([(k, np.zeros([self.stepmax-self.stepmin+1])) for k in keys])
            for istep in range(self.stepmin,self.stepmax+1):
                q = self['n%0.5d_integral_quantities' % istep]
                for k, v in q.items():
                    self._integral[k][istep-self.stepmin] = v
            return self._integral

    def ekin(self,iv=None):
        nam = ('ekinx', 'ekiny', 'ekinz')
        i = self.integral()
        if iv is None:
            return i['ekinx'] + i['ekiny'] + i['ekinz']
        else:
            return i[nam[iv]]

    def mass(self):
        return self.integral()['mass']

    def etot(self):
        return self.integral()['etot']

class slhsequence(object):
    """Plot a single value from a sequence of grids

    Computed values are cached!

    Example:

    # create new instance, constructor calls s.find_files()
    # to find all grid_n?????.slh
    s=slhsequence()

    # plot mean Mach number
    s.plot("g.mach().mean()",color="red")

    # update the list of files
    s.find_files()
    s.plot("g.mach().mean()",color="red")

    # set the filelist manually
    s.set_range(20,100,5)
    s.plot("g.mach().max()",color="blue")

    """

    def __init__(self,dir='.', auto_cache=True, gridclass=slhgrid, **kwargs):

        self.dir = os.path.abspath(dir)

        self.filenumbers=[]

        self.cached = True

        self.auto_cache = auto_cache
        self.cache_filename = 'slhsequence.cache'

        self.gridclass = gridclass

        self.allplots = []

        self.clear_cache(delete_file=False)

        self.find_files()

        self.sdim = self.g0.sdim


        if self.auto_cache and os.path.isfile(os.path.join(self.dir,self.cache_filename)):
            try:
                self.load_cache()
            except Exception as e:
                logger.warning("could not load cache file (%s)" % str(e))

        super(slhsequence, self).__init__(**kwargs)

    @staticmethod
    def savecachecrit(expr):
        """returns whether an expression should be cached"""
        m = re.match(r'g.\w+\([^)]*\)$', expr.strip())
        # Do not cache a value if is a basic expression that is better loaded via mmapping
        return m is None

    def __del__(self):
        if self.auto_cache:
            self.save_cache()

    def clear_cache(self, delete_file=True):
        self.cache = {}
        fname = os.path.join(self.dir,self.cache_filename)
        if delete_file and os.path.isfile(fname):
            os.remove(fname)

    def save_cache(self):
        try:
            import cPickle as p
        except ImportError:
            import pickle as p

        if len(self.cache) > 0:
            logger.info('writing cache file '+os.path.join(self.dir,self.cache_filename))
            # Remove values from cache that don't fulfill the criterion.
            cache_to_save = {}
            for k in self.cache:
                if self.savecachecrit(k):
                    cache_to_save[k] = self.cache[k]
            with open(os.path.join(self.dir,self.cache_filename),'a+b') as fh:
                fcntl.lockf(fh.fileno(), fcntl.LOCK_EX)
                fh.seek(0)
                # Load old cache from file for merging.
                try:
                    old_cache = p.load(fh)
                except EOFError:
                    old_cache = {}
                # Loop for merging time steps for one expression.
                for k in old_cache:
                    old_cache[k].update(cache_to_save.get(k,{}))
                    cache_to_save[k] = old_cache[k]
                fh.truncate(0)
                p.dump(cache_to_save, fh, protocol=2)

    def load_cache(self):
        try:
            import cPickle as p
        except ImportError:
            import pickle as p

        logger.info('reading cache file '+os.path.join(self.dir,self.cache_filename))
        with open(os.path.join(self.dir,self.cache_filename),'rb') as fh:
            fcntl.lockf(fh.fileno(), fcntl.LOCK_SH)
            self.cache = p.load(fh)

    def set_range(self,i1,i2,istep):
        self.i1 = i1
        self.i2 = i2
        self.istep = istep
        self.filenumbers = list(range(i1,i2+1,istep))

    def takespread(self, steps):
        """trim down the number of gridfiles, to get a handle of a too large number
           steps: number of steps to retain"""
        self.find_files()
        length = float(len(self.filenumbers))
        if length <= steps:
            return
        indices = list((np.floor([(length-1)/(steps-1)*i for i in range(steps)])).astype(int))
        fn = np.copy(self.filenumbers)
        fn = fn[indices]
        self.filenumbers = tuple(fn)

    def set_time(self, times):
        """Sets the sequence to use grid files closest to the times given.
        If times is scalar, it will be used as temporal spacing."""
        self.find_files()
        if np.isscalar(times):
            t = self.get_time()
            times = np.arange(t.min(), t.max(), times)
        self.filenumbers = [self.filenumbers[i] for i in map(self.nearest_snapshot, times)]

    def iterate(self,steps):
        self.takespread(steps)
        gridfiles = [slhgrid(i) for i in self.filenumbers]
        return iter(gridfiles)

    def start_at_time(self, tmin, takespread=None):
        '''Sets the sequence to only use grid files with g.time > tmin

           takespread (int): uses self.takespread to reduce number of considered files
        '''
        times = self.get_time(takespread=takespread)
        indext = np.argmin(abs(times-tmin))
        if takespread is None:
            self.find_files()
        else:
            self.takespread(takespread)
        self.filenumbers = self.filenumbers[indext:]

    def between_time(self, tmin, tmax, nfiles=None):
        """Sets the sequence to only use grid files with tmin < g.time < tmax

           nfiles (int): reduce number of files to nfiles
        """
        if tmax > self.g1.time:
            warnings.warn('tmax is larger than total simulation time')

        # index of snapshot nearest to tmax
        itmax = self.nearest_snap(tmax)
        itmin = self.nearest_snap(tmin)

        # load all available grid files. Resets any former self.takespreads
        self.find_files()
        # only consider files up to tmax
        filenumbers = self.filenumbers[itmin:itmax]
        # number of files which are left
        nfnumbers = len(filenumbers)
        if nfiles is not None:
            if nfiles > nfnumbers:
                print(
                    'Warning,  nfiles set from {} '.format(nfiles) +
                    'to number of available files of {}.'.format(nfnumbers))
                nfiles = nfnumbers
            dt = np.floor(nfnumbers/nfiles)
            sl = slice(None, None, int(dt))
            filenumbers = filenumbers[sl]

        self.filenumbers = tuple(filenumbers)

    def find_files(self):
        files = self.gridclass.file_list(prefix=self.dir)
        if len(files) > 0:
            self.filenumbers, self.filenames = zip(*files)
        else:
            self.filenames = []
            self.filenumbers = []
            warnings.warn('No gridfiles found in {}.'.format(self.dir))

    def get_time(self,takespread=None):
        """Get times for all gridfiles of slhsequence

        takespread: int
                    Reduce number of considered gridfiles to int.
        """
        try:
            if len(self.time) == len(self.filenumbers):
                return self.time
        except:
            pass

        if takespread is not None:
            self.takespread(takespread)

        time = []

        cached = self.cached and ("<time>" in self.cache)
        if self.cached and ("<time>" not in self.cache):
            self.cache["<time>"] ={}

        if use_progressbar:
            bar = Bar('Extracting time', max=len(self.filenumbers), suffix="%(index)d/%(max)d (%(eta_td)s)")

        for i in self.filenumbers:

            from_cache = False
            if cached:
                if i in self.cache["<time>"]:
                    t = self.cache["<time>"][i]
                    from_cache = True

            if not from_cache:
                g=self.gridclass(i,prefix=self.dir)
                t = g.time
                if self.cached:
                    self.cache["<time>"][i] = t

            time.append(t)

            if use_progressbar:
                bar.next()

        if use_progressbar:
            bar.finish()

        if not takespread:
            self.time = np.array(time)
            return self.time
        else:
            return np.array(time)

    def nearest_snap(self,t):
        ''' primitive bisection algorithm to find snapshot nearest to t

        faster than first looping through all grid files
        '''

        self.find_files()
        a = 0
        b = float(len(self.filenumbers))
        im = int((a+b)/2)
        while((a != im) and (b != im)):
            tmp = self.g(im).time
            if tmp<t:
                a = im
            else:
                b = im
            im = int((a+b)/2)
        return im

    def nearest_snapshot(self, t):
        time = self.get_time()
        return int(abs(time-t).argmin())

    def get_data_parallel(self, expr, dt=None, addlocals=None, nprocs=None):
        """ For each gridfile in self.filenumbers, evaluate the string *expr*.
        """

        if nprocs is None:
            nprocs = 1

        dtidxs = slice(None)
        if dt is not None:
            dtidxs = []
            t, times = self.get_data_parallel(
                'g.time', addlocals=addlocals, nprocs=nprocs)

            tlast = 0
            for ind, t in enumerate(times):
                if (t-tlast) - dt > 0:
                    dtidxs.append(ind)
                    tlast = t

        @vectorize_parallel(
            method='processes', num_procs=nprocs, use_progressbar=True,
            label='load data from sequence')
        def eval_expr(i, expr):
            g = self.gridclass(i, prefix=self.dir)
            t = g.time
            loc = locals()
            if addlocals is not None:
                loc.update(addlocals)
            d = eval(expr, globals(), loc)
            return g.time, d

        res = eval_expr(np.array(self.filenumbers)[dtidxs], expr)
        t = []
        data = []

        tlast = 0
        for tt, dat in res:
            t.append(tt)
            data.append(dat)
        return np.array(t), np.array(data)

    def get_data(self,expr, addlocals=None, addlocals_iterative=None):
        """
            For each gridfile in self.filenumbers, evaluate the string *expr*.

            Parameters
            ----------
            expr : string
                Expression applied to each gridfile, e.g. g.mach().max()
            addlocals : optional
                Pass a local variable for the evalation of *expr*
            addlocals_iterative : dictionary, optional
                Must have the form {*varname*: *list_of_values*},
                where len(list_of_values) = len(self.filenumbers).
                varname* is passed a value in each iteration of eval("some_funcion(*varname*)")

            Returns
            -------
            time, data: arrays
                time:   g.time for all g in s
                data:   contains oper(eval(expr))

            Examples
            --------
                time, Tmax = s.get_data('g.Temp().max()')
                t, rho = s.get_data('g.Havgp(g.rho())[idx]', addlocals_iterative={'idx': Tindices})

            See Also
            --------
            self.get_time
        """

        time = []
        data = []

        cached = self.cached and expr in self.cache
        if self.cached and expr not in self.cache:
            self.cache[expr] = {}

        if self.cached and ("<time>" not in self.cache):
            self.cache["<time>"] ={}

        try:
            g0 = self.gridclass(0, prefix=self.dir)
        except:
            logger.warning('No grid 0 found in %s' % self.dir)

        if addlocals_iterative is None:
            iterative_locals = [None]*len(self.filenumbers)
        else:
        #TODO implement for more than one iterable
            varname = addlocals_iterative.keys()[0]
            iterative_locals = addlocals_iterative[varname]
            if len(iterative_locals) != len(self.filenumbers):
                raise RuntimeError('addlocals_iterative[1] must have the same lenght as s.filenumbers')

        if use_progressbar:
            bar = Bar('Extracting data', max=len(self.filenumbers), suffix="%(index)d/%(max)d (%(eta_td)s)")
        for i, iterloc in zip(self.filenumbers, iterative_locals):

            from_cache = False

            if cached:
                if i in self.cache[expr]:
                    t = self.cache[expr][i][0]
                    d = self.cache[expr][i][1]

                    from_cache = True

            if not from_cache:
                logger.info('data for gridfile %s not in cache, calculating...' % i)
                g=self.gridclass(i,prefix=self.dir)
                t=g.time
                loc = locals()
                if addlocals:
                    loc.update(addlocals)
                if addlocals_iterative is not None:
                    iterloc={varname:iterloc}
                    loc.update(iterloc)
                    logger.info("%s %s" % (str(i), str(iterloc)))
                d = eval(expr, globals(), loc)

                if self.cached:
                    self.cache[expr][i] = [t,d]

            if self.cached:
                self.cache["<time>"][i] = t
                time.append(t)

            data.append(d)

            if use_progressbar:
                bar.next()

        if use_progressbar:
            bar.finish()

        time = np.array(time)
        data = np.array(data)
        return time, data

    def time_average(self, expr, dx=3e7, v_conv=1e7,**kwargs):
        t_conv = dx/v_conv
        self.find_files()
        t, data = self.get_data(expr, **kwargs)
        fn = self.filenumbers
        div = int((t[-1]-t[0])/t_conv)
        I, J = [0]*div, [0]*div
        for m in range(div):
            I[m] = np.where((t[0]+m*t_conv)<t)[0][0]
            J[m] = np.where((t[0]+(m+1)*t_conv)<t)[0][0]
        drms = []
        dmin, dmax = [], []

        for i, j in zip(I, J):
            Data = data[i:j]
            drms.append(rms([rms(d) for d in Data]))
            dmin.append(np.min([np.min(d) for d in Data]))
            dmax.append(np.max([np.max(d) for d in Data]))
        t = t[J]
        return t, drms, dmin, dmax

    def HavgStats(self, expr, t0, t1, XScale=1., Draw=True, Clear=True, **kwargs):
        """ Plot the horizontal average of a quantity `expr` averaged over a
            selected timestep ranges `t1`, `t2`, and show extreme values, averaged over
            the same period.
        """
        self.find_files()
        self.filenumbers = self.filenumbers[t0:t1+1]
        t, data = self.get_data(expr)
        dmean = [] ; dmin = [] ; dmax = []
        va = self.g0.vaxis
        nr = self.g0.nr
        # process each timestep
        for d in data:
            dmean.append( Havg(d, va, repeat=False).reshape((nr,)) )
            dmin.append(  Hmin(d, va, repeat=False).reshape((nr,)) )
            dmax.append(  Hmax(d, va, repeat=False).reshape((nr,)) )
        # time-average
        dmean = np.mean(dmean, axis=0)
        dmin  = np.min(dmin, axis=0)
        dmax  = np.max(dmax, axis=0)

        if Draw:
            if Clear:
                plt.clf()
            r = self.g0.r * XScale
            plt.plot(r, dmean, color='k', lw=1.5)
            plt.fill_between(r, dmin, dmax, color='0.8')
            plt.draw_if_interactive()

        self.find_files()
        return dmean, dmin, dmax

    def get_tdiff(self,expr,**kwargs):
        time,data = self.get_data(expr,**kwargs)
        return time[:-1],(data[1:]-data[:-1])/(time[1:]-time[:-1])

    def postprocess(self, name, expr, addlocals=None, overwrite=False, parallel=True, processes=None):
        """Adds the result of evaluating expr to each grid file in the sequence. name is the key to be used. Already existing keys are skipped unless overwrite is True. Evaluation is performed in parallel using multiprocessing if parallel is True."""
        global slhsequence_postprocess_worker
        def slhsequence_postprocess_worker(i):
            g = self.gridclass(i, prefix=self.dir)
            if not overwrite and name in g:
                logger.info("%s already present in %s, skipping" % (name, os.path.basename(g.filename)))
                return

            loc = locals()
            if addlocals:
                loc.update(addlocals)
            val = eval(expr, globals(), loc)
            logger.info("adding %s to %s" % (name, os.path.basename(g.filename)))
            g.add_vector_data(name, val)

        if parallel:
            from multiprocessing import Pool
            p = Pool(processes=processes)
            try:
                p.map(slhsequence_postprocess_worker, self.filenumbers)
            finally:
                p.close()
                p.join()
        else:
            for i in self.filenumbers:
                slhsequence_postprocess_worker(i)

    def check_files(self):
        """checks whether all files are valid SLH output
           return a list of broken files and a list of the exceptions"""
        broken = []
        exceptions = []
        for f in self.filenames:
            try:
                self.gridclass(f, 'f', prefix=self.dir)
            except Exception as e:
                broken.append(f)
                exceptions.append(e)
        return broken, exceptions

    def get_tdiff_avg(self,expr,t1,t2,**kwargs):
        time,data = self.get_tdiff(expr,**kwargs)

        i1 = -1
        i2 = -1

        for i in range(time.shape[0]):
            if i1<0 and time[i]>=t1:
                i1=i
            if i2<0 and time[i]>=t2:
                i2=i

        return data[i1:i2].mean()

    def power_spectrum(self,expr,n=None):
        time,data = self.get_data(expr)
        dt = np.diff(time)
        if np.any(dt[0] != dt):
            logger.warning("Time steps are not equal. Using average")
        dt = dt.mean()

        spec = np.fft.rfft(data,n=n)
        spec = np.abs(spec)**2
        if n is None:
            m = data.shape[0]
        else:
            m = n
        if m % 2 == 0:
            spec = spec[:-1]
        f = np.fft.fftfreq(m,d=dt)[:spec.shape[0]]
        return f, spec


    def plot(self,expr,tdiff=False,smooth=0,tunit=1.,ax=None,addlocals=None,norm0=False,**kwargs):
        """plot single value from every slh_grid versus time

        expr: string with python code that evaluates to the desired value
              the current grid object can be referenced with g

              Example: plot mean Mach number versus time
              plot("g.mach().mean()")

        addlocals: dictionary of variables that should be available when evaluating expr

        norm0: normalizes expr to initial value

        **kwargs: additional arguments to the plot command
        """
        if tdiff:
            time,data = self.get_tdiff(expr,addlocals=addlocals)
        else:
            time,data = self.get_data(expr,addlocals=addlocals)
        if smooth>0:
            data = self.smooth(data,smooth)
        time = time / tunit
        if ax is None:
            ax = plt.gca()
        if norm0:
            data = data / data[0]
        line, = ax.plot(time,data,**kwargs)
        opt = {'tdiff': tdiff, 'smooth': smooth, 'tunit': tunit}
        self.allplots.append((line, expr, opt))
        plt.xlabel('Time')

        plt.draw_if_interactive()

        return [line]

    def time_pcolor(self,expr,vaxis=1,ax=None,cax=None,Colorbar=True,tunit=1.,runit=1.,radial=False,Clear=True,Avg=True,addlocals=None,ptype='pcolor',cbar_kwargs={},**kwargs):
        """plot horizontal averages of a quantity over time as a "pcolor" plot

        expr: string with python code that evaluates to the desired value
              the current grid object can be referenced with g
        ptype: type of plot ['contour', 'pcolor'] (default: 'pcolor')

        **kwargs: additional arguments to the pcolormesh command
        """

        if ptype == 'pcolor':
            kwargs.setdefault('rasterized', True)

        if Clear:
            if ax:
                ax.cla()
            else:
                plt.clf()
        if ax is None:
            ax = plt.gca()

        g=self.gridclass(self.filenumbers[0],prefix=self.dir)

        haxes = list(range(g.sdim))
        haxes.remove(vaxis)

        # indices to extract the vertical direction
        vind = g.sdim * [slice(None)]
        vind_str = g.sdim * [':']
        for h in haxes:
            vind[h]     = 0
            vind_str[h] = '0'

        if radial:
            r = g.radius()
        else:
            r = g.coords(vaxis)
        r = r[vind]
        if Avg:
            expr = "np.squeeze(g.Havg(%s,%d))[%s]" % (expr, vaxis, ','.join(vind_str))
        else:
            expr = "np.squeeze(%s)" % expr
        time, data = self.get_data(expr, addlocals=addlocals)
        time = time / tunit
        r = r / runit
        if ptype == 'contour':
            mesh = ax.contour(time.T, r.T, data.T, **kwargs)
        elif ptype == 'pcolor':
            mesh = ax.pcolormesh(time.T, r.T, data.T, **kwargs)
        else:
            raise ValueError("unknown plot type ({0})".format(ptype))
        ax.set_xlabel('time')
        if Colorbar:
            if cax is None:
                divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(mesh, cax=cax, **cbar_kwargs)
            if cbar.solids is not None:
                cbar.solids.set_edgecolor('face')
            plt.sca(ax)
        ax.set_xlim(time.min(),time.max())
        ax.set_ylim(r.min(),r.max())

        plt.draw_if_interactive()

        return mesh

    def smooth(self,data,n):
        r = np.arange(data.shape[0])

        ndata = np.zeros(data.shape)

        for i in range(-n,n+1):
            ndata += data.take(r+i,mode='clip')

        return ndata/(2.*n+1.)

    def time_hist(self, expr, nhist=4, bins=20):
        """ Plots histograms of given expr to find appropiate colorbar and range"""
        self.find_files()
        fn = np.copy(self.filenumbers)
        fn = fn[::len(fn)/nhist]

        fig, axs = plt.subplots(nrows=nhist)
        for n, ax in zip(fn, range(nhist)) :
            g = self.gridclass(n, prefix=self.dir)
            data = '('+expr+').flatten()'
            plt.sca(axs[ax])
            #plt.hist(eval(data), bins=bins, label='time = '+str(g.time), normed=True)
            plt.hist(eval(data), bins=bins,  normed=True)
            plt.title('time = '+str(g.time))
            plt.legend(loc='best', frameon=False)
        plt.xlabel(expr)

    def storyboard(self, expr, nframes=4, **kwargs):
        """ Try out value range and colormap for given expression before
            the time-consuming rendering of an animation"""
        self.find_files()
        fn = np.copy(self.filenumbers)
        fn = fn[::len(fn)//nframes]

        #fig, axs = plt.subplots(ncols=nframes+1)
        fig, axs = plt.subplots(ncols=nframes, figsize=(20,5))
        for n, ax in zip(fn, range(nframes)) :
            g = self.gridclass(n, prefix=self.dir)
            plt.sca(axs[ax])
            g.pcolor(eval(expr), Clear=False, **kwargs)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(wspace=0.01)
        #plt.sca(axs[nframes])
        #plt.colorbar(orientation='horizontal')

    def radius(self, vaxis=1):
        g = self.gridclass(min(self.filenumbers), prefix=self.dir)
        return np.squeeze(g.Havg(g.coords(vaxis), vaxis, repeat=False))

    def replot(self):
        oldplots = self.allplots
        self.allplots = []
        for l, expr, opt in oldplots:
            style = {'c': l.get_c(), 'ls': l.get_ls(), 'lw': l.get_lw(), 'marker': l.get_marker(), 'markersize': l.get_markersize(), 'label': l.get_label()}
            style.update(opt)
            ax = l.axes
            ax.lines.remove(l)
            self.plot(expr, ax=ax, **style)
        plt.draw_if_interactive()

    def semilogy(self,*args,**kwargs):
        """similar to plot with logarithmic y-axis"""
        self.plot(*args, **kwargs)
        plt.yscale('log')

    def hdf5_export(self,overwrite=False):
        import os
        for i in self.filenumbers:
            g = self.gridclass(i,prefix=self.dir)
            if (not overwrite) and os.path.exists(g.filename.replace('.slh','.h5')):
                continue
            logger.info("Converting "+g.filename)
            g.hdf5_export()

    def animate(self, f):
        fig = plt.figure()

        def update_line(num, line):
            g = self.g(num)
            x, y = f(g)
            line.set_data(x, y)
            return line,

        ax = fig.add_subplot(111)
        x, y = f(self.g(0))
        l, = ax.plot(x, y)

        #line_ani = HTML5FuncAnimation(fig, update_line, frames=range(len(self.filenumbers)), fargs=(l,))

        #return line_ani
        return None

    @property
    def g0(self):
        return self.gridclass(min(self.filenumbers), prefix=self.dir)

    @property
    def g1(self):
        return self.gridclass(max(self.filenumbers), prefix=self.dir)

    def g(self, index):
        return self.gridclass(self.filenumbers[index], prefix=self.dir)

    @property
    def grids(self):
        for i in range(len(self.filenumbers)):
            yield self.g(i)

    def plot_posevo(self, expr=None, rpos=None, vaxis=None, addstd=True,
                    color=None, ax=None, label=None, locs={}, tunit='s',
                    **kwargs):
        '''plot evolution of positions according to
        the position of the peaks of data at a lower and upper
        domain. Can be used e.g. to follow boundaries of a CZ

        expr: taken to identify boundaries by searching for peaks in
              upper and lower domain. Result of eval(expr) must be at least
              1D with len self.g0.gnc[0]

              Predefined options:
              'grad_abar': (Default) Peaks in radial gradient of
                           mean atomic weight
              'grad_vhor': Peaks in radial gradient of horizontal
                           velocity

        rpos: [rbot, rtop] defining lower and upper boundary of CZ
              Domain is then divided in between. If not passed, domain
              is divided at g.gnc[0]//2
        addstd: If true, shade area of pos +- std-deviation

        #TODO Only implemented for polar/spherical grids so far.
        '''
        from stelo.model_reader import h



        if rpos is None:
            mind = self.g0.gnc[0]//2
        else:
            mind = sum(np.argmin(np.abs(r-rcz) for rcz in rpos))//2

        _locs = {'h':h,
                'r':self.g0.r,
                'mind':mind,
               }
        locs = {**locs, **_locs}

        if expr == 'grad_abar' or expr is None:
            expr = 'g.mean_std_twopeakspos(np.abs(h.grad(r, g.abar())), mind, r, 0)'

        elif expr == 'grad_vhor':
            axis = list(range(self.sdim))
            del(axis[0])
            expr = 'g.mean_std_twopeakspos(np.abs(np.mean(h.grad(r, g.vel(idir=0,' \
                'orthogonal=True)), axis=%s)), mind, r, 0)'%(str(tuple(axis)))

        elif expr == 'grad_ps':
            expr = 'g.mean_std_twopeakspos(np.abs(h.grad(r, g.ps()[-1])), mind, r , 0)'


        t,d = self.get_data(expr, addlocals=locs)

        dl, dlstd = d[:,0,:].T
        du, dustd = d[:,1,:].T

        if ax is None:
            ax = plt.gca()
        t = format_time(t,tunit, return_scaled_t=True)
        pl = ax.plot(t, dl, color=color, label=label, **kwargs)
        co = pl[0].get_color()
        ax.plot(t, du, color=co, **kwargs)

        tlab = tunit
        if tunit == 'hour':
            tlab = 'hours'

        ax.set_xlabel('t in '+tlab)
        ax.set_ylabel('r in cm')

        if addstd:
            ax.fill_between(t, dl+dlstd, dl-dlstd, alpha=0.2, color=co)
            ax.fill_between(t, du+dustd, du-dustd, alpha=0.2, color=co)

        return t,d,pl

    # movie related methods
    def globalminmax(self, expr, addlocals=None, parallel=True, processes=None):
        """Returns the global minimum and maximum of expr over all time steps.
        The current grid can be referenced with g"""
        global process_one

        def process_one(i):
            g = self.g(i)
            loc = locals()
            if addlocals:
                loc.update(addlocals)
            data = eval(expr,globals(),loc)
            if isinstance(data, list):
                return [d.min() for d in data], [d.max() for d in data]
            else:
                return [data.min()], [data.max()]

        if parallel:
            from multiprocessing import Pool
            p = Pool(processes=processes)
            try:
                res = p.map_async(wrap_mproc, range(len(self.filenumbers)))
                while not res.ready():
                    # this makes KeyboardInterrupt possible during parallel operation
                    res.wait(1000)
                res = res.get()
                p.close()
            finally:
                p.terminate()
                p.join()
        else:
            res = list(map(process_one, range(len(self.filenumbers))))

        vmin, vmax = res[0]
        for ivmin, ivmax in res:
            vmin = [min(d, v) for d, v in zip(ivmin, vmin)]
            vmax = [max(d, v) for d, v in zip(ivmax, vmax)]

        return vmin, vmax

    def pcolor(self, *args, **kwargs):
        return self._moviemaker('pcolor', *args, **kwargs)

    def plot_multi(self, *args, **kwargs):
        return self._moviemaker('plot_multi', *args, **kwargs)

    def pcolor_multi(self, *args, **kwargs):
        return self._moviemaker('pcolor_multi', *args, **kwargs)

    def quiver(self, *args, **kwargs):
        return self._moviemaker('quiver', *args, **kwargs)

    def _moviemaker(self, plottype, expr, targetfile='movie.webm', fps=25, rmtmp=True, parallel=False, processes=None, useglobalminmax=True, addlocals=None, encoder='ffmpeg', quality=4, maxbitrate='1M', html5=False, dpi=None, **kwargs):
        if parallel:
            from multiprocessing import Pool
        if plottype == 'pcolor':
            plotfun = lambda x: x.pcolor
        elif plottype == 'pcolor_multi':
            plotfun = lambda x: x.pcolor_multi
        elif plottype == 'quiver':
            plotfun = lambda x: x.quiver
        elif plottype == 'semilogy':
            plotfun = lambda x: x.semilogy
        elif plottype == 'plot_multi':
            plotfun = lambda x: x.plot_multi
        else:
            raise ValueError("unknown plottype %s" % plottype)

        tempdir = tempfile.mkdtemp()
        prefix = 'mov'
        wasinteractive = plt.isinteractive()
        plt.interactive(False)
        try:
            if plottype in ('pcolor', 'pcolor_multi','plot_multi'):
                if useglobalminmax:
                    gvmin, gvmax = self.globalminmax(expr,addlocals=addlocals, parallel=parallel, processes=processes)
                    kwargs['vmin'] = [b if a is None else a for a, b in zip(ensure_list(kwargs.get('vmin', [None]*len(gvmin)), length=len(gvmin)), gvmin)]
                    kwargs['vmax'] = [b if a is None else a for a, b in zip(ensure_list(kwargs.get('vmax', [None]*len(gvmin)), length=len(gvmin)), gvmax)]
            global process_one
            def process_one(ind):
                i = self.filenumbers[ind]
                logger.info("processing %d" % i)
                fig = plt.figure()
                g = self.gridclass(i, prefix=self.dir)
                loc = locals()
                if addlocals:
                    loc.update(addlocals)
                try:
                    plotfun(g)(eval(expr,globals(),loc), **kwargs)
                except Exception as e:
                    print(e)

                fig.savefig(os.path.join(tempdir, '%s%06d.png' % (prefix, ind) ), dpi=dpi)
                plt.close(fig)

            if parallel:
                def init_pool():
                    from imp import reload
                    import signal
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    # do not try this at home
                    try:
                        import matplotlib
                        matplotlib.rcParams['backend'] = 'agg'
                        matplotlib.use('agg',warn=False,force=True)
                    except TypeError:
                        # old matplotlib version
                        for x in list(sys.modules):
                            if x.startswith('matplotlib'):
                                del sys.modules[x]
                        import matplotlib
                        matplotlib.rcParams['backend'] = 'agg'
                        matplotlib.use('agg')
                        global plt
                        del plt
                        import matplotlib.pyplot as plt
                    reload(plt)

                    logger.debug("backend: " + matplotlib.get_backend())
                p = Pool(processes=processes, initializer=init_pool)
                try:
                    res = p.map_async(wrap_mproc, range(len(self.filenumbers)))
                    while not res.ready():
                        # this makes KeyboardInterrupt possible during parallel operation
                        res.wait(1000)
                    p.close()
                finally:
                    p.terminate()
                    p.join()
            else:
                for i in range(len(self.filenumbers)):
                    process_one(i)

            with tempfile.NamedTemporaryFile(suffix='.webm') as f:
                if html5:
                    targetfile = f.name
                ffmpegopts =  ['-framerate', str(fps), 
                               '-i', os.path.join(tempdir, prefix + '%06d.png'),
                               '-y', '-crf', str(quality),
                               '-b:v', maxbitrate, '-pix_fmt','yuv420p']
                if processes:
                    ffmpegopts += ['-threads', str(processes)]
                ffmpegopts += [targetfile]
                encodercall = {'mencoder': ['mencoder', '-o', targetfile, '-ovc', 'x264', '-mf', 'fps=%f' % fps, 'mf://' + os.path.join(tempdir, prefix + '*.png')],
                               'avconv': ['avconv'] + ffmpegopts,
                               'ffmpeg': ['ffmpeg'] + ffmpegopts,
                              }
                with tempfile.TemporaryFile() as stdout:
                    try:
                        subprocess.check_call(encodercall[encoder], stdout=stdout, stderr=stdout)
                    except subprocess.CalledProcessError as e:
                        stdout.seek(0)
                        logger.error('error in encoder call: ' + str(stdout.read()))
                        raise e
                if html5:
                    f.seek(0)
                    video = base64.b64encode(f.read()).decode('ascii')
        finally:
            if rmtmp:
                shutil.rmtree(tempdir)
            plt.interactive(wasinteractive)

        if html5:
            class animation(object):
                VIDEO_TAG = """<video controls>
                 <source src="data:video/x-m4v;base64,{0}" type="video/webm">
                  Your browser does not support the video tag.
                  </video>"""
                def __init__(self, video):
                    self._encoded_video = video

                def _repr_html_(self):
                    return self.VIDEO_TAG.format(self._encoded_video)

            return animation(video)


class slhmovie(slhsequence):
    def __init__(self, **kwargs):
        """The slhmovie class has been merged in to slhsequence.
        Please change you scripts accordingly."""
        logger.warning("slhmovie has been merged into slhsequence and will be removed at some point in the future. Please update your scripts.")
        super(slhmovie, self).__init__(**kwargs)

class multisequence(object):
    """ Compare multiple simulations effectively"""
    colors = ['#000000', '#332288', '#CC6677', '#DDCC77', '#117733', '#88CCEE',
              '#AA4499', '#44AA99', '#999933', '#882255', '#661100', '#6699CC',
              '#AA4466']
    dashes = [
      [],
      [2,2],
      [5,3],
      [7,3,2,3],
      [7,3,2,3,2,3],
      [7,3,7,3,2,3],
      [7,3,7,3,2,3,2,3],
      [7,3,2,3,2,3,2,3],
      ]
    markers = ['x', '+', 'o', 's', 'p', 'h', 'D']
    styletypes = {'color': colors, 'dashes': dashes, 'marker': markers}

    def __init__(self, dirs, labels=None, gridclass=slhgrid, takespread=None, step=None):
        self.dirs = [os.path.normpath(dir) + os.sep for dir in dirs]
        self.labels = labels
        self.seqs = []
        self.g0s = []

        if self.dirs == None:
            raise ValueError('You must specify at least two directories.')
        if self.labels == None:
            logger.warning('No labels specified. Using directory names as labels.')
            self.labels = [os.path.dirname(dir) for dir in self.dirs]
        for dirpath in self.dirs:
            logger.debug('loading', dirpath)
            try:
                s = slhsequence(dir=dirpath, gridclass=gridclass)
                g0 = s.g0
            except Exception as e:
                logger.warning('skipping %s (%s)' % (dirpath, e))
                continue
            self.seqs.append(s)
            self.g0s.append(g0)
            if takespread != None:
                s.takespread(takespread)

    def find_files(self):
        for seq in self.seqs:
            seq.find_files()

    def penultimate(self):
        """ When analysing simulations still running it is best to ignore the last gridfile."""
        for seq in self.seqs:
            seq.filenumbers = seq.filenumbers[:-1]

    def plot(self, expr, legend=True, Clear=True, **kwargs):
        for seq, lab in zip(self.seqs, self.labels):
            logger.debug('plotting '+lab)
            seq.plot(expr, label=lab)
        if legend:
            plt.legend(loc='best', frameon=False)

    def plot_criteria(self, expr, criteria, order=['color', 'marker', 'dashes'], legend=True, label=None, verbtext=True, **kwargs):
        order = [[{o:t} for t in self.styletypes[o]] for o in order]
        names = [list(map(c, self.g0s)) for c in criteria]
        inames = list(zip(*names))
        styles = [dict(zip(set(n),o)) for n, o in zip(names, order)]

        for seq,name in zip(self.seqs,inames):
            sty = {}
            for n,s in zip(name, styles):
                sty.update(s[n])
            sty.update(kwargs)
            lab = ' '.join(name)
            if plt.rcParams['text.usetex'] and verbtext:
                lab = r'\verb|{0}|'.format(lab)
            if label:
                lab = label + ' ' + lab
            sty['label'] = lab
            seq.plot(expr, **sty)
        if legend:
            plt.legend(loc='best')


class _rans_setting(object):

    def __init__(self, g, dt=None):
        '''
        if dt given, it sets a slice for RANS data to give values
        approximately in steps of dt. Assumes that time steps are
        roughly the same for the whole simulation and well resembled
        by the passed grid slhgrid instance
        '''

        self.g = g
        vidxs = g['grid%rans%vidxs']
        self.theta = []
        self.phi = []
        phi = self.g.phi()
        the = self.g.theta()

        self.dinds = self.g['grid%rans%vidxs']

        for i1,i2 in vidxs.T:
            if self.g.sdim == 2 :
                sl = tuple([slice(None),i1])
            elif self.g.sdim == 3:
                sl = tuple([slice(None),i1,i2])
            self.theta.append(the[sl])
            self.phi.append(phi[sl])

        self.dt = self._get_meandt()
        self.sl = slice(None,None,1)

        if dt is not None:

            if (int(dt//self.dt) == 0):
                print("Warning, chosen dt={:.2f} smaller than " \
                       "mean time step of {:.2f}. Set step size to 1".format(
                          dt, self.dt))
            else:
                self.sl = slice(None,None,int(dt//self.dt))
                print(
                    "Set slice to roughly match a " \
                    "time step of dt={:.2f}".format(dt))
                print('Slicer step:', self.sl.step)


        self.nrad = np.array([np.cos(self.phi)*np.sin(self.theta),
                    np.sin(self.phi)*np.sin(self.theta),
                    np.cos(self.theta)*np.sin(self.phi)/np.sin(self.phi)])

        self.ntheta = np.array([np.cos(self.phi)*np.cos(self.theta),
                      np.sin(self.phi)*np.cos(self.theta),
                     -np.sin(self.theta)*np.sin(self.phi)/np.sin(self.phi)])

        self.nphi = np.array([-np.sin(self.phi),np.cos(self.phi),np.sin(self.phi)*0])

    def _get_meandt(self):
        try:
            t = self.g['grid%rans%times']
            return np.mean(t[1:]-t[:-1])
        except KeyError:
            return 10


class RANS(slhsequence):
    def __init__(self, tc, tframe, vaxis=1, gfonly=False, gdata=False,
                 takespread=None, **kwargs):
        '''add description
        '''


        self.tc = tc*60*60
        self.tframe = tframe*60*60
        self.vaxis = vaxis
        self._time = None

        super(RANS, self).__init__(**kwargs)

        self.gind = self.nearest_snapshot(self.tc)
        self.lgind = self.nearest_snapshot(self.tc-self.tframe/2)
        self.ugind = self.nearest_snapshot(self.tc+self.tframe/2)
        self.lg = self.g(self.lgind)
        self.ug = self.g(self.ugind)
        self.r = self.g0.r

        self.gfonly = gfonly
        self.gdata = gdata

        self._gr = None
        self.volf = None


        if takespread is not None:
            self.takespread(takespread)

        if self.gdata:
            self._load_gdata_prereq()
        else:
            self._load_data()

        # stuff to load data explicitly from grid
        self._grho = None
        self.filenumbers = self.filenumbers[self.lgind:self.ugind]
        if self.lgind == self.ugind:
            raise RuntimeError(
                'Chosen time frame {}s to {}s does not contain any data'.format(
                    self.tc-self.tframe/2, self.tc+self.tframe/2))


    def nearest_snapshot(self,t):
        ''' primitive bisection algorithm to find snapshot nearest to t

        much more efficient than first looping through loading grid files
        '''

        self.find_files()
        a = 0
        b = float(len(self.filenumbers))
        im = int((a+b)/2)
        while((a != im) and (b != im)):
            tmp = self.g(im).time
            if tmp<t:
                a = im
            else:
                b = im
            im = int((a+b)/2)
        return im

    def _load_gdata_prereq(self):
        '''set some variables if grid data is used only

        actual data is loaded in self._data
        '''
        self.t0 = self.lg.time
        self.t1 = self.ug.time

    def _load_data(self):
        '''load RANS data from gridfiles

        Reyleigh average is calculated on the fly for the sake of
        efficiency and memory
        '''

        from scipy.integrate import trapz

        self._ident = self.g(self.lgind)['grid%rans']
        self.nidents = len(self.idents) - 1
        inds = np.arange(self.lgind,self.ugind+1)

        if self.gfonly:
            sl1 = slice(-1,None)
        else:
            sl1 = slice(None)

        sl2 = (self._ident['vol']-1, sl1, slice(None))
        sl3 = (slice(None, self.nidents), sl1, slice(None))

        if use_progressbar:
            bar = Bar('Extracting RANS data', max=len(inds),
                      suffix="%(index)d/%(max)d (%(eta_td)s)")

        # stores number of data points used for the average
        self.ndat = 0
        dint = np.zeros((self.nidents,self.g0.gnc[0]))

        self.times = []
        off = 0
        for i in inds:
            g = self.g(i)
            if np.any(np.isnan(g['grid%rans%data'][3])) and not self.gdata:
                print('\nNaNs detected\n')
                off += 1
                if use_progressbar:
                    bar.next()
                continue

            # Divide by volume and integrate in time
            dint += trapz(
                g['grid%rans%data'][sl3]/g['grid%rans%data'][sl2],
                g['grid%rans%times'][sl1],axis=1)
            self.ndat += len(g['grid%rans%times'][sl1])

            # this integrates between the beginning and end of two subsequent
            # grid files
            if (i != (self.lgind+off)):

                if self.gfonly:
                    t1 = g['grid%rans%times'][-1]
                    d1 = g['grid%rans%data'][:self.nidents,-1,:]/g['grid%rans%data'][self._ident['vol']-1,-1,:]
                else:
                    t1 = g['grid%rans%times'][0]
                    d1 = g['grid%rans%data'][:self.nidents,0,:]/g['grid%rans%data'][self._ident['vol']-1,0,:]

                dint += trapz([d0,d1],[t0,t1],axis=0)

            t0 = g['grid%rans%times'][-1]
            d0 = g['grid%rans%data'][:self.nidents,-1,:]/g['grid%rans%data'][self._ident['vol']-1,-1,:]

            # If SLH terminates it might override the last grid file. Since RANS data
            # is already reset by then this grid file only contains zeros
            # and should be skipped.
            #if not any(g['grid%rans%times'] == 0):
            if not any(np.isnan(g['grid%rans%times'])):
                self.times = self.times + list(g['grid%rans%times'])
            else:
                print('\n RANS, load_data warning: Grid file {} not valid'.format(
                    g.filename))

            if use_progressbar:
                bar.next()

        self.times = np.array(self.times)

        if use_progressbar:
            bar.finish()

        # divide integral by time span to get time average
        if self.gfonly:
            self._vals = dint / (self.g(self.ugind)['grid%rans%times'][-1] -
                           self.g(self.lgind)['grid%rans%times'][-1])
        else:
            self._vals = dint / (self.g(self.ugind)['grid%rans%times'][-1] -
                           self.g(self.lgind)['grid%rans%times'][0])

        # Save time of first and last snapshot for time derivative
        self.t0 = self.g(self.lgind)['grid%rans%times'][0]
        self.t1 = self.g(self.ugind)['grid%rans%times'][-1]

    def _itg(self, ident):
        def vr():
            if self.g0.geometry.type in ['polar', 'spherical']:
                return 'g.vel(idir=0)'
            elif self.g0.geometry.type in ['cartesian']:
                if self.sdim == 3:
                    return 'g.vel(2)'
                else:
                    return 'g.vel(1)'
            else:
                raise NotImplementedError
        def vp():
            if self.g0.geometry.type in ['polar', 'spherical']:
                if self.sdim == 2:
                    return 'g.vel(idir=1)'
                elif self.sdim == 3:
                    return 'g.vel(idir=2)'
            elif self.g0.geometry.type in ['cartesian']:
                if self.sdim == 2:
                    return 'g.vel(0)'
                if self.sdim == 3:
                    return 'g.vel(1)'
            else:
                raise NotImplementedError
        def vt():
            if self.g0.geometry.type in ['polar', 'spherical']:
                if self.sdim == 2:
                    return 'np.zeros_like(g.vel(1))'
                elif self.sdim == 3:
                    return 'g.vel(idir=2)'
            elif self.g0.geometry.type in ['cartesian']:
                if self.sdim == 2:
                    return 'np.zeros_like(g.vel(1))'
                if self.sdim == 3:
                    return 'g.vel(2)'
            else:
                raise NotImplementedError


        if ident=='rho':
            return 'g.rho()'
        elif ident=='rhopres':
            return self._jexpr('g.rho()','g.pres()')
        elif ident=='ekin':
            return self._jexpr('g.ekin()/g.rho()')
        elif ident=='rhoekin':
            return self._jexpr('g.ekin()/g.rho()','g.rho()')
        elif ident=='vol':
            return 'g.vol()'
        elif ident=='pres':
            return 'g.pres()'
        elif ident=='vdiv':
            #return 'g.metric_divergence(g.vel())'
            return 'g.divergence(g.vel)'
        elif ident=='vrad':
            return vr()
        elif ident=='presvrad':
            return self._jexpr('g.pres()', vr())
        elif ident=='rhovrad':
            return self._jexpr('g.rho()',vr())
        elif ident=='rhovradpres':
            return self._jexpr('g.rho()',vr(),'g.pres()')
        elif ident=='rhovradvrad':
            return self._jexpr('g.rho()',vr()+'**2')
        elif ident=='rhovradvradvrad':
            return self._jexpr('g.rho()',vr()+'**3')
        elif ident=='presvdiv':
            return self._jexpr('g.pres()',
                   #'g.metric_divergence(g.vel())')
                   'g.divergence(g.vel)')
        elif ident=='rhovdiv':
            return self._jexpr('g.rho()',
                   #'g.metric_divergence(g.vel())')
                   'g.divergence(g.vel)')

        elif ident=='vphi':
            return vp()
        elif ident=='rhovphi':
            return self._jexpr('g.rho()',vp())
        elif ident=='rhovphivphi':
            return self._jexpr('g.rho()',vp()+'**2')
        elif ident=='rhovphivrad':
            return self._jexpr('g.rho()',vp(),vr())
        elif ident=='rhovphivphivrad':
            return self._jexpr('g.rho()',vp()+'**2',vr())
        elif ident=='vthe':
            return vt()
        elif ident=='rhovthe':
            return self._jexpr('g.rho()',vt())
        elif ident=='rhovthevrad':
            return self._jexpr('g.rho()',vt(),vr())
        elif ident=='rhovthevthe':
            return self._jexpr('g.rho()',vt()+'**2')
        elif ident=='rhovthevthevrad':
            return self._jexpr('g.rho()',vt()+'**2',vr())
        elif ident=='eint':
            return self._jexpr('g.eps()')
        elif ident=='rhoeint':
            return self._jexpr('g.rho()','g.eps()')
        elif ident=='rhoeintvrad':
            return self._jexpr('g.rho()','g.eps()',vr())
        elif ident=='rhoenuc':
            # to be implemented
            try:
                self.g0['grid%dedt%data']
                res = self._jexpr('g.rho()','np.squeeze(g["grid%dedt%data"])')
            except KeyError as e:
                res = '0'
            return res
        elif ident=='temp':
            return self._jexpr('g.temp()')
        elif ident=='rhotemp':
            return self._jexpr('g.rho()', 'g.temp()')
        elif ident=='vradtemp':
            return self._jexpr(vr(), 'g.temp()')
        elif ident=='rhovradtemp':
            return self._jexpr('g.rho()', vr(), 'g.temp()')
        elif ident=='rhovradtemp':
            return self._jexpr('g.rho()', vr(), 'g.temp()')
        else:
            raise RuntimeError(
            'RANS, _itg: unknown ident %s'%ident)

    def _jexpr(self, *args):
        expr = ''
        first = True
        for arg in args:
            if first:
                expr+=arg
                first = False
            else:
                expr+='*'+arg
        return expr

    def _cexpr(self, *args):
        expr = ''
        first = True
        argl = []
        for arg in args:
            argl.append(self._itg(arg))
        return self._jexpr(*argl)

    @property
    def grho(self):
        if self._grho is None:
            self._grho = self.get_data('g.rho()')
        return self._grho

    def gvolf(self):
        if self.volf is None:
            self.volf = self.g0.vol()
        return self.volf

    def _reyavg_fromg(self, idents, fordt=False):
        from scipy.integrate import trapz

        if self.g1.geometry.type == 'spherical':
            axis = '(1,2)'
        elif self.g1.geometry.type == 'polar':
            axis = '(1)'
        elif self.g1.geometry.type == 'cartesian':
            axis = '(0,1)' if self.sdim == 3 else '(0)'
        else:
            raise RuntimeError('Geometry not implemented')

        #vol = eval('np.sum(self.g0.vol(),%s)'%axis)
        volf = self.gvolf()
        vol = eval('np.sum(volf,%s)'%axis)

        idents = ensure_list(idents)
        dexpr = self._cexpr(*idents)
        fexpr = 'np.sum(volf*{},axis={})'.format(
            dexpr, axis)
        t, dat = self.get_data(fexpr,
            addlocals={'volf':volf})

        self._gdt = t[-1]-t[0]
        avg = trapz(dat/vol, t, axis=0)/(self._gdt)

        if not fordt:
            return avg
        else:
            g = self.lg
            v0 = eval(fexpr)/vol
            g = self.ug
            v1 = eval(fexpr)/vol
            return np.array([v0,avg,v1])

    def data(self, ident):
        '''return Reyleigh averaged RANS data for given ident

        Reyleigh average is average over space and time

        Parameters
        ----------
        ident: string
            Identifier for data, e.g. "rho", "rhovrad", "ekin",...

        Returns
        -----
        data: 1D array
        '''

        return self._vals[self._ident[ident]-1]

    def fT(self):
        '''heat flux

        according to Mocak 2014, Table 1
        '''

        return -self.rey_avg('heatflux')

    def fI(self):
        '''internal energy flux

        according to Mocak 2014, Table 1
        '''

        return self.rey_avg('rho') * self.avg_p('eint','vrad',fa=True)


    def fp(self):
        '''acoustic flux

        according to Mocak 2014, Table 1
        '''

        return self.avg_p('pres','vrad',fa=False)


    def calGalpha(self,spec):
        ''' geometric factor for species alpha

        Apparently not described in Mocak 2014, so this
        formula is taken from Mocak 2018 in the text below
        Eq. 9.

        calGalpha = a+b+c

        a = -<rho*X_i''*vthe''*vthe''/r>
        b = -<rho*X_i''*vphi''*vphi''/r>
        c = -<X_i'' * GrM>
        GrM = (-rho_vthe**2/r - rho * vphi**2/r)
        '''
        r = self.g(self.lgind).r
        # TODO refer to my rans notes
        def gi(dir):
            a = self.rey_avg('rho'+dir+dir+spec)/r                    \
              - self.rey_avg('rho'+dir+spec)/r*self.favre_avg(dir) \
              - self.rey_avg('rho'+dir+spec)/r*self.favre_avg(dir) \
              + self.rey_avg('rho'+spec)/r*self.favre_avg(dir)**2  \
              - self.rey_avg('rho'+dir+dir)/r*self.favre_avg(spec)    \
              + self.rey_avg('rho'+dir)/r*self.favre_avg(spec) * self.favre_avg(dir) \
              + self.rey_avg('rho'+dir)/r*self.favre_avg(spec) * self.favre_avg(dir) \
              - self.rey_avg('rho')/r*self.favre_avg(spec)*self.favre_avg(dir)**2
            return a

        a = -gi('vphi')
        b = -gi('vthe')

        c1 = -(self.rey_avg('rhovphivphi'+spec)/r \
           - self.favre_avg(spec) * self.rey_avg('rhovphivphi')/r)
        c2 = -(self.rey_avg('rhovthevthe'+spec)/r \
           - self.favre_avg(spec) * self.rey_avg('rhovthevthe')/r)
        c = -(c1+c2)
        return a+b+c


    def specflux(self,spec):
        '''Flux of the turbulent flux of species spec

        According to Mocak 2014 Eq. 30
        See also Eq. 9 of Mocak 2018
        '''
        falpha = self.falpha(spec,fordt=True)
        rho = self.rey_avg('rho',fordt=True)
        falpharho = [falpha[i]/rho[i] for i in range(len(rho))]
        lhs = self.rey_avg('rho') * self.lagr_deriv(falpharho)
        p1 = -self.nabla_r(self.fralpha(spec))
        p2 = -self.falpha(spec)*self.partial_r(self.favre_avg('vrad'))
        p3 = -self.Rij('rad','rad') * self.partial_r(self.favre_avg(spec))
        # <a''> = (<a> - ~a~), see Mocak 2014 eq. 114
        # TODO add reference to my rans pdf
        p4 = -(self.rey_avg(spec) - self.favre_avg(spec)) * self.partial_r(self.rey_avg('pres'))
        # TODO add reference to my rans pdf
        p5 = self.rey_avg('dpdr'+spec) - self.favre_avg(spec)*self.partial_r(self.rey_avg('dpdr'))
        p6 = self.rey_avg('rhovraddt'+spec) - self.favre_avg('vrad')*self.rey_avg('rhodt'+spec)
        p7 = self.calGalpha(spec)

        rhs = p1,p2,p3,p4,p5,p6,p7

        return lhs,rhs


    def sigmaalpha(self,spec):
        '''variance of the mass fraction of species spec

        according to Mocak 2014 Eq. 41
        see also Mocak 2018 Eq. 12
        '''


        sigalpha = self.favre_avg('squ'+spec,fordt=True)-self.favre_avg(spec,fordt=True)**2
        lhs = self.rey_avg('rho') * self.lagr_deriv(sigalpha)

        p1_1 = self.rey_avg('rhovradsqu'+spec) \
             - 2 * self.rey_avg('rhovrad'+spec) * self.favre_avg('vrad') \
             + 2 * self.rey_avg('rho'+spec)*self.favre_avg(spec)*self.favre_avg('vrad') \
             + self.rey_avg('rhovrad')*self.favre_avg(spec)**2 \
             - self.rey_avg('rhosqu'+spec) \
             + self.rey_avg('rho') * self.favre_avg(spec)**2 * self.favre_avg('vrad')
        p1 = -self.nabla_r(p1_1)
        p2 = -2*self.falpha(spec)*self.partial_r(self.favre_avg(spec))
        p3 = 2 * self.rey_avg('rhodtnuc'+spec) - self.favre_avg(spec)*self.rey_avg('rhodt'+spec)

        rhs = p1,p2,p3

        return lhs,rhs


    def fralpha(self,nuc,**kwargs):
        ''' radial flux of falpha

        according to Mocak 2014, Table 1
        '''

        return self.rey_avg('rho',**kwargs) * self.avg_p('vrad','vrad',nuc,fa=True,**kwargs)


    def falpha(self,nuc,**kwargs):
        '''flux of species 'nuc'

        according to Mocak 2014, Table 1
        '''
        return self.rey_avg('rho',**kwargs) * self.avg_p('vrad',nuc,fa=True,**kwargs)


    def fk(self):
        '''turbulent kinetic energy flux

        according to Mocak 2014, Table 1
        '''
        fk = 0.5*self.rey_avg('rho') * (self.avg_p('vrad','vrad','vrad',fa=True) +
                                        self.avg_p('vphi','vphi','vrad',fa=True) +
                                        self.avg_p('vthe','vthe','vrad',fa=True))
        return fk


    def Rij(self,i,j):
        '''Reynolds stress tensor

        i,j in [rad, phi, theta]

        according to Mocak 2014, Table 1
        '''

        for x in [i,j]:
            if x not in ['rad','phi','the']:
                raise Warning('RANS, Rij: {} not recognized'.format(x))

        # create identifier vrad/vthe/vphi
        vi = 'v'+i
        vj = 'v'+j

        return self.rey_avg('rho') * self.avg_p(vi,vj,fa=True)


    def Wp(self):
        '''turbulent pressure dilatation

        according to Mocak 2014, Table 1
        '''

        # <P' d''> = <Pd> - <P><d>, see Eq. 37
        # TODO add my notes.pdf

        return self.rey_avg('presvdiv') - self.rey_avg('pres')*self.rey_avg('vdiv')


    def Wb(self):
        '''buoyancy

        according to Mocak 2014, Table 1

        Note I:  This assumes the grav. force to be g(r,theta,phi,t) = g(r)
        Note II: The sign might be wrong in Mocak 2014. Their expression transforms to
                     Wb = <rho>~g_r~<u_r''> = g_r(<rho><u_r> - <rho u_r>)
                 where g_r<0. However, their plot shows that Wb is mostly positiv.
                 In Viallet et al. 2012 they say:
                     Wb = <rho' u_r' g_r> (eq. 49)
                 This translates to
                    Wb = g_r(<rho u_r> - <rho><u_r>)
                 which is identical but with an opposit sign. However, this gives Wb>0 in the main part of
                 the convection zone. So this method returns Mocak 2014 multiplied by -1
        '''

        import stelo.helpers as h
        # radial part of gravity
        # must be the same for all theta/phi
        gr = self.gr
        gr = h.grad(self.r, self.rey_avg('pres'))

        # <u''_r> = <u> - ~<u>~ (see Mocak Eq. 114)
        upart = self.rey_avg('vrad') - self.favre_avg('vrad')

        #return -self.rey_avg('rho') * upart * gr
        return -upart * gr


    def Gmr(self):
        '''Radial geometric factor

        according to Mocak 2014 Eq. 102

        Note: Viscosity tensor is set to zero here
        '''

        return -(self.data('rhovthevthe') + self.data('rhovphivphi'))/self.r


    def vrad(self):
        '''radial velocity

        according to Mocak 2014 Eq. 4
        '''

        vrad = self.favre_avg('vrad',fordt=True)
        lhs = self.rey_avg('rho') * self.lagr_deriv(vrad)

        R_rr = self.Rij('rad','rad')
        p1 = - self.nabla_r(R_rr)
        p2 = - self.Gmr()
        p3 = - self.partial_r(self.rey_avg('pres'))
        p4 = + self.rey_avg('rho') * self.gr

        rhs =  p1,p2,p3,p4
        return lhs,rhs


    def species(self,spec):
        '''species spec

        according to Mocak 2014 Eq. 14
        '''
        specfav = self.favre_avg(spec,fordt=True)
        lhs = self.rey_avg('rho') * self.lagr_deriv(specfav)

        p1 = -self.nabla_r(self.falpha(spec))
        p2 = self.rey_avg('rho')*self.favre_avg('dt'+spec)

        rhs = p1,p2

        return lhs,rhs


    def ekin(self):
        '''kinetic energy

        according to Mocak 2014 Eq. 8
        '''

        ekin = self.favre_avg('ekin',fordt=True)
        #lhs = self.rey_avg('rho') * self.lagr_deriv(ekin)
        lhs = self.lagr_deriv(ekin)

        p0 = -self.nabla_r(self.fk())
        p1 = -self.nabla_r(self.fp())
        p2 = -np.sum([self.Rij(i,'rad') * self.partial_r(self.favre_avg('v'+i))
            for i in ['rad','phi','the']],axis=0)
        p3 = self.Wb()
        p4 = self.Wp()

        #<rho> ~D(~u_i~u_i/2) according to Mocak 2014 Eq. 8
        u = self.favre_avg('vrad',fordt=True)
        uradurad = np.array([u[i]**2/2 for i in [0,1,2]])

        u = self.favre_avg('vthe',fordt=True)
        uphiuphi = np.array([u[i]**2/2 for i in [0,1,2]])

        u = self.favre_avg('vthe',fordt=True)
        utheuthe = np.array([u[i]**2/2 for i in [0,1,2]])

        uiuihalf = uradurad + uphiuphi + utheuthe
        #p5 = self.rey_avg('rho') * self.lagr_deriv(uiuihalf)
        p5 = self.lagr_deriv(uiuihalf)

        rhs = p0,p1,p2,p3,p4,p5

        return lhs, rhs

    def eint(self):
        '''internal energy

        according to Mocak 2014 Eq.7
        '''

        eint = self.favre_avg('eint',fordt=True)
        lhs = self.rey_avg('rho') * self.lagr_deriv(eint)

        p1 = -self.grad_r(self.fI() + self.fT())
        p2 = self.rey_avg('pres') * self.rey_avg('vdiv')
        p3 = -self.Wp()
        # From Mocak 2014 Eq. 7 not clear what kind of average this should be
        p4 = self.rey_avg('rhoenuc')

        rhs = p1,p2,p3,p4

        return lhs,rhs

    def etot(self):
        '''total energy

        according to Mocak 2014 Eq. 9
        '''
        eintls, eintrs = self.eint()
        ekinls, ekinrs = self.ekin()

        return [eintls,ekinls],[eintrs,ekinrs]


    def avg_p(self,a,b,c=None,fa=True,**kwargs):
        '''return favre (fa=True) or reynolds average of equally primed variables
        Arguments a,b,c muss _all_ be string

        According to eq (TODO add my notes.pdf)

        ~a''b''~    = ~ab~ - ~a~*~b~
        ~a''b''c''~ = ~abc~ - ~ab~*~c~ - ~ac~*~b~ - ~a~*~bc~ + 2 * ~a~*~b~*~c~

        where ~a~ is the favre/reynolds average of variable a
        '''
        if fa:
            av = lambda x: self.favre_avg(x,**kwargs)
        else:
            av = lambda x: self.rey_avg(x,**kwargs)
        if not isinstance(a,str):
            raise RuntimeError('RANS, avg_p, all arguments need to be strings')

        if c is None:
            res = av(a+b) - av(a)*av(b)
        else:
            res = av(a+b+c) - av(a+b)*av(c) - av(a+c)*av(b) \
                - av(a)*av(b+c) + 2 * av(a)*av(b)*av(c)
        return res


    def tavg(self, data):
        '''calculate time average of data

        integrate in time using trapezoidal rule, divide by time span

        Parameters
        ----------
        data : array of shape (#timestep,#radial gridpoints)

        Returns
        -------
        tavg :  time average of shape (#radial gridpoints)

        Note
        ----
            Integration not necessarily symmetric
            TODO: Fix this
        '''

        from scipy.integrate import trapz

        integral = trapz(data, self.trans, axis=0)
        tspan = self.trans[-1] - self.trans[0]

        return integral/tspan


    def rey_avg(self, data, fordt=False):
        ''' add description
        '''
        if self.gdata:
            return self._reyavg_fromg(data, fordt=fordt)
        elif not fordt:
            return self.data(data)
        else:
            # load data of first and last snapshot
            if self.gfonly:
                v0 = self.lg['grid%rans%data'][self._ident[data]-1,-1,:] / self.lg['grid%rans%data'][0,-1,:]
            else:
                v0 = self.lg['grid%rans%data'][self._ident[data]-1,0,:] / self.lg['grid%rans%data'][0,0,:]

            v1 = self.ug['grid%rans%data'][self._ident[data]-1,-1,:]/ self.ug['grid%rans%data'][0,-1,:]
            return np.array([v0,self.data(data),v1])


    def favre_avg(self, data, **kwargs):
        '''add description
        '''

        rho = self.rey_avg('rho', **kwargs)
        if isinstance(data, str):
            data = 'rho'+data
            try:
                rhoval = self.rey_avg(data, **kwargs)
            except KeyError:
                raise KeyError('Value <{}> not found.'.format(data) \
                              +'Try to change order as only one of ' \
                              +'the possible combination is ' \
                              +'stored in the grid file')
        else:
            raise NotImplementedError('RANS, favre_avg: Input needs to be type string '
                'not type {}'.format(type(data)))
        if not 'fordt' in kwargs:
           return rhoval / rho
        else:
           return np.array([rhoval[0]/rho[0],rhoval[1]/rho[1],rhoval[2]/rho[2]])


    def prime(self, expr, n=1, **kwargs):
        '''add description
        '''

        if n == 1:
            f = self.rey_avg
        elif n == 2:
            f = self.favre_avg
        else:
            raise ValueError("n must be 1 or 2")
        valavg = f(expr, **kwargs)
        t, val = self.get_data(expr)
        return val - valavg


    def dtavgdt(self,q0,q1):
        '''Compute the time derivative of a temporal average according to

            d/dt <q> = 1/(Dt) [q(t+Dt/2) - q(t-Dt/2)]

            where Dt is the window size of the temporal average
            (Casey Meakin, priv. comm.)

        Parameters
        ----------
        q0: data at first time step
        q1: data at last time step

        Returns
        -------
            float
            Time derivative of temporal average
        '''

        return (q1-q0)/(self.t1-self.t0)


    def nabla_r(self, q):
        ''' return radial part of the divergence of q
        '''
        import stelo.helpers as h

        if self.g0.geometry.type == 'polar':
            return 1/self.r * h.grad(self.r,self.r*q)
        elif self.g0.geometry.type == 'spherical':
            return 1/self.r**2 * h.grad(self.r,self.r**2*q)
        elif self.g0.geometry.type == 'cartesian':
            if self.sdim == 3:
                return h.grad(self.g0.coords(2)[0,0,:], q)
            else:
                return h.grad(self.g0.coords(1)[0,:], q)
        else:
            raise NotImplementedError(
                'RANS: nabla_r, geometry type {} not implemented'.format(self.g0.geometry.type))


    def partial_r(self, q, **kwargs):
        '''add description
        '''
        import stelo.helpers as h

        if self.g0.geometry.type in ['polar','spherical']:
            return h.grad(self.r, q, **kwargs)
        elif self.g0.geometry.type == 'cartesian':
            if self.sdim == 3:
                return h.grad(self.g0.coords(2)[0,0,:], q)
            else:
                return h.grad(self.g0.coords(1)[0,:], q)
        else:
            raise NotImplementedError(
                'RANS: partial_r, geometry type {} not implemented'.format(self.g0.geometry.type))


    def divergence(self, q):
        '''add description
        '''
        import stelo.helpers as h
        raise RuntimeError('Should not be called')
        sdim = self.sdim
        r = [self.radius(vaxis=i) for i in range(sdim)]
        return np.array([h.grad(r[i], q[:, i], axis=i+1)
                        for i in range(sdim)]).sum(axis=0)


    def lagr_deriv(self, q, **kwargs):
        '''add description
        '''

        ur_favre = self.favre_avg('vrad', **kwargs)
        rrey = self.rey_avg('rho', fordt=True)

        if np.shape(q)[0] != 3:
           raise RuntimeError('RANS, lagr_deriv: input q needs to be data at first, central and last time step')
        return self.dtavgdt(q[0],q[-1]) + ur_favre * self.partial_r(q[1])
        #return self.dtavgdt(rrey[0]*q[0],rrey[-1]*q[-1]) + self.nabla_r(rrey[1]*q[1]*ur_favre)


    def continuity(self):
        '''continuity equation

        according to Mocak 2014 eq. 3

        Returns
        -------
        [lhs,rhs]

        lhs = ~D_t(<rho>)
        rhs = [-<rho> ~vdiv]

        '''

        rho_rey = self.rey_avg('rho',fordt=True)
        return self.lagr_deriv(rho_rey), -rho_rey[1] * self.favre_avg('vdiv')
        #return self.lagr_deriv([1,1,1]), -rho_rey[1] * self.favre_avg('vdiv')


    @property
    def idents(self):
        '''add description
        '''

        return list(self._ident.keys())


    @property
    def trans(self):
        '''array containing times at which RANS data were written
        '''
        return self._trans


    @property
    def rtime(self):
        '''add description
        '''
        return self._trans


    @property
    def vol(self):
        '''add description
        '''
        return self.data('vol')

    @property
    def gr(self):

        if self._gr is None:
            if self.g0.geometry.type in ['polar', 'spherical']:
                sl = tuple([slice(None)]+[0]*(self.sdim-1))
                self._gr = self.g0.grav(idir=0)[sl]
            elif self.g0.geometry.type in ['cartesian']:
                if self.sdim == 3:
                    self._gr = self.g0.grav(2)[0,0,:]
                else:
                    self._gr = self.g0.grav(1)[0,:]
            else:
                raise NotImplementedError(
                'slhouput_vec: gravitation not implemented for {}'.format(
                    self.g0.geometry.type))

        return self._gr


def Hoperation(data,vaxis,operation,repeat=True):
    """horizontal average
    vaxis is the vertical axis
    if repeat is true, the output array has the same shape as data"""

    if data.ndim == 1:
        return data
    else:
        d = list(range(data.ndim))
        del d[vaxis]
        m = np.apply_over_axes(operation, data, d)
        return np.apply_over_axes(lambda x, a: x.repeat(data.shape[a] if repeat else 1, axis=a), m, d)


def Havg(data,vaxis,repeat=True):
    """horizontal average
    vaxis is the vertical axis
    if repeat is true, the output array has the same shape as data"""
    return Hoperation(data,vaxis,np.mean,repeat=repeat)

def Hmin(data,vaxis,repeat=False):
    return Hoperation(data,vaxis,np.min,repeat=repeat)

def Hmax(data,vaxis,repeat=False):
    return Hoperation(data,vaxis,np.max,repeat=repeat)

def rms(data):
    return np.sqrt(np.mean(np.square( data )))

class constants(object):
    """constants in CGS units"""
    Rgas = 8.31446261815324e7      # erg / (mol K)
    hbar = 1.0545716e-27   # erg s
    sigma = 5.6703725e-05  # g / (K**4 * s**3)
    NAvog = 6.022e23
    erg = 1.60217657e-6 # 1 MeV in erg
    c_light = 29979245800 # cm s**-1 CODATA
    G = 6.67428e-8

def format_time(t, unit, digits=2, return_scaled_t=False):
    factors = {'s': 1., 'min': 60., 'hour': 3600., 'day': 24.*3600., 'year': 31556926.}
    ifactors = dict([(v,k) for k, v in factors.items()])
    if isinstance(unit, str):
        fac = factors[unit]
        name = unit
    else:
        fac = unit
        name = ifactors.get(unit, '')

    if not return_scaled_t:
        return ('{0:.%df} {1}'%digits).format(t/fac, name)
    else:
        return t/fac

#class HTML5FuncAnimation(matplotlib.animation.FuncAnimation):
#    VIDEO_TAG = """<video controls>
#    <source src="data:video/x-m4v;base64,{0}" type="video/webm">
#    Your browser does not support the video tag.
#    </video>"""
#
#    @property
#    def encoded_video(self):
#        try:
#            return self._encoded_video
#        except AttributeError:
#            with tempfile.NamedTemporaryFile(suffix='.webm') as f:
#                self.save(f.name, writer='ffmpeg', codec='vp8', extra_args=['-crf', '4', '-b:v', '1M'])
#                self._encoded_video = base64.b64encode(f.read()).decode('ascii')
#            return self._encoded_video
#
#    def _repr_html_(self):
#        return self.VIDEO_TAG.format(self.encoded_video)


class grid_label(object):
    '''class to ease creation of label strings
    e.g.

    grid_label.join(grid_label.hydro_flux or grid_label.resolution)
    '''

    def __init__(self, grid):

        self.grid = grid
        self.sd = self.grid.sd
        self.fname = False

    @property
    def geometry(self):
        geo = self.grid.geometry.type

        if self.fname:
            geo = geo
        elif geo == 'cartesian':
            geo = "Cartesian"
        elif geo == 'polar':
            geo = 'Polar'
        elif geo == 'spherical':
            geo = 'Spherical'
        elif geo == 'curvilinear':
            geo = 'Curvilinear'
        else:
            print("geometry, %s not implemented"%hf)
            geo = 'unknown'
        return geo

    @property
    def wellbalancing(self):
        wb = self.grid.wellbalancing

        if self.fname:
            wb = wb
        elif wb == 'none':
            wb = 'no WB'
        elif wb == 'cargoleroux':
            wb = 'Cargo-LeRoux WB'
        elif wb == 'alphabeta':
            wb = r'$\alpha$-$\beta$ WB'
        elif wb == 'deviation':
            wb = r'Deviation WB'
        else:
            print("wellbalancing, %s not implemented"%wb)
            wb = 'unknown'
        return wb

    @property
    def hydro_flux(self):
        hf = self.sd.hydro_flux

        if hf == 'roe_lowmach2':
            if self.fname:
                hfl = r'roe'
            else:
                hfl = r'Roe'

            if self.sd.Mref < 1.0:
                if self.fname:
                    hf = r'roe_lowmach'
                else:
                    hf = r'Roe lowmach'

        elif hf == 'ausmpup':
            if (self.sd.Mref == 1.0 and self.sd.Mref == 1.0):
                if self.fname:
                    hfl = r'ausmup'
                else:
                    hfl = r'AUSM$^{+}$'
            else:
                if self.fname:
                    hfl = r'ausmpup'
                else:
                    hfl = r'AUSM$^{+}$-up'

        elif hf == 'split_ligu':
            if self.fname:
                hfl = r'ligu'
            else:
                hfl = r'Ligu'
        else:
            print("hydro_flux, flux %s not implemented"%hf)
            hfl = 'unknown'
        return hfl

    @property
    def rc_vars(self):
        rcv = self.sd.rc_hydro.vars
 
        if self.fname:
            return rcv
 
        labels = {'rhop':r'$\rho$-$p$',
                  'rhoT':r'$\rho$-$T$',
                  'cons':r'CV'}
 
        if rcv in labels:
            rcv = labels[rcv]
 
        return rcv

    @property
    def resolution(self):

        if self.grid.sdim == 1:
            raise NotImplementedError(
                "res_label, only 2D and 3D implemented yet")

        if self.fname:
            if self.grid.sdim == 2:
                rstr = r'{}x{}'.format(*self.grid.gnc[:-1])
            elif self.grid.sdim == 3:
                rstr = r'{}x{}x{}'.format(*self.grid.gnc)
        else:
            if self.grid.sdim == 2:
                rstr = r'${}\times{}$'.format(*self.grid.gnc[:-1])
            elif self.grid.sdim == 3:
                rstr = r'${}\times{}\times{}$'.format(*self.grid.gnc)

        return rstr

    @property
    def boost(self):
        boost = self.grid['grid']['networkboost']
        mstr = r'b{:.0e}'.format(boost)

        return mstr

    @property
    def mref(self):
        if self.fname:
            raise NotImplementedError('res_label, file name option not yet implemented')
        mstr = r'Mref = {:.2e}'.format(self.sd.Mref)

        return mstr

    @property
    def mref_pdiff(self):
        if self.fname:
            raise NotImplementedError('res_label, file name option not yet implemented')
        mstr = r'Mref$_\mathrm{{pdiff}}$ = {:.2e}'.format(self.sd.Mref_pdiff)

        return mstr

    def join(self, *args):
        label = args[0]
        for arg in args[1:]:
            label += ' ' + arg

        return label





def ensure_scalar(v):
    if not np.isscalar(v):
        if len(v) != 1:
            raise ValueError('{0} is not scalar'.format(v))
        else:
            v = v[0]
    return v

def ensure_list(v, length=1,strict=False):
    if np.isscalar(v):
        return [v] * length
    elif strict and len(v) != length:
        raise ValueError('ensure_list: list of length %d must have length of %d'%(len(v),length))
    else:
        return v
