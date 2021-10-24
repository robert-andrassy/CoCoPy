import numpy as np
#!pwd
#!ls
#!cat get-data.f
class fdmp():
    def __init__(self, filename):
        f = open(filename, 'rb')
        lens = np.fromfile(f, dtype='int32', count=2)
#    print('lens=',lens)
        aaa = np.fromfile(f, dtype='float32', count=lens[0])
        iaaa = np.fromfile(f, dtype='int32', count=lens[1])
        nvars = int(np.fromfile(f, dtype='int32', count=1))

        self.time = aaa[6-1]
        self.airmu= aaa[18-1]
        self.cldmu= aaa[19-1]
        self.deex=aaa[90-1]
        self.radin0=aaa[32-1]
        self.Rgasconst=aaa[24-1]
        self.totallum=aaa[29-1]
        self.gamma=aaa[10-1]

        nsugar = iaaa[7-1]
        NXBricks = iaaa[23-1]
        NYBricks = iaaa[24-1]
        NZBricks = iaaa[25-1]
        NBricksPerTeam = iaaa[32-1]
        nnbqs = iaaa[43-1]

        ndim=nsugar*NXBricks*nnbqs
        nxcellsinbk=nnbqs*nsugar
#        print('grid resolution:',ndim)
        self.ndim=ndim

        dmp=np.zeros((ndim,ndim,ndim,nvars))

        for bk in range(0,NBricksPerTeam):
            ijkbk = int(np.fromfile(f, dtype='int32', count=1))
#            print(ijkbk)
            kbk0=int((ijkbk-1)/(NXBricks*NYBricks))
            jbk0=int((ijkbk - kbk0*(NXBricks*NYBricks)-1)/NXBricks)
            ibk0 = ijkbk - kbk0*(NXBricks*NYBricks) - jbk0*NXBricks-1
#        print(ijkbk,ibk0,jbk0,kbk0)
            icoffs=np.array([ibk0,jbk0,kbk0])*nxcellsinbk
            for ivar in range(0,nvars):
                for iz in range(0,nxcellsinbk):
                    for iy in range(0,nxcellsinbk):
                        dmp[icoffs[0]:(icoffs[0]+nxcellsinbk),icoffs[1]+iy,icoffs[2]+iz,ivar]= \
                      np.array(np.fromfile(f, dtype='float32', count=nxcellsinbk))

        self.rho1=dmp[:,:,:,0]
        self.p1=dmp[:,:,:,1]
        self.ux=dmp[:,:,:,2]
        self.uy=dmp[:,:,:,3]
        self.uz=dmp[:,:,:,4]
        self.fv=dmp[:,:,:,5]
        self.fvair=dmp[:,:,:,6]
        self.rho0=dmp[:,:,:,7]
        self.p0=dmp[:,:,:,8]
        self.mu0=dmp[:,:,:,9]

        vel=np.zeros((3,ndim,ndim,ndim))
        vel[0]=dmp[:,:,:,2]
        vel[1]=dmp[:,:,:,3]
        vel[2]=dmp[:,:,:,4]
        self.vel=vel


        xcoord=np.zeros((ndim,ndim,ndim))
        ycoord=np.zeros((ndim,ndim,ndim))
        zcoord=np.zeros((ndim,ndim,ndim))
        for i in range(0,ndim):
            xcoord[i,:,:]=0.5*self.deex+(i-ndim/2)*self.deex
            ycoord[:,i,:]=0.5*self.deex+i*self.deex+self.radin0
            zcoord[:,:,i]=0.5*self.deex+(i-ndim/2)*self.deex

        self.xcoord=xcoord
        self.ycoord=ycoord
        self.zcoord=zcoord

        coords=np.zeros((3,ndim,ndim,ndim))
        coords[0]=xcoord
        coords[1]=ycoord
        coords[2]=zcoord
        self.coords=coords

def filename(runid,dump,suffix):
    namestr=("{:s}-"+"{:04d}."+"{:s}").format(runid,dump,suffix)
    return namestr

#print(filename('SHDV',3,'aaa'))

#filename='SHDVtest128-0500.aaa'
#dump=fdmp(filename)
#print(dump.p0.shape,dump.rho0.shape,dump.mu0.shape,\
#dump.rho1.shape,dump.p1.shape,dump.ux.shape,\
#dump.uy.shape,dump.uz.shape,dump.xcoord.shape,dump.ycoord.shape,dump.zcoord.shape)
