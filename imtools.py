import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np
from skimage.io import *
from skimage.filters import gabor_kernel, gabor
from skimage.transform import EuclideanTransform, warp
from scipy.ndimage import *

def tifread(path):
    return tifffile.imread(path)

def tifwrite(I,path):
    tifffile.imsave(path, I)

def imshow(I,**kwargs):
    if not kwargs:
        plt.imshow(I,cmap='gray')
    else:
        plt.imshow(I,**kwargs)
        
    plt.axis('off')
    plt.show()

def imshowlist(L,**kwargs):
    n = len(L)
    for i in range(n):
        plt.subplot(1, n, i+1)
        if not kwargs:
            plt.imshow(L[i],cmap='gray')
        else:
            plt.imshow(L[i],**kwargs)
        plt.axis('off')
    plt.show()

def imwrite(I,path):
    imsave(path,I)

def im2double(I):
    if I.dtype == 'uint16':
        return I.astype('float64')/65535
    elif I.dtype == 'uint8':
        return I.astype('float64')/255
    elif I.dtype == 'float32':
        return I.astype('float64')
    elif I.dtype == 'float64':
        return I
    else:
        print('returned original image type: ', I.dtype)
        return I

def size(I):
    return list(I.shape)

def normalize(I):
    m = np.min(I)
    M = np.max(I)
    if M > m:
        return (I-m)/(M-m)
    else:
        return I

def snormalize(I):
    m = np.mean(I)
    s = np.std(I)
    if s > 0:
        return (I-m)/s
    else:
        return I

def imgaussfilt(I,sigma,**kwargs):
    return gaussian_filter(I,sigma,**kwargs)

def imlogfilt(I,sigma,**kwargs):
    return -gaussian_laplace(I,sigma,**kwargs)

def imderivatives(I,sigmas):
    if type(sigmas) is not list:
        sigmas = [sigmas]
    nDerivatives = len(sigmas)*8 # d0,dx,dy,dxx,dxy,dyy,sqrt(dx^2+dy^2),sqrt(dxx^2+dyy^2)
    sI = size(I)
    D = np.zeros((sI[0],sI[1],nDerivatives))
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        D[:,:,8*i  ] = imgaussfilt(I,sigma)
        D[:,:,8*i+1] = imgaussfilt(I,sigma,order=[0,1])
        D[:,:,8*i+2] = imgaussfilt(I,sigma,order=[1,0])
        D[:,:,8*i+3] = imgaussfilt(I,sigma,order=[0,2])
        D[:,:,8*i+4] = imgaussfilt(I,sigma,order=[1,1])
        D[:,:,8*i+5] = imgaussfilt(I,sigma,order=[2,0])
        D[:,:,8*i+6] = np.sqrt(D[:,:,8*i+1]**2+D[:,:,8*i+2]**2)
        D[:,:,8*i+7] = np.sqrt(D[:,:,8*i+3]**2+D[:,:,8*i+5]**2)
    return D

def circcentlikl(I,radius,scale=2,n0piAngles=8):
    angles = np.arange(0,np.pi,np.pi/n0piAngles)
    A = np.zeros(I.shape)
    for i in range(len(angles)):
        angle = angles[i]
        K = gabor_kernel(1/scale,angle).imag
        J = convolve(I,K)
        dx = -radius*np.cos(angle)
        dy = -radius*np.sin(angle)
        T = EuclideanTransform(translation=(-dx,-dy))
        L1 = warp(J,T)
        T = EuclideanTransform(translation=(dx,dy))
        L2 = warp(-J,T)

        # imshowlist([I, resize(K,J.shape), J, np.multiply(L1,L1 > 0), np.multiply(L2,L2 > 0)])
        A += np.multiply(L1,L1 > 0)+np.multiply(L2,L2 > 0)
    return A

def circlikl(I,radii,scale=2,n0piAngles=8,thr=0.75,dst=0.25):
    # warning: radii should either be a number or a python list (not a numpy array)
    C1 = np.zeros(I.shape)
    C2 = np.zeros(I.shape)
    if type(radii) is not list:
        radii = [radii]
    for i in range(len(radii)):
        radius = radii[i]
        A = circcentlikl(I,radius,scale,n0piAngles)
        Cr1 = np.zeros(I.shape)
        Cr2 = np.zeros(I.shape)
        r0,c0 = np.where(A > thr*np.max(A))
        for j in range(len(r0)):
            A0 = A[r0[j],c0[j]]
            for angle in np.arange(0,2*np.pi,1/radius):
                row = int(np.round(r0[j]+radius*np.cos(angle)))
                col = int(np.round(c0[j]+radius*np.sin(angle)))
                if row > -1 and row < I.shape[0] and col > -1 and col < I.shape[1]:# and I[row,col] > 15:
                    Cr1[row,col] += A0
                    # Cr1[row,col] = np.max([Cr1[row,col],A0])
            rs = np.arange(1,radii[i]+1,1)
            rs = rs[np.where(np.random.rand(len(rs)) < dst)]
            for r in rs:
                angles = np.arange(0,2*np.pi,1/r);
                angles = angles[np.where(np.random.rand(len(angles)) < dst)]
                for angle in angles:
                    row = int(np.round(r0[j]+r*np.cos(angle)))
                    col = int(np.round(c0[j]+r*np.sin(angle)))
                    if row > -1 and row < I.shape[0] and col > -1 and col < I.shape[1]:
                        Cr2[row,col] += A0
                        # Cr2[row,col] = np.max([Cr2[row,col],A0])
        C1 = np.maximum(C1,Cr1)
        C2 = np.maximum(C2,Cr2)
    CL = np.zeros((I.shape[0],I.shape[1],2))
    CL[:,:,0] = imgaussfilt(C1,scale)
    CL[:,:,1] = imgaussfilt(C2,scale)
    return CL

def imfeatures(I=[],sigmaDeriv=1,sigmaLoG=1,cfRadii=[],cfSigma=2,cfThr=0.75,cfDst=0.25,justfeatnames=False):
    # warning: cfRadii should either be a number or a python list (not a numpy array)
    if type(sigmaDeriv) is not list:
        sigmaDeriv = [sigmaDeriv]
    if type(sigmaLoG) is not list:
        sigmaLoG = [sigmaLoG]
    if type(cfRadii) is not list:
        cfRadii = [cfRadii]
    nDerivFeats = len(sigmaDeriv)*8
    nLoGFeats = len(sigmaLoG)
    nCircFeats = len(cfRadii)*2
    nFeatures = nDerivFeats+nLoGFeats+nCircFeats
    if justfeatnames == True:
        featNames = []
        derivNames = ['d0','dx','dy','dxx','dxy','dyy','normD1','normD2']
        for i in range(len(sigmaDeriv)):
            for j in range(len(derivNames)):
                featNames.append('derivSigma%d%s' % (sigmaDeriv[i],derivNames[j]))
        for i in range(len(sigmaLoG)):
            featNames.append('logSigma%d' % sigmaLoG[i])
        for i in range(len(cfRadii)):
            featNames.append('cfRad%dCirc' % cfRadii[i])
            featNames.append('cfRad%dDisk' % cfRadii[i])
        return featNames
    sI = size(I)
    F = np.zeros((sI[0],sI[1],nFeatures))
    F[:,:,:nDerivFeats] = imderivatives(I,sigmaDeriv)
    for i in range(nLoGFeats):
        F[:,:,nDerivFeats+i] = imlogfilt(I,sigmaLoG[i])
    for i in range(len(cfRadii)):
        F[:,:,nDerivFeats+nLoGFeats+2*i:nDerivFeats+nLoGFeats+2*(i+1)] = circlikl(I,cfRadii[i],scale=cfSigma,thr=cfThr,dst=cfDst)
    return F

def stack2list(S):
    L = []
    for i in range(size(S)[2]):
        L.append(S[:,:,i])
    return L