from imtools import *
from ftools import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def parseLabelFolder(trainPath):
    imPathList = listfiles(trainPath,'.tif')
    imLabelList = listfiles(trainPath,'.png')
    nClasses = int(len(imLabelList)/len(imPathList))

    nSamples = 0
    imList = []
    lbList = []
    for imPath in imPathList:
        [p,n,e] = fileparts(imPath)
        imList.append(im2double(tifread(imPath)))
        L = []
        nSamplesPerClass = np.zeros(nClasses)
        for iClass in range(nClasses):
            lPath = pathjoin(p,'%s_Class%d.png' % (n,iClass+1))
            Li = imread(lPath)
            if len(Li.shape) > 2:
                Li = Li[:,:,0]
            L.append(Li)
            nSamplesPerClass[iClass] = np.sum(Li > 0)
        iMin = np.argmin(nSamplesPerClass)
        vMin = nSamplesPerClass[iMin]
        for iClass in range(nClasses):
            if iClass != iMin:
                L[iClass] = L[iClass]*(np.random.rand(Li.shape[0],Li.shape[1]) < vMin/nSamplesPerClass[iClass])
            nSamples += np.sum(L[iClass] > 0)
        lbList.append(L)

    return nClasses, nSamples, imList, lbList

def setupTraining(nClasses,nSamples,imList,lbList,sigmaDeriv=2,sigmaLoG=[],cfRadii=[],cfSigma=2,cfThr=0.75,cfDst=0.25):
    # sigmaDeriv: image is convolved with gaussian kernel of this scale before computing derivatives
    # sigmaLoG: laplacian of gaussian scale
    # cfRadii: list of radii on which to compute circularity features
    # cfSigma: image is convolved with gaussian kernel of this scale before computing circularity features
    # cfThr: threshold to decide to draw circles/disks from center likelihoods (see imtools.circlikl for details)
    # cfDst: density with which to draw disks from centers that pass cfThr (see imtools.circlikl for details)
    featNames = imfeatures(sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,cfRadii=cfRadii,justfeatnames=True)
    nFeatures = len(featNames)
    X = np.zeros((nSamples,nFeatures))
    Y = np.zeros((nSamples))
    i0 = 0
    for iImage in range(len(imList)):
        I = imList[iImage]
        F = imfeatures(I=I,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,cfRadii=cfRadii,cfSigma=cfSigma,cfThr=cfThr,cfDst=cfDst)
        for iClass in range(nClasses):
            Li = lbList[iImage][iClass]
            indices = Li > 0
            l = np.sum(indices)
            x = np.zeros((l,nFeatures))
            for iFeat in range(nFeatures):
                Fi = F[:,:,iFeat]
                xi = Fi[indices]
                x[:,iFeat] = xi
            y = (iClass+1)*np.ones((l))
            X[i0:i0+l,:] = x
            Y[i0:i0+l] = y
            i0 = i0+l
    return X, Y, {'nClasses': nClasses,
                  'nFeatures': nFeatures,
                  'featNames': featNames,
                  'sigmaDeriv': sigmaDeriv,
                  'sigmaLoG': sigmaLoG,
                  'cfRadii': cfRadii,
                  'cfSigma': cfSigma,
                  'cfThr': cfThr,
                  'cfDst': cfDst}

def rfcTrain(X,Y,params):
    rfc = RandomForestClassifier(n_estimators=10)
    rfc = rfc.fit(X, Y)
    model = params
    model['rfc'] = rfc
    model['featImport'] = rfc.feature_importances_
    return model

def train(trainPath,**kwargs):
    nClasses, nSamples, imList, lbList = parseLabelFolder(trainPath)
    X, Y, params = setupTraining(nClasses,nSamples,imList,lbList,**kwargs)
    model = rfcTrain(X,Y,params)
    return model

def classify(I,model,output='classes'):
    rfc = model['rfc']
    F = imfeatures(I=I,
                   sigmaDeriv=model['sigmaDeriv'],
                   sigmaLoG=model['sigmaLoG'],
                   cfRadii=model['cfRadii'],
                   cfSigma=model['cfSigma'],
                   cfThr=model['cfThr'],
                   cfDst=model['cfDst'])
    sI = size(I)
    M = np.zeros((sI[0],sI[1],model['nClasses']))
    if output == 'classes':
        out = rfc.predict(F.reshape(-1,model['nFeatures']))
        C = out.reshape(I.shape)
        for i in range(model['nClasses']):
            M[:,:,i] = C == i+1
    elif output == 'probmaps':
        out = rfc.predict_proba(F.reshape(-1,model['nFeatures']))
        for i in range(model['nClasses']):
            M[:,:,i] = out[:,i].reshape(I.shape)
    return M