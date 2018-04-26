import pixelclassifier as pc
from imtools import *
import matplotlib.pyplot as plt
from skimage.morphology import local_maxima, disk, watershed

# -------------------------
# train

trainPath = '/home/mc457/Workspace/DataForPC/Train'
model = pc.train(trainPath,sigmaDeriv=[2,4,8,16],sigmaLoG=[2,4,8,16])

def plotFeatImport(fi,fn):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    y_pos = range(len(fn))
    ax.barh(y_pos, fi)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fn)
    ax.invert_yaxis()
    ax.set_title('Feature Importance')
    plt.show()

plotFeatImport(model['featImport'],model['featNames'])

# -------------------------
# segment

path = '/home/mc457/Workspace/DataForPC/Train/I003.tif'
I = im2double(tifread(path))

C = pc.classify(I,model,output='classes')
P = pc.classify(I,model,output='probmaps')

imshowlist([I]+stack2list(C)+stack2list(P))

# -------------------------
# watershed 

def pmwatershed(I,estRad,smCoef,fgThr,frg):
    fwhm = estRad # full width at half maximum 
    sigma = fwhm/2.355
    J = imlogfilt(I,sigma)
    S = imgaussfilt(I,smCoef)
    M =  S > fgThr
    diskCoef = (1-frg)*fwhm/4+frg
    d = disk(2*int(diskCoef)+1)
    lm = local_maxima(J,selem=d)*M
    markers = label(lm)
    labels = watershed(-S, markers[0], mask=M, watershed_line=True)
    return labels > 0

estRad = 25 # estimated radius of nuclei (used to find markers)
smCoef = 2 # smoothness coefficient (sigma with which to gaussian blur prob. map)
fgThr = 0.4 # foreground threshold; 0 (everything foreground) to 1
frg = 0 # fragmentation; 0 (less) to 1 (more)

W = pmwatershed(P[:,:,2],estRad,smCoef,fgThr,frg)

imshowlist([I, C[:,:,2], P[:,:,2], W])