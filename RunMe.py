import pixelclassifier as pc
from imtools import im2double, tifread, imshow, stack2list, imshowlist
import matplotlib.pyplot as plt


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


# test

path = '/home/mc457/Workspace/DataForPC/Test/I003.tif'
I = im2double(tifread(path))

C = pc.classify(I,model,output='classes')
P = pc.classify(I,model,output='probmaps')

imshowlist([I]+stack2list(C)+stack2list(P))