# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 2017

@author: Nicolas Loffreda

Module to read all the images located on a particular folder, convert them to
an np array, flatten them and put them together on an np Array format to use as
Data Matrix (Feature Vector)
"""

import os
import numpy as np
from PIL import Image
import pylab as pl

def readImages(path, chk_shape=(50, 50)):
    '''
    Read all images from a directory and return an np Array X to use
    as data matrix (Feature Vectors).
    The function will also check for a specific shape and raise an error if 
    dimensions don't match what's expected (usually 50x50, but can be changed)
    
    Parameters & Return
    ----------
    :param path: Path where the images are located
    :param chk_shape: Tuple (1x2) to check image shape
    :return X: Data matrix with each image on a flattened format
    
    Example
    -----------
    :Example:
        
    >> X = readImages("./imgs", chk_shape=(145,148))
    >> X = readImages("./imgs")
    '''
    # Read Images
    images = list()
    class_labels = list()
    for img in os.listdir(path):
        if img.endswith(".png") or img.endswith(".jpg"):
            im = Image.open(os.path.join(path, img))
            im_grey = im.convert('L')
            im_array = np.array(im_grey)
            if im_array.shape != chk_shape:
                raise ValueError('{} doesnt have the expected dim {}, {} instead'.format(img, chk_shape, im_array.shape))
            images.append(im_array)
            class_labels.append(int(img[1:2]))
            
    # Flatten the images and append to Data matrix
    flatimages = list()
    for i in images:
        flatimages.append(i.ravel())
    X = np.asarray(flatimages)
    Y = np.asarray(class_labels)
    return X, Y


def plotImages(*args, dm=(50, 50)):
    '''
    Plot one or multiple images from the X matrix. Takes flatten images as input, 
    so need to set up the width and length to re-shape the vector image if different
    than 50x50. This will be set up as 
    
    Examples
    -------------
    :Example:
        
    >> plotImages(X[0], dm=(145, 148))
    >> plotImages(X[100])
    '''
    n = len(args)
    fig = pl.figure()
    for i,arg in enumerate(args):
        pl.subplot(1, n, i+1)
        pl.imshow(arg.reshape(dm[0], dm[1]), interpolation='None', cmap=pl.get_cmap('gray'))
        pl.axis('off')
    fig.tight_layout(pad=0)
    pl.show()
    