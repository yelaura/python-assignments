# -*- coding: utf-8 -*-
"""
@author: Team Cortes
@professor: Shashidhar Sathyanarayana 

Final Project code for Berkeley Extension course: Introduction to Machine
Learning with Python
"""
import cv2
import os
import re
import operator
import numpy as np
import pandas as pd
from PIL import Image
import pylab as pl
from PCA import applyPCA
from sklearn.metrics import confusion_matrix
from datetime import datetime, timedelta
from tkinter import Tk, Message, mainloop

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
    Y = Y.reshape((np.alen(Y), 1))
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
    
def calcMultiLinPred(Xa, W):
    '''
    Get the class number out of the values of W
    '''
    pred_vals = np.dot(Xa, W)
    pred = []
    for varr in pred_vals:
        mx = varr.max()
        for i,v in enumerate(varr):
            if varr[i] == mx:
                pred.append(i)
    
    preds = np.array(pred)
    
    return preds


def image_capture(image_path, n, start_id):
    """
    Take a series of images with face from vedio
    
    Input:
    -image_path: path of image folder to store captured images.
    -n: number of images to capture. 
    -start_id: the start number of image used in filename. Default start from 0.
    
    Output:
    -n images with faces stored in designiated image folder.
    """
    # senitize path and create folder if not exists
    image_path = image_path
    if not image_path.endswith('\\'):
        image_path +='\\'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        
    img_id = start_id
    end_id = img_id+n

    start = False
    oldtime = datetime.now()
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while(img_id<end_id):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face tracking on gray image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame  
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start = True
            oldtime = datetime.now()

        # take next image after 0.5 second if face is detected.
        newtime = datetime.now()
        diff = (newtime-oldtime).seconds

        if len(faces) and start and diff>0.5:               
            filename = 'img_%i.jpg'%img_id
            cv2.imwrite(str(image_path)+filename,frame)
            img_id+=1
            oldtime = datetime.now()
        
        if start:
            cv2.putText(gray, "Capture image {0}; last image is {1}".format(img_id, end_id-1), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        else:
            cv2.putText(gray, "Press s to start, press q to quit!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        cv2.imshow('frame',gray)
        faces = None

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
def face_crop(image_path, face_path, class_id, resolution):
    """
    crop out face region from images, save as filename cX_imgageY_face_Z.jpg X is the class number, Y is yth image of X class, Z is zth face detected in imageY. 
    
    Input:
    -image_path: path of image folder to be processed, images from same class are stored in same folder.
    -face_path: path of processed images
    -class_id: class number of face images.
    -resolution: resize resolution (px, px)
    
    Output:
    -croped face image with resolution (100px, 100px)
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    class_id = class_id
    image_path = image_path
    face_path = face_path
    img_id = None
    
    #senitize check of path, create face folder if not exists
    if not image_path.endswith('\\'):
        image_path +='\\'
    if not face_path.endswith('\\'):
        face_path +='\\'   
    if not os.path.exists(face_path):
        os.mkdir(face_path)
    
    #crop face from images
    for file in os.listdir(image_path):
        if file.endswith('.jpg'):
            img_id = re.findall(r'img_(\d+).jpg', file)[0]
        if img_id:
            filename = 'img_%s.jpg'%img_id
            img = cv2.imread(str(image_path)+filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            i = 1
            for (x,y,w,h) in faces:
                output = 'c{0}_image{1}_face{2}.jpg'.format(class_id, img_id, i)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, resolution)
                cv2.imwrite(str(face_path)+output,roi_gray)
                i += 1
        img_id = None

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Dictionary with class and keys
    name_dict = {0:"Pryianka", 
                 1:"Weiya", 
                 2:"Nicolas", 
                 3:"Krupa", 
                 4:"Mukul", 
                 5:"Laura"}
    
    #######################################################
    ### 1) Read Data and put in np array and Data Frame ###
    #######################################################
    '''
    This portion of the code reads the images from a specified folder, convert 
    them to an np array, matrix X, and to a pandas DataFrame.
    
    Inputs:
    + PCA_COMPS (int): Number of PCA components to keep on P matrix
    + PATH: Path were images of the Training Data Set are
    
    Outputs:
    + M, C, P, X_rec (np array): Matrices and vectors used on PCA
    + X (np array): Feature array with flatten images (nx2500)
    + Y (np array): Class label array with corresponding classes to each image (2500, 1)
    + df (pandas DataFrame): Data Frame with P matrix and class labels (columns labeled appropriately)
    '''
    # Define number of PCA components to use
    PCA_COMPS = 2
    PATH = "./imgs"
    
    # Load data from images
    X, Y = readImages(PATH, chk_shape=(50, 50)) 
    # Apply PCA
    M, C, P, X_rec = applyPCA(X, components=PCA_COMPS, return_code="MCPX")
    # Create pandas Data Frame
    intercept = np.array([1 for i in range(np.alen(Y))]).reshape((np.alen(Y), 1))
    df = pd.DataFrame(np.concatenate((intercept, P, Y), axis=1)) 
    df.columns = ['intercept']+["P%d" %(i+1) for i in range(PCA_COMPS)]+["Y"]
    # Scatter plot of PCA's first 2Ds
    pl.scatter(df['P1'], df['P2'], alpha=0.5, c=df['Y']) 
    
    # Kesler construction for classes
    types = [(np.array([1 if i==0 else -1 for i in df['Y']]), 'type0'),
             (np.array([1 if i==1 else -1 for i in df['Y']]), 'type1'),
             (np.array([1 if i==2 else -1 for i in df['Y']]), 'type2'),
             (np.array([1 if i==3 else -1 for i in df['Y']]), 'type3'),
             (np.array([1 if i==4 else -1 for i in df['Y']]), 'type4'),
             (np.array([1 if i==5 else -1 for i in df['Y']]), 'type5')]
    for t in types:
        df[t[1]] = t[0]
        
    ###########################
    ### Finish reading Data ###
    ###########################
    
    #####################################################
    #### 2) Grab the New sample to test the model on ####
    #####################################################
    '''
    This portion of the code gets the sample of the new data to predict.
    Will take 10 pictures and put it on an np array, X_test.
    It will create a folder called "photo"and one called "face" on the directory
    where the code is placed.
    It will also create a Y_label with a class that isn't any of us. This can be
    disregarded.
    
    Inputs:
    + n (int): Number of images to take as sample, can't be 1
    + image_path (str): Name of the path to place photos, recommend not changing it
    + start_id, resolution, class_id: Leave as it is
    
    Outputs:
    + X_test (np array): Array with sample data to apply the classifier (nx2500)
    + Y_test (np array): Array with undefined class, used to get the length of the sample data only
    '''
    n = 10 # Define how many samples to test on (can't be 1)
    image_path = 'photo'
    start_id = 0
    resolution = (50, 50)
    class_id = 9
    
    filelist = [ f for f in os.listdir("./photo") ]
    for f in filelist:
        os.remove(os.path.join("./photo", f))
        
    filelist = [ f for f in os.listdir("./face") ]
    for f in filelist:
        os.remove(os.path.join("./face", f))
    
    image_capture(image_path, n, start_id)
    face_crop('photo', 'face', class_id, resolution)
    X_test, Y_test = readImages('./face')
    #########################################
    ##### Finished grabbing sample data #####
    #########################################
    
    ################
    ### 3) Model ###
    ################
    '''
    Apply your own classifier to predict the X_test data here.
    Need to output a prediction class.
    
    Outputs:
    + PREDICTION_CLASS (int): Class predicted by the model
    '''
    PREDICTION_CLASS = 0
    
    
    ####################
    ### Finish Model ###
    ####################
    
    #############################################################
    #### 4) Display message with greeting for prediction name ###
    #############################################################
    '''
    This portion of the code will create a Pop up window with the name 
    corresponding to the predicted class by the model.
    It takes the names and classes from the name_dict created on the first line of code
    
    Inputs:
    + PREDICTED_CLASS (int): Predicted class number from model
    '''
    mw = Tk()
    msg = Message(mw, text="Hello {}!".format(name_dict[PREDICTION_CLASS]))
    msg.config(font=("Courier", 44))
    msg.config(width=800)
    msg.config(fg='red')
    msg.pack()
    mainloop()
    ##############################################################
    #### End Display message with greeting for prediction name ###
    ##############################################################