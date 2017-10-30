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
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
from tkinter import Tk, Message, mainloop
from collections import Counter

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
    return X

def Ensemble_predict(query, params):
    query_a = np.hstack((np.ones((len(query),1)), query))

    C = np.tanh(np.dot(query_a, params['w'].T))
    Ca = np.hstack((np.ones((len(C),1)), C))

    pred = np.argmax(np.dot(Ca, params['e']), axis=1)
   
    return pred


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
    -croped face image with resolution (50px, 50px)
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
        
    #####################################################
    #### 2) Grab the New sample to test the model on ####
    #####################################################
    '''
    This portion of the code gets the sample of the new data to predict.
    Will take 10 pictures and put it on an np array, X_test.
    It will create a folder called "photo_test"and one called "face_test" on the directory
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
    image_path = 'photo_test'
    start_id = 0
    resolution = (50, 50)
    class_id = 9
    
    image_capture('photo_test', n, start_id)
    face_crop('photo_test', 'face_test', class_id, resolution)
    X_test = readImages('./face_test')
    ####################################
    ##### Finished grabbing sample #####
    #################################### 
    
    ################
    ### 3) Model ###
    ################
    '''
    Applying linear model to try to predict class based on sample data
    '''
    # Load parameters for trained Guassian model
    npzfile = np.load('PCA_2D.npz')
    mu = npzfile['mu']
    V = npzfile['V']

    params = np.load("Ensemble_2D.npy")
    params = params[()]
    
    # Utilize classifier to get the name of a photo
    # Apply PCA to the test set
    Z_test = X_test - mu
    P_test = np.dot(Z_test,V.T)
    
    # Resulting data frame
    # Get predictions
    pred = Ensemble_predict(P_test, params)
    print (pred)

    # Get the most common class of the n test samples
    PREDICTION_CLASS = Counter(pred).most_common(1)[0][0]
    
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