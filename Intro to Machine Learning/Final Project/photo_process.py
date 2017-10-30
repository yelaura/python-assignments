import cv2
import numpy as np
import os
import re
from datetime import datetime, timedelta

resolution = (50, 50)
class_id = 0

# crop out face region within images, resize to 100x100 and save face images to faces folder
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

# crop face out of images in folder 'weiya', store processed face images in 'faces' folder, class label is 1. Filename is c1_imageX_faceY.jpg

face_crop('photo', 'faces', class_id, resolution)