import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

model = tf.keras.models.load_model('./static/models/model.h5')

def objectdetection(path,filename):
    image = load_img(path)
    image = np.array(image,dtype=np.uint8)
    image1 = load_img(path,target_size=(224,224))
    imagearr224=img_to_array(image1)/225.0
    h,w,d = image.shape
    testarr = imagearr224.reshape(1,224,224,3)
    cords = model.predict(testarr)
    denorm = np.array([w,w,h,h])
    cords = cords * denorm
    cords = cords.astype(np.int32)
    xmin,xmax,ymin,ymax = cords[0]
    pt1 = (xmin,ymin)
    pt2 = (xmax,ymax)
    print(pt1,pt2)
    cv2.rectangle(image,pt1,pt2,(0,225,0),5)
    imagebgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),imagebgr)
    return cords

def OCR(path,filename):
    img = np.array(load_img(path))
    cods = objectdetection(path,filename)
    xmin,xmax,ymin,ymax = cods[0]
    roi=img[ymin:ymax,xmin:xmax]
    roibgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),roibgr)
    text = pt.image_to_string(roi)
    print(text)
    return text