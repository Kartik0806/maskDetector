import streamlit as st
import cv2
import keras
from keras.models import load_model
from keras import preprocessing
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

st.header('Mask Detector')

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_label = {0:'MASK',1:'NO MASK'}

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt",
                                            caffeModel="res10_300x300_ssd_iter_140000.caffemodel")


def main():
    file_uploaded=st.file_uploader('Choose the file', type=['jpg','jpeg','jpg','png'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        image = image.convert('RGB')
        predict_class(image)
def predict_class(image):
    model=keras.models.load_model('augmented.h5',compile=False)
    # model=keras.models.load_model('modelnew.h5',compile=False)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='adam')
    image_array = np.asarray(image)
    H,W,C=image_array.shape
    print(H,W,C)
    preprocessed_image = cv2.dnn.blobFromImage(image_array, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    opencv_dnn_model.setInput(preprocessed_image)
    results = opencv_dnn_model.forward()    

    faces = face_model.detectMultiScale(image_array,scaleFactor=1.1, minNeighbors=5)
    height=64
    width=64
    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > 0.2:
            bbox = face[3:]
            x1 = int(bbox[0] * W)
            y1 = int(bbox[1] * H)
            x2 = int(bbox[2] * W)
            y2 = int(bbox[3] * H)
            print(x1,x2,y1,y2)
            crop = image_array[y1:y2,x1:x2]
            # hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # value=30
            # h, s, v = cv2.split(hsv)
            # lim = 255 - value
            # v[v > lim] = 255
            # v[v <= lim] += value
            # final_hsv = cv2.merge((h, s, v))
            # crop = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            # cv2.convertScaleAbs(crop, 5, -127)
            crop = cv2.resize(crop,(height,width))
            plt.imshow(crop)
            crop = np.reshape(crop,[1,height,width,3])
            predictions = model.predict(crop)
            org = (20, 20)
            score = float(predictions[0])
            print(score)
            arg=0
            if(score>0.001):
                arg=1
            cv2.putText(image_array,mask_label[arg],(x1+10,y1+30),cv2.FONT_HERSHEY_SIMPLEX,color=(36,255,12), thickness=2,fontScale=0.8)
            cv2.rectangle(image_array,(x1,y1),(x2,y2),(255,0,0), thickness=3)
    #         print(mask_label[arg])
    #     # plt.imshow(crop)

        # cv2.putText(image_array,mask_label[arg],org,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
    # cv2.rectangle(image_array,(x1,y1),(x2,y2),1, thickness=3)
    st.image([image_array]) 
    # figure=plt.figure()
    # plt.imshow(image_array)
    # plt.axis('off')
    # st.pyplot(figure)






    # img=image
    # if(len(faces)==0):
    #     st.write("No Faces Detected")
    #     st.image([img])
    # else:
    #     for i in range(len(faces)):
    #         (x,y,w,h) = faces[i]
    #         crop = image_array[y:y+h,x:x+w]
    #         st.image([crop]) 
    #         crop = cv2.resize(crop,(64,64))
    #         st.image([crop])
    #         crop = np.reshape(crop,[1,64,64,3])
    #         predictions = model.predict(crop)
    #         print(predictions[0][0])
    #         org = (x+20, y+40)
    #         score = float(predictions[0])   
    #         arg=0
    #         if(score>0.5):
    #             arg=1
    #         print(x,y,x+w,y+h)
    #         cv2.putText(image_array,mask_label[arg],org,cv2.FONT_HERSHEY_SIMPLEX,color=(255,0,0), thickness=2,fontScale=0.8)
    #         cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0), thickness=3)
        # st.image([image_array]) 
        # figure=plt.figure()
        # plt.imshow(image_array)
        # plt.axis('off')
        # st.pyplot(figure)

if __name__=="__main__":
    main()
