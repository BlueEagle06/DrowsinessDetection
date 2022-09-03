

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import keras
import tensorflow as tf
import os
from pygame import mixer

faces=cv2.CascadeClassifier(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML3\haar_cascades\haarcascade_frontalface_alt.xml")
leye=cv2.CascadeClassifier(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML3\haar_cascades\haarcascade_lefteye_2splits.xml")
reye=cv2.CascadeClassifier(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML3\haar_cascades\haarcascade_righteye_2splits.xml")

#convert to gray
mixer.init()
sound=mixer.Sound(r"D:\Dekstop\Dekstop folders\PC stuff\also PC stuff\Tracks\ProdConferenceTrack_Final_mixdown.mp3")

path=os.getcwd()

cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
model=keras.models.load_model(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML3\drowsiness_model_final")
score=0
thicc=0
dict1={1:"closed",0:"open"}

#x_train=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_train.npy")
#from sklearn.preprocessing import MaxAbsScaler
#scaler=MaxAbsScaler()
#scaler.fit(x_train.reshape(-1,64*64*3))
a=0

while True:
    leyes=0
    new=0
    faces_new=0
    ret,frame=cap.read()
    height,width = frame.shape[:2]

    gray_frames=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces_new=faces.detectMultiScale(gray_frames)
        
    
    for (x,y,w,h) in faces_new:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
        new=frame[y:y+h,x:x+w]
    new=cv2.resize(new,(height,width))

    gray_frames=0

    #cv2.imshow("d",gray_frames)
    try:
        gray_frames=cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
        leyes=leye.detectMultiScale(gray_frames)
        reyes=reye.detectMultiScale(gray_frames)
        #eyes=eye.detectMultiScale(gray_frames)

        print(leyes,len(leyes))
        

    
        #gray_frames=cv2.cvtColor(frame_new,cv2.COLOR_BGR2GRAY)
        

        #for (x,y,w,h) in leyes:
        #    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #for x,y,w,h in reyes:
        #    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
        for (x,y,w,h) in leyes: 

            leye_roi=new[y:y+h,x:x+w]
            #leye_roi_temp=cv2.resize(leye_roi,(height,width))

            #cv2.imshow("eye",leye_roi_temp)
            leye_roi=cv2.resize(leye_roi,(64,64))
            leye_roi=leye_roi/255
            leye_roi=leye_roi.reshape(-1,64,64,3)
            #print(leye_roi.shape)
            prediction=np.round(model.predict(leye_roi))
            #label_left=int(prediction)
            label_left=dict1[int(prediction)]
            #print(label_left)


        for (x,y,w,h) in reyes:  
            reye_roi=new[y:y+h,x:x+w]
            #reye_roi_temp=cv2.resize(reye_roi,(height,width))

            #cv2.imshow("eye",reye_roi_temp)
            reye_roi=cv2.resize(reye_roi,(64,64))
            reye_roi=reye_roi/255
            reye_roi=reye_roi.reshape(-1,64,64,3)
            #print(leye_roi.shape)
            prediction=np.round(model.predict(reye_roi))
            #label_right=int(prediction)
            label_right=dict1[int(prediction)]
            #print(label_right) 

            

        

        if (label_left=="closed" and label_right=="closed"):

            cv2.putText(frame,"Closed",(10,height-20),font,1,(0,0,255),1,cv2.LINE_AA)
            score=score+1
        else:
            cv2.putText(frame,"Open",(10,height-20),font,1,(0,255,0),1,cv2.LINE_AA)
            score=score-1
        
        if score<0:
            score=0
        
        cv2.putText(frame,"Score: "+ str(score),(100,height-20),font,1,(255,255,255),1,cv2.LINE_AA)

        if score>15:
            a=1
            #cv2.imwrite(os.path.join(path,"image.jpg"),frame)
            try:
                sound.play()
            except:
                pass
            
            if thicc<16:
                thicc=thicc+2
            else:
                thicc=thicc-2
            if thicc<2:
                thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thickness=thicc)
        elif a==1:
            a=0
            sound.stop()

    except:
        
        pass

    cv2.imshow("Final",frame)
    if cv2.waitKey(1)==ord("q"):
        break

    #cv2.imshow("frame",frame)
    



cap.release()
cv2.destroyAllWindows()

#x_test=np.load(r"C:\Users\Gurmehar\Desktop\Data Science\ML\ML3\data\x_test.npy")/255

#i=np.random.randint(1,218)
#print(x_test[i].shape)
#x_test_temp=x_test[i].reshape(-1,64,64,3)
#prediction=int(np.round(model.predict(x_test_temp)))
#plt.imshow(x_test[i])
#plt.show()
#print(dict1[prediction])
'''
img=cv2.cvtColor(new,cv2.COLOR_BGR2RGB)
detector = dlib.get_frontal_face_detector()
detections = detector(img, 1)


sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = dlib.full_object_detections()
for det in detections:
    faces.append(sp(img, det))


bb = faces[0].rect

right_eye = [faces[0].part(i) for i in range(36, 42)]
left_eye = [faces[0].part(i) for i in range(42, 48)]
'''