import cv2                                                  # OpenCV
import os                                                   # standart Bib
from keras.models import load_model                         # Keras  /lädt das Model
import numpy as np                                          # Bib für Arrays
from pygame import mixer                                    # Zum abspielen vom Sound
import time                                                 # Zeit


mixer.init()
sound = mixer.Sound('alarm.wav')             # Initalisirung vom Alarm Sound

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')          # Initalisirung der OpenCv xml Files
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']                                    #Init Open/Close

model = load_model('models/cnncat2.h5')                 # Lädt das Model
path = os.getcwd()                                      # Gibt das Arbeitsverzeichnis des aktuellen Prozesses.
cap = cv2.VideoCapture(0)                               # Öffnet die Webcam
font = cv2.FONT_HERSHEY_COMPLEX_SMALL                   # Text schreiben mit OpenCV
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]                                                                 # Laden vom Rahmen

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                 # Wandelt das Bild in Grau um

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)                                                         # Gesichtserkennung
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:                                                                        # Rechteck um das Gesicht zeichnen
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255                                                                           # Erkennung vom Rechtenauge
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)                                                       # Weißt dem Auge eine 0/1 zu
        if(rpred[0]==1):
            lbl='Open'
        if(rpred[0]==0):                                                                           # Zuweisung Open oder Closed
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)                                                              # Gleich wie beim r_eye
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:                                                                                         # Berechnung, festellung von Müdigkeit
        score=score-5
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


    if(score<0):
        score=0
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):                                                                                 # Abspielen vom sound.play()
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)                                 # Roter Rahmen anzeigen (Warnung)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
