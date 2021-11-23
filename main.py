# IMPORT STATEMENTS

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# CLASSIFIERS
face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\PythonProject\DrowsinessDetection\haarcascade_frontalcatface.xml')
eye_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\PythonProject\DrowsinessDetection\haarcascade_eye.xml')
classifier =load_model(r'C:\Users\Admin\Desktop\PythonProject\DrowsinessDetection\model.h5')
labels = ['Drowsy', 'Attentive']
cap = cv2.VideoCapture(0)

# LOOP

while (cap.isOpened()):
    _, frame =cap.read()

    # GETTING GRAY FRAME
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECTING EYES
    eyes = eye_classifier.detectMultiScale(gray)


    for ex,ey,ew,eh in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (255,0,0),3)
        roi_gray_eye = gray[ey:ey+eh,ex:ex+ew]
        roi_gray_eye = cv2.resize(roi_gray_eye,(64,64),interpolation=cv2.INTER_AREA)

        # IMAGE PREPROCESSING
        if np.sum([roi_gray_eye])!=0:
            roi = roi_gray_eye.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # PREDICTION
            prediction = classifier.predict(roi)[0]
            label=labels[prediction.argmax()]
            label_position = (ex,ey)
            if(label=='Drowsy'):
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        else:
            cv2.putText(frame,'No Eyes Detected',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
