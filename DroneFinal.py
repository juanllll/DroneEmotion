from djitellopy import Tello
import tellopy
import cv2
import numpy as np
import time
import datetime
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=3,
    help='use -d to change the distance of the drone. Range 0-6')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
    help='use -sx to change the saftey bound on the x axis . Range 0-480')
parser.add_argument('-sy', '--saftey_y', type=int, default=55,
    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-os', '--override_speed', type=int, default=1,
    help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

S = 20
S2 = 5
UDOffset = 150

faceSizes = [1026, 684, 456, 304, 202, 136, 90]

acc = [500,250,250,150,110,70,50]
FPS = 25
dimensions = (960, 720)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_tellotv.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_classifier = load_model('emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
c=[]

if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)


class FrontEnd(object):
    
    def __init__(self):
        self.tello = Tello()

        #수직, 수평 속도
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False
        # 실행 함수
    def graphic(self):
        df = pd.read_csv("happiness.csv") 

        emotions = df['Happy']
        e = list(emotions)
        e = int(sum(e)/len(e))
        print(df.dtypes)

        plt.figure(figsize=(10, 10))
        plt.plot(emotions)
        plt.title('Happiness')
        plt.xlabel('aver\n'+str(e))
        plt.ylabel('happiness')
        plt.show()
    def run(self):

        
        drone = tellopy.Tello()
        drone.takeoff()
 
        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return
          
        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        imgCount = 0
        OVERRIDE = False
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()
        
        szX = args.saftey_x
        szY = args.saftey_y
        
        if args.debug:
            print("DEBUG MODE ENABLED!")

        while not should_stop:

            self.update()
            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame

            vid = self.tello.get_video_capture()
            
            k = cv2.waitKey(20)
            if k == 27:
                should_stop = True
                break

            gray  = cv2.cvtColor(frameRet, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            tSize = faceSizes[tDistance]

            cWidth = int(dimensions[0]/2)
            cHeight = int(dimensions[1]/2)

            noFaces = len(faces) == 0

            if len(faces) > 0:
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = face
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
        
                cv2.putText(frameRet, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameRet, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    if emotion =="Happy":
                        c.append(prob *100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.imshow('Emotion Recognition', frameRet)
            cv2.imshow("Probabilities", canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                np.savetxt("happiness.csv",c,delimiter= '',header = "Happy",comments='')
                self.graphic()
                break
              
        self.tello.get_battery()
        self.tello.end()


    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)

def lerp(a,b,c):
    return a + c*(b-a)

def main():
    frontend = FrontEnd()
    frontend.run()
    frontend.graphic()

if __name__ == '__main__':
    main()
