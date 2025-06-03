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
# 표준 argparse 관련 코드
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


# 드론의 속도
S = 20
S2 = 5
UDOffset = 150

# openCV가 생성하는 얼굴 인식 상자의 크기
faceSizes = [1026, 684, 456, 304, 202, 136, 90]

# 가속 시키는 값입니다. 다만 이 값들은 최종화되지 않았기에 주의 바람.
acc = [500,250,250,150,110,70,50]

# pygame 창의 fps 지정
FPS = 25
dimensions = (960, 720)

# 얼굴 인식 데이터
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_tellotv.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_classifier = load_model('emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
c=[]

# 세션을 저장한다면, 디렉토리가 있는지 확인.
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)


#object 클래스를 상속받은 FrontEnd 클래스
class FrontEnd(object):
    
    def __init__(self):

        
       
        # 드론과 상호작용하는 Tello 객체
        self.tello = Tello()

        # 드론의 속도 (-100~100)
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

        # 시간에 따른 감정값을 가져옵니다(y축)
        emotions = df['Happy']
        e = list(emotions)
        e = int(sum(e)/len(e))
        # df의 데이터 타입을 출력합니다.
        print(df.dtypes)

        # plot2D 함수를 호출하여 그래프를 보입니다.
        plt.figure(figsize=(10, 10))

        # 그래프의 색상, 데이터, 점의 크기 등을 정하고 그래프를 그립니다.
        plt.plot(emotions)

        #그래프에 제목을 붙여줍니다.
        plt.title('Happiness')
        
        # x축에 label을 달아줍니다.
        plt.xlabel('aver\n'+str(e))

        # y축에 label을 달아줍니다.
        plt.ylabel('happiness')

        # 그래프를 보입니다.
        plt.show()
    def run(self):

        
        drone = tellopy.Tello()
        drone.takeoff()
 


        #드론이 연결이 되지 않으면 함수 종료
        if not self.tello.connect():
            print("Tello not connected")
            return


        # 프로그램을 비정상적인 방법으로 종료를 시도하여 비디오 화면이 꺼지지 않은 경우 종료.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # 비디오가 켜지지않는 경우 종료.
        if not self.tello.streamon():
            print("Could not start video stream")
            return

        #프레임 단위로 인식
        frame_read = self.tello.get_frame_read()

        should_stop = False
        imgCount = 0
        OVERRIDE = False
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()
        
        # X축 안전 범위
        szX = args.saftey_x
        # Y축 안전 범위
        szY = args.saftey_y
        
        #디버깅 모드
        if args.debug:
            print("DEBUG MODE ENABLED!")


        #비행을 멈취야할 상황이 주어지지 않은 경우
        while not should_stop:



            self.update()
            #프레임 입력이 멈췄을 경우 while문 탈출
            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame

            vid = self.tello.get_video_capture()
            
            # 키보드 입력을 기다림
            k = cv2.waitKey(20)

            # 프로그램 종료
            if k == 27:
                should_stop = True
                break

            gray  = cv2.cvtColor(frameRet, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

            # 비어있는 이미지를 생성
            canvas = np.zeros((250, 300, 3), dtype="uint8")
            #canvas2 = np.zeros((250, 300, 3), dtype="uint8")

            # 대상 크기
            tSize = faceSizes[tDistance]

            # 중심 차원들
            cWidth = int(dimensions[0]/2)
            cHeight = int(dimensions[1]/2)

            noFaces = len(faces) == 0

            # 얼굴이 찾아진 경우에만 감정 인식을 실행
            if len(faces) > 0:
                # 가장 큰 이미지에 대해서 실행
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = face
                # 이미지를 48x48 사이즈로 재조정 (neural network 위함)
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # 감정을 예측 
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
        
                # label 을 할당, 2020.08.11 frame -> frameRet
                cv2.putText(frameRet, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameRet, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

                # label 을 출력
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    if emotion =="Happy":
                        c.append(prob *100)
                #[angry 값, 시간]
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.imshow('Emotion Recognition', frameRet)
            cv2.imshow("Probabilities", canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                np.savetxt("happiness.csv",c,delimiter= '',header = "Happy",comments='')
                self.graphic()
                break

        # 종료시에 배터리를 출력
        self.tello.get_battery()


        # 종료 전에 항상 호출. 자원들을 해제함.
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
    #frontend 객체 생성
    frontend = FrontEnd()
    
    # run 함수를 실행
    frontend.run()
    frontend.graphic()

if __name__ == '__main__':
    main()