import numpy as np
import cv2


class Pupil(object):
    #눈의 홍채를 감지하고 눈동자의 위치를 추정하는 클래스

    def __init__(self, eye_frame, threshold):   #__init__()의 첫 번째 인수는 반드시 self로 지정
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod     # 데코레이터, 정적메소드
    def image_processing(eye_frame, threshold):
        #프레임에서 홍채 분리 작업을 수행.
        kernel = np.ones((3, 3), np.uint8)  # 3x3 크기의 매트릭스 생성.
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)  # Bilateral Filter = 영상에서 엣지와 노이즈를 줄여주는 필터 
        new_frame = cv2.erode(new_frame, kernel, iterations=3) # erode = 받은 프레임 이미지를 얇게 수축
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]   
        # threshold = 어떤 임계점을 기준으로 두 가지 부류로 나누는 방법을 의미
        # 스레시 홀드를 이용해 바이너리 이미지로 만듬.

        return new_frame

    def detect_iris(self, eye_frame):
        # 홍채를 감지하고 중심점을 계산하여 홍채의 위치를 추정.
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]  # contours = 이미지 윤곽선 추출.
        contours = sorted(contours, key=cv2.contourArea) # contourArea : contour가 그린 면적

        try:
            moments = cv2.moments(contours[-2])  #moment = 윤곽선, 중심등을 계산할 수 있는 알고리즘
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])  # 동공의 무게중심 추출 
        except (IndexError, ZeroDivisionError):
            pass
