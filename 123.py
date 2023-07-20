import subprocess
import pyautogui
import time
import keyboard
import os
import dlib
import numpy as np
import cv2
from gaze_tracking import GazeTracking


class Calibration(object):

    # 사용자 및 웹캠에 대한 최적의 이진화 임계값.

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        # 교정 완료되면 true 반환
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        # 주어진 눈의 임계값 반환
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        # 눈의 표면에서 홍채가 차지하는 퍼센티지 반환
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        # 주어진 눈에 대해 프레임을 이진화할 최적의 임계값 계산
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        # 이미지 교정.
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

#눈 개체에 랜드마크 속성 추가


class Eye(object):
    # eye 클래스는 눈을 분리하고 동공 감지를 실행

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        # 두 눈 사이의 중간점 반환
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        #얼굴의 다른 부분이 없는 프레임을 가지기 위해 눈을 독립.
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        #눈 다듬기 과정.
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _analyze(self, original_frame, landmarks, side, calibration):
        # 새 프레임에서 눈을 감지하고 독립, 데이터를 calibration으로 보냄
        # pupil 오브젝트 초기화
        # side = 왼쪽눈 (0) 인지 오른쪽 눈(1) 인지 나타냄.
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)


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


class GazeTracking(object):
    #사용자의 시선을 추적하는 클래스, 깜빡임 여부까지 알 수 있음.

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # face detector = 얼굴 감지에 사용.
        self._face_detector = dlib.get_frontal_face_detector()

        # predictor = 주어진 얼굴에 랜드마크 형성에 사용
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        # 동공 위치 체크
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        # 얼굴 감지, eye 클래스 초기화
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
         #프레임 새로고침, 분석
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        #왼쪽 눈동자 좌표 반환
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        #오른쪽 눈동자 좌표 반환
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        # 수평 비율, 0.0과 1.0 사이의 숫자를 반환합니다.
        # 시선의 극우값 0.0 중심 0.5 극좌는 1.0
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        # 수직 비율, 0.0과 1.0 사이의 숫자를 반환합니다.
        # 시선의 맨위 0.0 중심 0.5 끝 1.0
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        # 사용자가 오른쪽을 보고 있으면 true를 반환
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.6

    def is_left(self):
        # 사용자가 왼쪽을 보고 있으면 true를 반환
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.9

    def is_center(self):
        #사용자가 가운데를 보고 있으면 true를 반환
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def annotated_frame(self):
        # 동공 강조 반환.
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)        #웹캠 실행

filename = 'Ch08_ml.pdf'
pdf = subprocess.Popen(filename, shell=True)

while True:
    # 웹캠으로부터 새 프레임 받음
    _, frame = webcam.read()

    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    start = time.perf_counter()

    left_pupil = gaze.pupil_left_coords()  #왼쪽 눈동자의 좌표(x,y)를 반환.
    right_pupil = gaze.pupil_right_coords()  # 오른쪽 눈동자의 좌표(x,y)를 반환.
    cv2.putText(frame, "Left eyes:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right eyes: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

    a = 3

    if a.name == "esc":
        pyautogui.keyDown('a')
        break
    elif a.event_type == "down":
        b = a
        if a.name == "e" or a.name == "r":
            while not b.event_type == "up" and b.name == a.name:
                b = f
        end = time.perf_counter()
        if a.name == 'e' and end - start > 1.2:
                print("다음 페이지")
                # 우측키 입력
                pyautogui.press('right')
        elif a.name == 'r' and end - start > 1.2:
                print("이전 페이지")
                # 좌측키 입력
                pyautogui.press('left')

webcam.release()
cv2.destroyAllWindows()
