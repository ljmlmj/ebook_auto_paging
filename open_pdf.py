import subprocess
import pyautogui as pyautogui
import time
import keyboard


filename = 'Ch08_ml.pdf'
pdf = subprocess.Popen(filename, shell=True)

while True:
    start = time.perf_counter()
    a = keyboard.read_event()

    if a.name == "esc":
        break
    elif a.event_type == "down":
        b = keyboard.read_event()
        if a.name == "e" or a.name == "r":
            while not b.event_type == "up" and b.name == a.name:
                b = keyboard.read_event()
            end = time.perf_counter()

            if a.name == 'e' and end-start > 1.2:
                print("다음 페이지")
                # 우측키 입력
                pyautogui.press('right')
            elif a.name == 'r' and end-start > 1.2:
                print("이전 페이지")
                # 좌측키 입력
                pyautogui.press('left')
