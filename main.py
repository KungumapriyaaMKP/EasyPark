import cv2
import pickle
import cvzone
import numpy as np

# Video feed
video_path = 'carPark.mp4'
location_name = video_path
cap = cv2.VideoCapture(video_path)

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

frame_counter = 0  # to control printing frequency


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

    return spaceCounter


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    if not success:
        break

    frame_counter += 1

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 16)

    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    vacant_count = checkParkingSpace(imgDilate)

    # Print once every 30 frames (~1 sec depending on FPS)
    if frame_counter % 30 == 0:
        print(f"{location_name} , {vacant_count}")

    cv2.imshow("Image", img)
    cv2.waitKey(10)