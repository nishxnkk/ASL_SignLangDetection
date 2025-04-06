import time

import cv2
from PIL.ImageChops import offset
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from pyparsing import countedArray

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset=20
imageSize = 300

folder = "data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hands = hands[0]
        x,y,w,h=hands['bbox']
        imgWhite = np.ones((imageSize,imageSize,3), np.uint8)*255


        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imageSize/h
            wCal=math.ceil(w*k)
            imgResize = cv2.resize(imgCrop,(wCal,imageSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imageSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imageSize / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap: hCal+hGap, :] = imgResize


        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image{time.time()}.jpeg', imgWhite)
        print(counter)
