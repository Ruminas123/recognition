import cv2
import numpy as np

path = 'resources/ong.jpg'
plate = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

minArea = 1000

while True:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_plate = plate.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in number_plate:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "Number Plate", (x,y-35),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        imgRoi = img[y-30:y+h, x:x+w+30]
        cv2.imshow('ROI', imgRoi)
        cv2.imshow("Original", img)
        cv2.waitKey(500)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        # cv2.imwrite("detectLicensePlate/"+str(count)+".jpg", img)
        break