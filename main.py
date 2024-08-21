import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

harcascade_path = 'model/haarcascade_russian_plate_number.xml'

cap = cv2.VideoCapture(0)

## PARAMETERS
minArea = 500
width, height = 640, 480
count = 0

cap.set(3, width) 
cap.set(4, height)

while True:
    _, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plate_cascade = cv2.CascadeClassifier(harcascade_path)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:

        area = w * h

        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, 'Number Plate', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2) 

            plate_image = img[y:y+h, x:x+w]
            cv2.imshow('Detected Plate', plate_image)

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        try:
            cv2.imwrite('plates/plate_'+str(count)+'.jpg', plate_image)
            print(f'Image saved as plates/plate_{count}.jpg')
            count += 1

        except Exception as e:
            print('Error saving image: {}'.format(e))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plate_image = cv2.imread('plates/plate_0.jpg')
plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
cl1 = clahe.apply(plate_image_gray)

reader = easyocr.Reader(['en'], gpu=False)

result = reader.readtext(cl1)  

threshold = 0.25

for detection in result:
    
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    score = detection[2]
    
    if score > threshold:
        plate_image = cv2.rectangle(plate_image, top_left, bottom_right, (0,0,255), 4)
        plate_image = cv2.putText(plate_image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()