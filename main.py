import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

image_path = 'data/image_3.png'

img = cv2.imread(image_path)

reader = easyocr.Reader(['en'], gpu=False)

result = reader.readtext(img)  

threshold = 0.25

for detection in result:
    
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    score = detection[2]
    
    if score > threshold:
        img = cv2.rectangle(img, top_left, bottom_right, (0,0,255), 1)
        img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()