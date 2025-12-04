import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: Could not open video stream.") 
  exit()
while True:
  ret, frame = cap.read()
  if not ret:
    print("Error: Could not read frame.")
    break
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lower_skin = np.array([0, 20, 70], dtype=np.uint8)
  upper_skin = np.array([20, 255, 255], dtype=np.uint8)

  mask = cv2.inRange(hsv, lower_skin, upper_skin)
  result = cv2.bitwise_and(frame, frame, mask=mask)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  max_contour = max(contours, key=cv2.contourArea)
  if cv2.contourArea(max_contour) > 500:
    x, y, w, h = cv2.boundingRect(max_contour)
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), 2)

  cv2.imshow('Original Frame', frame)
  cv2.imshow('Filtered Frame', result)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    exit()

def color_filter(frame, filter):
  ret, frame = cap.read()
  red_tint = np.array([0, 0, 255], dtype=np.uint8)
  blue_tint = np.array([255, 0, 0], dtype=np.uint8)
  green_tint = np.array([0, 255, 0], dtype=np.uint8)
  print('press one of the following keys to filter the color:')
  print('r for red')
  print('b for blue')
  print('g for green')
  print('q to quit')
  while True:
    key = cv2.waitKey(1)
    if key == ord('r'):
      filter = (red_tint)
    if key == ord('b'):
      filter = (blue_tint)
    if key == ord('g'):
      filter = (green_tint)
    filtered_image = cv2.add(frame, filter)
    cv2.imshow('Filtered Frame', filtered_image)
    if key == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
