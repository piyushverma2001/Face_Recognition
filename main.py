import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error loading cascade classifiers.")
    exit()

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

fps = 0
frame_count = 0
time_prev = cv2.getTickCount()

while True:
    ret, img = cap.read()
    
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    frame_count += 1
    time_current = cv2.getTickCount()
    time_diff = (time_current - time_prev) / cv2.getTickFrequency()
    if time_diff >= 1.0:
        fps = frame_count
        frame_count = 0
        time_prev = time_current
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face and Eye Detection', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()