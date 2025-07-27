import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
detector = FaceDetector()

print(" Press 's' to take the first photo (face1)")
face1 = None
face2 = None

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        face1 = frame.copy()
        print(" Face 1 captured. Now press 'd' for second photo.")
        break

# Wait for second photo
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord('d'):
        face2 = frame.copy()
        print(" Face 2 captured. Processing...")
        break

cap.release()
cv2.destroyAllWindows()

# Resize both to same size
face1 = cv2.resize(face1, (640, 480))
face2 = cv2.resize(face2, (640, 480))

# Detect faces
face1, bboxs1 = detector.findFaces(face1)
face2, bboxs2 = detector.findFaces(face2)

if bboxs1 and bboxs2:
    x1, y1, w1, h1 = bboxs1[0]['bbox']
    x2, y2, w2, h2 = bboxs2[0]['bbox']

    crop1 = face1[y1:y1 + h1, x1:x1 + w1]
    crop2 = face2[y2:y2 + h2, x2:x2 + w2]

    # Resize crops to fit
    crop1_resized = cv2.resize(crop1, (w2, h2))
    crop2_resized = cv2.resize(crop2, (w1, h1))

    # Paste each face onto the other image
    face1[y1:y1 + h1, x1:x1 + w1] = crop2_resized
    face2[y2:y2 + h2, x2:x2 + w2] = crop1_resized

    # Show results
    cv2.imshow("Swapped Face 1", face1)
    cv2.imshow("Swapped Face 2", face2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Face not detected in one or both photos.")
