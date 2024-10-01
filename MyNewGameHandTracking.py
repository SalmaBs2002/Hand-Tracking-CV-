import cv2
import mediapipe as mp
import time
import HandTruckigModule as htm

pTime = 0  # Previous time
cTime = 0  # Current time
cap = cv2.VideoCapture(0)  # Start video capture
detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show the image
    cv2.imshow("Image", img)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
