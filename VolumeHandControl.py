import time
import HandTruckigModule as htm
import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0

# Initialize the hand detector
detector = htm.handDetector(detectionCon=0.7)

# Set up audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get and print the volume range
volume_range = volume.GetVolumeRange()
print(f"Volume Range: {volume_range}")

# Define minVol and maxVol based on the volume range
minVol = volume_range[0]  # Minimum volume level
maxVol = volume_range[1]  # Maximum volume level
vol = 0
volBar = 400
volPer = 0



if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    success, img = cap.read()  # Read the frame from the camera

    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)  # Detect hands
    lmList = detector.findPosition(img, draw=False)  # Get hand landmark positions

    if len(lmList) != 0:
        # Get the coordinates of the thumb and index finger tips
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point

        # Draw circles and lines on the image
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate the length between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)
        #print(f"Length: {length}")

        # Hand range 50 - 300
        # Volume Range -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])


        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)
        # Check if the fingers are close enough
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (255, 0, 0),3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'FPS: {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)



    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)


    # Display the captured frame in a window
    cv2.imshow("Img", img)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
