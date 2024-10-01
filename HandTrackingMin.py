import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialisation des variables de temps
pTime = 0  # Temps précédent
cTime = 0  # Temps actuel

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Vérifier si 'handLms' possède bien des landmarks
            if handLms.landmark:  # Vérifier s'il existe des landmarks
                for id, lm in enumerate(handLms.landmark):  # Notez le bon attribut 'landmark'
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Obtenir les coordonnées des landmarks
                    print(id, cx, cy)
                    if id == 4:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            # Dessiner les landmarks de la main
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calcul du FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Afficher le FPS sur l'image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Affichage de l'image
    cv2.imshow("Image", img)

    # Quitter si 'q' est appuyé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
