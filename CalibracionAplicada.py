import cv2
import numpy as np

# Cargar parametros stereo
data = np.load("stereo_params.npz")
left_map1 = data['left_map1']
left_map2 = data['left_map2']
right_map1 = data['right_map1']
right_map2 = data['right_map2']

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # Rectificar imágenes
    rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

    cv2.imshow('Rectificada Izquierda', rectL)
    cv2.imshow('Rectificada Derecha', rectR)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
