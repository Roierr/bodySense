import cv2
import numpy as np
import glob
import os


CHESSBOARD = (13, 9)
square_size = 18 


folder_left = "calibracionParaUnaCamara1"
folder_right = "calibracionParaUnaCamara2"

objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD[0],0:CHESSBOARD[1]].T.reshape(-1,2)
objp *= square_size


objpoints = []        
imgpoints_left = []   
imgpoints_right = []   


images_left = sorted(glob.glob(os.path.join(folder_left, "*.jpg")))
images_right = sorted(glob.glob(os.path.join(folder_right, "*.jpg")))

if len(images_left) != len(images_right):
    print("Advertencia: Las carpetas no tienen la misma cantidad de imágenes.")

for imgL_path, imgR_path in zip(images_left, images_right):
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, flags)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, flags)

    if foundL and foundR:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        cv2.drawChessboardCorners(imgL, CHESSBOARD, cornersL, foundL)
        cv2.drawChessboardCorners(imgR, CHESSBOARD, cornersR, foundR)
        cv2.imshow("Izquierda", imgL)
        cv2.imshow("Derecha", imgR)
        cv2.waitKey(100)

cv2.destroyAllWindows()
print(f"Se detectaron esquinas en {len(objpoints)} pares de imágenes.")

retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-6)

retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1],
    criteria=criteria,
    flags=flags
)

print("\nMatriz de rotacion entre cámaras (R):\n", R)
print("\nVector de traslacion (T):\n", T)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

np.savez("stereo_params.npz",
         mtxL=mtxL, distL=distL,
         mtxR=mtxR, distR=distR,
         R=R, T=T, R1=R1, R2=R2,
         P1=P1, P2=P2, Q=Q,
         left_map1=left_map1, left_map2=left_map2,
         right_map1=right_map1, right_map2=right_map2)

print("Parámetros estéreos guardados en 'stereo_params.npz'")
