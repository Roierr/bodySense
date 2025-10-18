import cv2
import mediapipe as mp
import json
import os

# --- Configuración de MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Inicialización de cámaras ---
cap1 = cv2.VideoCapture(0)  # Cámara 1
cap2 = cv2.VideoCapture(1)  # Cámara 2

# --- Lista para guardar todos los keypoints ---
all_frames_keypoints = []

frame_idx = 0  # contador de frames

# --- Loop principal ---
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Convertir a RGB
    image1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe Pose
    results1 = pose.process(image1_rgb)
    results2 = pose.process(image2_rgb)

    keypoints_cam1 = []
    keypoints_cam2 = []

    # Extraer keypoints 2D de cámara 1
    if results1.pose_landmarks:
        h1, w1, _ = frame1.shape
        for id, lm in enumerate(results1.pose_landmarks.landmark):
            x_px = int(lm.x * w1)
            y_px = int(lm.y * h1)
            visibility = lm.visibility
            keypoints_cam1.append({"x": x_px, "y": y_px, "visibility": visibility})
            cv2.circle(frame1, (x_px, y_px), 5, (0, 255, 0), -1)

    # Extraer keypoints 2D de cámara 2
    if results2.pose_landmarks:
        h2, w2, _ = frame2.shape
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            x_px = int(lm.x * w2)
            y_px = int(lm.y * h2)
            visibility = lm.visibility
            keypoints_cam2.append({"x": x_px, "y": y_px, "visibility": visibility})
            cv2.circle(frame2, (x_px, y_px), 5, (0, 0, 255), -1)

    # Guardar keypoints del frame actual con índice
    all_frames_keypoints.append({
        "frame": frame_idx,
        "cam1": keypoints_cam1,
        "cam2": keypoints_cam2
    })

    # --- Imprimir keypoints en la terminal ---
    print(f"\nFrame {frame_idx}:")
    print("Camara 1 Keypoints:")
    for i, kp in enumerate(keypoints_cam1):
        print(f"  ID {i}: x={kp['x']}, y={kp['y']}, visibility={kp['visibility']:.2f}")
    print("Camara 2 Keypoints:")
    for i, kp in enumerate(keypoints_cam2):
        print(f"  ID {i}: x={kp['x']}, y={kp['y']}, visibility={kp['visibility']:.2f}")

    frame_idx += 1

    # Mostrar resultados
    cv2.imshow("Camara 1", frame1)
    cv2.imshow("Camara 2", frame2)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

# --- Guardar todos los keypoints en un archivo JSON ---
output_file = "keypoints_2d_with_visibility_noamerica.json"
with open(output_file, "w") as f:
    json.dump(all_frames_keypoints, f, indent=4)

print(f"\nTodos los keypoints 2D guardados en {os.path.abspath(output_file)}")

# --- Liberar recursos ---
cap1.release()
cap2.release()
cv2.destroyAllWindows()
pose.close()