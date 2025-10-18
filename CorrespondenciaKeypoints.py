import cv2
import mediapipe as mp
import json
import os
import time
import signal
import sys

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

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened():
    print(" No se pudo abrir la cámara 0")
if not cap2.isOpened():
    print(" No se pudo abrir la cámara 1")

all_frames_keypoints = []
frame_idx = 0


output_file = "keypoints_2d_correspondence_noamerica.json"
AUTOSAVE_EVERY = 100 
def _make_serializable(obj):
    """Convierte recursivamente listas/dict/ndarrays a tipos json-serializables."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]

    try:
        return obj.item()
    except Exception:
        return str(obj)

def save_now():
    try:
        data = _make_serializable(all_frames_keypoints)
        tmp = output_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, output_file)  # escritura atómica
        print(f"[{time.strftime('%H:%M:%S')}] Guardado {len(all_frames_keypoints)} frames en {os.path.abspath(output_file)}")
    except Exception as e:
        print("Error guardando archivo:", e)

def handle_sigint(sig, frame):
    print("\nRecibido SIGINT (Ctrl+C). Guardando antes de salir...")
    save_now()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

try:
    while True:
        ret1, frame1 = (cap1.read() if cap1.isOpened() else (False, None))
        ret2, frame2 = (cap2.read() if cap2.isOpened() else (False, None))

        if not ret1 and not ret2:
            print("No hay frames disponibles de ninguna cámara. Terminando loop.")
            break

        keypoints_cam1, keypoints_cam2 = [], []

        if ret1 and frame1 is not None:
            image1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = pose.process(image1_rgb)
            h1, w1 = frame1.shape[:2]
            if results1.pose_landmarks:
                for i, lm in enumerate(results1.pose_landmarks.landmark):
                    x_px = int(lm.x * w1)
                    y_px = int(lm.y * h1)
                    vis = float(lm.visibility) if hasattr(lm, "visibility") else 0.0
                    keypoints_cam1.append({"id": int(i), "x": int(x_px), "y": int(y_px), "visibility": float(vis)})
                    cv2.circle(frame1, (x_px, y_px), 4, (0, 255, 0), -1)
            cv2.imshow("Camara 1", frame1)

        if ret2 and frame2 is not None:
            image2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results2 = pose.process(image2_rgb)
            h2, w2 = frame2.shape[:2]
            if results2.pose_landmarks:
                for i, lm in enumerate(results2.pose_landmarks.landmark):
                    x_px = int(lm.x * w2)
                    y_px = int(lm.y * h2)
                    vis = float(lm.visibility) if hasattr(lm, "visibility") else 0.0
                    keypoints_cam2.append({"id": int(i), "x": int(x_px), "y": int(y_px), "visibility": float(vis)})
                    cv2.circle(frame2, (x_px, y_px), 4, (0, 0, 255), -1)
            cv2.imshow("Camara 2", frame2)

        all_frames_keypoints.append({
            "frame": int(frame_idx),
            "cam1": keypoints_cam1,
            "cam2": keypoints_cam2
        })

        print(f"\nFrame {frame_idx} -> cam1:{len(keypoints_cam1)} pts, cam2:{len(keypoints_cam2)} pts")

        frame_idx += 1

        if frame_idx % AUTOSAVE_EVERY == 0:
            save_now()

        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pulsado. Saliendo.")
            break

finally:
    print("Guardando antes de cerrar (finally)...")
    save_now()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Recursos liberados. Archivo final en:", os.path.abspath(output_file))