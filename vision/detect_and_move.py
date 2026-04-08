import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Missing dependency: install with `pip install ultralytics opencv-python` and try again")
    exit()

MODEL_PATH = Path.home() / "Documents" / "Documents - Annie’s MacBook Air" / "setup" / "yolo11n.pt"
CONF_THRESHOLD = 0.35
BED_X_MM = 220
BED_Y_MM = 220
Z_SAFE = 30

if not MODEL_PATH.exists():
    print(f"❌ YOLO model not found: {MODEL_PATH}")
    exit()

model = YOLO(str(MODEL_PATH))
print(f"✅ Loaded YOLO model: {MODEL_PATH}")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()
print("✅ Camera connected (index 0)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    results = model(frame, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        xyxys = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cls in zip(xyxys, confs, clss):
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = results.names.get(cls, str(cls))

            H, W, _ = frame.shape
            
            # Relative position (normalized 0-1, where 0.5,0.5 is center)
            rel_x = cx / W
            rel_y = cy / H
            
            # Offset from center in pixels
            offset_x = cx - W // 2
            offset_y = cy - H // 2
            
            # Bed coordinates
            target_x = rel_x * BED_X_MM
            target_y = rel_y * BED_Y_MM
            gcode = f"G1 X{target_x:.1f} Y{target_y:.1f} Z{Z_SAFE} F3000"

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw center crosshair
            cv2.line(frame, (W // 2 - 20, H // 2), (W // 2 + 20, H // 2), (100, 100, 255), 1)
            cv2.line(frame, (W // 2, H // 2 - 20), (W // 2, H // 2 + 20), (100, 100, 255), 1)
            
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f} | Pos: ({rel_x:.2f}, {rel_y:.2f})",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Offset: ({offset_x:+d}, {offset_y:+d}) px",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            print("➡️ Phone detected")
            print(f"   Class: {class_name}")
            print(f"   Confidence: {conf:.2f}")
            print(f"   Pixel (absolute): ({cx}, {cy})")
            print(f"   Relative position (0-1): ({rel_x:.3f}, {rel_y:.3f})")
            print(f"   Offset from center (px): ({offset_x:+d}, {offset_y:+d})")
            print(f"   Bed position (mm): X={target_x:.1f}, Y={target_y:.1f}")
            print(f"   G-code: {gcode}")

    cv2.imshow("Detect", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
