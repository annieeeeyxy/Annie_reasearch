import cv2
import numpy as np

# =========================
# Camera (USB = index 0)
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()
print("✅ Camera connected (index 0)")

# =========================
# Printer parameters (rough)
# =========================
BED_X_MM = 220
BED_Y_MM = 220
Z_SAFE = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # --- simple detection pipeline ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2

            # draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            # pixel -> mm (very rough, will calibrate later)
            H, W, _ = frame.shape
            target_x = cx / W * BED_X_MM
            target_y = cy / H * BED_Y_MM

            gcode = f"G1 X{target_x:.1f} Y{target_y:.1f} Z{Z_SAFE} F3000"

            print("➡️ Target detected")
            print(f"   Pixel: ({cx}, {cy})")
            print(f"   G-code: {gcode}")

            # slow down spam
            cv2.waitKey(300)

    cv2.imshow("Detect", frame)
    #cv2.imshow("Thresh", thresh)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
