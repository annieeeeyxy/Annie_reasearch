import cv2
import sys

idx = int(sys.argv[1])
print(f"🎥 Testing camera index {idx}")


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit()

print("✅ Camera opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow(f"Camera {idx}", frame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
