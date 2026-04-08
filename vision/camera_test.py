import cv2

# =========================
# 打开摄像头（0 = 默认 USB 摄像头）
# macOS 需要 cv2.CAP_AVFOUNDATION
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("✅ Camera connected")

# =========================
# 主循环：实时读取画面
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # =========================
    # 获取画面尺寸
    # =========================
    h, w, _ = frame.shape

    # =========================
    # 手动定义 ROI（打印机 print bed 区域）
    # 你之后可以慢慢调这些数
    # =========================
    x1, y1 = 100, 100
    x2, y2 = w - 100, h - 100

    roi = frame[y1:y2, x1:x2]

    # =========================
    # 显示画面
    # =========================
    cv2.imshow("Full View", frame)
    cv2.imshow("Print Bed ROI", roi)

    # =========================
    # 按 q 退出
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 释放资源
# =========================
cap.release()
cv2.destroyAllWindows()

