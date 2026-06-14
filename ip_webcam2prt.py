from ultralytics import YOLO
import cv2
import numpy as np

# ===================== 重要！把这里改成你手机上显示的地址 =====================
PHONE_CAMERA_URL = "http://xxx.xxx.xxx.xxx:8080/video"  # 手机下载ip摄像头，填入IP地址和端口号
# ==========================================================================

# ===================== 自定义绘制参数（与原代码一致） =====================
FONT_SCALE = 0.5               # 字体大小
TEXT_THICKNESS = 1             # 文字粗细
BOX_THICKNESS = 2              # 预测框粗细
TEXT_COLOR = (0, 0, 0)         # 黑色
TEXT_OFFSET = 5                # 文字与预测框之间的间距（像素）
CONF_THRESHOLD = 0.3           # 置信度阈值（可自行调整）
# ==========================================================================

def get_color_for_class(class_id, num_classes=80):
    """为每个类别生成固定颜色（基于HSV色环）"""
    hue = (class_id * 137) % 180
    sat = 200
    val = 200
    color = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)

# 加载YOLO11模型（n=最快，s=平衡，m=更准）
model = YOLO("best.pt")
num_classes = len(model.names) if hasattr(model, 'names') else 80
# 预生成每个类别的固定颜色
class_colors = {i: get_color_for_class(i, num_classes) for i in range(num_classes)}

# 打开手机摄像头流
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():
    print("❌ 无法连接手机摄像头，请检查：")
    print("1. 手机和电脑在同一个WiFi")
    print("2. IP 地址正确")
    print("3. 手机APP已开启视频流")
    exit()

print("✅ 成功连接手机摄像头，开始实时检测...")

# 创建可调整大小的窗口
WINDOW_NAME = "手机摄像头 - YOLO实时检测"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# 可选：设置显示缩放比例（如果希望强制缩放画面，可改为0.5等，1.0为不缩放）
DISPLAY_SCALE = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 视频读取失败")
        break

    # 可选：缩放画面（减轻显示压力，提高性能）
    if DISPLAY_SCALE != 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * DISPLAY_SCALE)
        new_height = int(height * DISPLAY_SCALE)
        frame = cv2.resize(frame, (new_width, new_height))

    # YOLO 预测
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        show=False,
        device=0          # 使用GPU（没有GPU就写 device="cpu"）
    )

    annotated_frame = frame.copy()
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]   # 只取类别名称

            # 获取固定颜色
            color = class_colors.get(cls_id, (0, 255, 0))

            # 绘制预测框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # ---------- 绘制文字（仅类别名称）----------
            (text_w, text_h), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX,
                                                          FONT_SCALE, TEXT_THICKNESS)

            # 优先放在框的上方外部
            text_x = x1
            text_y = y1 - TEXT_OFFSET

            # 上方空间不足时，放到框的下方外部
            if text_y - text_h < 0:
                text_y = y2 + text_h + TEXT_OFFSET

            # 边界调整（防止超出左右和底部）
            if text_x + text_w > annotated_frame.shape[1]:
                text_x = annotated_frame.shape[1] - text_w - TEXT_OFFSET
            if text_x < 0:
                text_x = TEXT_OFFSET
            if text_y > annotated_frame.shape[0]:
                text_y = annotated_frame.shape[0] - TEXT_OFFSET
            if text_y - text_h < 0:
                text_y = text_h + TEXT_OFFSET

            # 绘制文字（背景透明）
            cv2.putText(annotated_frame, class_name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    cv2.imshow(WINDOW_NAME, annotated_frame)

    # 按 Q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
