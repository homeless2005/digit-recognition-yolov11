import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==================== 配置参数 ====================
MODEL_PATH = "best.pt"
VIDEO_DIR = "video_dir"

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
CONF_THRESHOLD = 0.4
WINDOW_NAME = "YOLO Detection - Press 'q' to quit"

# ------------------ 自定义绘制参数 ------------------
FONT_SCALE = 1               # 字体大小（可调）
TEXT_THICKNESS = 2             # 文字粗细
BOX_THICKNESS = 2              # 预测框粗细
TEXT_COLOR = (0, 0, 0)         # 黑色
TEXT_OFFSET = 5                # 文字与预测框之间的间距（像素）

# ==================== 辅助函数 ====================
def get_color_for_class(class_id, num_classes=80):
    """为每个类别生成固定颜色（基于HSV色环）"""
    hue = (class_id * 137) % 180
    sat = 200
    val = 200
    color = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)

# ==================== 加载模型 ====================
model = YOLO(MODEL_PATH)
num_classes = len(model.names) if hasattr(model, 'names') else 80

# ==================== 获取所有视频文件 ====================
video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(Path(VIDEO_DIR).glob(f"*{ext}"))
    video_files.extend(Path(VIDEO_DIR).glob(f"*{ext.upper()}"))

print(f"找到 {len(video_files)} 个视频文件")

# 预生成每个类别的固定颜色
class_colors = {i: get_color_for_class(i, num_classes) for i in range(num_classes)}

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ==================== 逐视频处理 ====================
for idx, video_path in enumerate(video_files):
    print(f"\n[{idx + 1}/{len(video_files)}] 正在播放: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        annotated_frame = frame.copy()

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
                # 获取文字尺寸
                (text_w, text_h), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX,
                                                              FONT_SCALE, TEXT_THICKNESS)

                # 优先放在框的上方外部
                text_x = x1
                text_y = y1 - TEXT_OFFSET   # 框上方

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

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("用户终止程序")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('n') or key == 27:
            print("跳过当前视频，进入下一个")
            break

    cap.release()

print("\n所有视频播放完毕！")
cv2.destroyAllWindows()
