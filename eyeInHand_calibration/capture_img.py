import pyrealsense2 as rs
import numpy as np
import cv2
import os

SAVE_DIR = "opencv_handeye_calib/images"
os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

# D405 彩色流（你已经确认是 848x480）
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# ====== 【关键：读取内参】 ======
color_profile = profile.get_stream(rs.stream.color)
color_video_profile = color_profile.as_video_stream_profile()
intr = color_video_profile.get_intrinsics()

print("\n====== RealSense Color Intrinsics ======")
print(f"Resolution : {intr.width} x {intr.height}")
print(f"fx = {intr.fx}")
print(f"fy = {intr.fy}")
print(f"cx = {intr.ppx}")
print(f"cy = {intr.ppy}")
print(f"distortion model = {intr.model}")
print(f"dist coeffs = {intr.coeffs}")
print("=======================================\n")

print("按 [s] 保存一张图像，按 [q] 退出")

idx = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("D405 Color", color_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"{SAVE_DIR}/img_{idx}.png"
            cv2.imwrite(filename, color_image)
            print(f"[OK] Saved {filename}")
            idx += 1

        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
