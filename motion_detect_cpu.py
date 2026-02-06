import os
import cv2
import sys
import time
import signal
import logging
import traceback
import configparser
import gc
from datetime import datetime
from pathlib import Path

# --- 配置日志 ---
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

# --- 1. 环境与硬件优化 (CPU 模式) ---
cv2.ocl.setUseOpenCL(False)  # 显式关闭 OpenCL，确保稳定性
cv2.setNumThreads(8)         # 充分利用 5700G 的 8 核
logging.info("Hardware Mode: Pure CPU Optimization (AMD Ryzen 7 5700G)")

# --- 2. 配置加载与权限检查 ---
try:
    config = configparser.ConfigParser()
    with open('private_config.txt', 'r') as f:
        config.read_file(f)
    
    droidcampass = config.get('cam_setting', 'droidcampass')
    camip = config.get('cam_setting', 'camip')
    storage_path = config.get('cam_setting', 'storage_path')
    
    Path(storage_path).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    sys.exit(1)

# --- 3. 信号处理 ---
need_to_end = False
def signal_handler(sig, frame):
    global need_to_end
    logging.info("Interrupt received, shutting down...")
    need_to_end = True

signal.signal(signal.SIGINT, signal_handler)

def safe_release(video, out2f):
    """安全释放资源的辅助函数"""
    try:
        if video is not None: video.release()
        if out2f is not None: out2f.release()
        cv2.destroyAllWindows()
    except:
        pass

# --- 4. 主程序循环 ---
while not need_to_end:
    video = None
    out2f = None
    avg_background = None  # 在每次重新连接摄像头时重置背景
    is_recording = False
    last_motion_time = 0
    recording_delay = 10
    
    try:
        source_url = f'http://{droidcampass}@{camip}:4747/video'
        video = cv2.VideoCapture(source_url)
        
        if not video.isOpened():
            raise ValueError("Could not connect to camera stream.")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        logging.info(f"Connected. Resolution: {frame_width}x{frame_height}")

        skip_frame_cnt = 0
        write_cnt = 0

        while not need_to_end:
            check, frame = video.read()
            if not check:
                logging.warning("Frame read failed, reconnecting...")
                break

            # 核心处理：灰度化与模糊（CPU 模式直接操作 NumPy）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # 1. 初始化/更新动态背景
            if avg_background is None:
                # 预分配 float32 内存块，避免后续重复分配
                avg_background = gray.astype("float32")
                continue

            # 原地运算：直接修改 avg_background 内存地址里的值
            cv2.accumulateWeighted(gray, avg_background, 0.05)
            avg_abs = cv2.convertScaleAbs(avg_background)
            
            # 2. 运动检测
            diff = cv2.absdiff(gray, avg_abs)
            thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion = 0
            for contour in cnts:
                if cv2.contourArea(contour) < 8000:
                    continue
                motion = 1
                break

            if motion == 1:
                last_motion_time = time.time()

            # 3. 录制逻辑
            if motion == 1 and not is_recording:
                is_recording = True
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                current_file_name = os.path.join(storage_path, f"{timestamp}.avi")
                out2f = cv2.VideoWriter(current_file_name, fourcc, fps, (frame_width, frame_height))
                write_cnt = 0
                logging.info(f"Recording started: {current_file_name}")

            if is_recording and out2f:
                out2f.write(frame)
                write_cnt += 1
                
                # 停止录制条件
                time_since_motion = time.time() - last_motion_time
                if motion == 0 and time_since_motion > recording_delay:
                    logging.info("Motion stopped. Closing file.")
                    out2f.release()
                    out2f = None
                    is_recording = False
                
                # 强制分段条件
                elif write_cnt > 1200:
                    logging.info("Segment limit reached. Rolling file.")
                    out2f.release()
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    current_file_name = os.path.join(storage_path, f"{timestamp}_cont.avi")
                    out2f = cv2.VideoWriter(current_file_name, fourcc, fps, (frame_width, frame_height))
                    write_cnt = 0

            # 定期清理内存（由于现在是纯 NumPy，这一步其实非常快）
            skip_frame_cnt += 1
            if skip_frame_cnt > 2000:
                gc.collect()
                skip_frame_cnt = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                need_to_end = True
                break

    except Exception:
        logging.error(f"Runtime error:\n{traceback.format_exc()}")
    finally:
        safe_release(video, out2f)
        if not need_to_end:
            time.sleep(5)  # 失败后等待重连

logging.info("Program terminated cleanly.")
