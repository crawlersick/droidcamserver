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

# 配置日志
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

# --- 1. 初始化配置 (移出循环) ---
try:
    config = configparser.ConfigParser()
    with open('private_config.txt', 'r') as f:
        config.read_file(f)
    
    droidcampass = config.get('cam_setting', 'droidcampass')
    camip = config.get('cam_setting', 'camip')
    storage_path = config.get('cam_setting', 'storage_path')
    
    # 权限检查
    test_file = Path(storage_path) / "test_perm.tmp"
    test_file.touch()
    test_file.unlink()
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    sys.exit(1)

# --- 2. 信号处理 (全局设置) ---
need_to_end = False

def signal_handler(sig, frame):
    global need_to_end
    logging.info("Interrupt received, shutting down...")
    need_to_end = True

signal.signal(signal.SIGINT, signal_handler)

def release_resources(video, out2f):
    """安全释放资源的辅助函数"""
    if video is not None:
        video.release()
    if out2f is not None:
        out2f.release()
    cv2.destroyAllWindows()
    gc.collect() # 显式回收内存

# --- 3. 主程序循环 ---
while not need_to_end:
    video = None
    out2f = None
    current_file_name = None
    
    try:
        source_url = f'http://{droidcampass}@{camip}:4747/video'
        video = cv2.VideoCapture(source_url)
        
        if video:
            if not video.isOpened():
                raise ValueError("Could not connect to camera stream.")

        # 动态获取分辨率
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        static_back = None
        motion_list = [0, 0]
        is_in_motion = False
        write_cnt = 0
        skip_frame_cnt = 0

        logging.info("Camera connected. Starting monitoring...")

        while not need_to_end:
            check, frame = video.read()
            if not check:
                logging.warning("Frame read failed, attempting reconnect...")
                break

            # 前10帧用于环境适应，不处理
            if skip_frame_cnt < 10:
                skip_frame_cnt += 1
                continue

            # 运动检测预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if static_back is None:
                static_back = gray
                continue

            # 计算差异
            diff_frame = cv2.absdiff(static_back, gray)
            thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion = 0
            for contour in cnts:
                if cv2.contourArea(contour) < 10000:
                    continue
                motion = 1
                #(x, y, w, h) = cv2.boundingRect(contour)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 更新运动状态
            motion_list.append(motion)
            motion_list = motion_list[-2:]

            # 状态机逻辑
            if motion_list[-1] == 1 and motion_list[-2] == 0:
                is_in_motion = True
                # 开始新文件录制
                current_file_name = os.path.join(storage_path, datetime.now().strftime("%Y%m%d%H%M%S") + '.avi')
                out2f = cv2.VideoWriter(current_file_name, fourcc, fps, (frame_width, frame_height))
                logging.info(f"Motion detected! Recording to: {current_file_name}")

            if is_in_motion and out2f:
                out2f.write(frame)
                write_cnt += 1
                
                # 每录制约30秒(600帧)滚动一次文件，防止文件过大
                if write_cnt > 600:
                    out2f.release()
                    out2f = None
                    write_cnt = 0
                    static_back = None # 重新捕捉背景以适应光线变化
                    is_in_motion = False 

            if motion_list[-1] == 0 and motion_list[-2] == 1:
                logging.info("Motion stopped.")
                if out2f:
                    out2f.release()
                    out2f = None
                is_in_motion = False
                write_cnt = 0

            # 内存优化：每1000帧清理一次垃圾回收
            if skip_frame_cnt % 1000 == 0:
                gc.collect()
                skip_frame_cnt = 0

            # 退出检测
            if cv2.waitKey(1) & 0xFF == ord('q'):
                need_to_end = True
                break

    except Exception:
        logging.error("Runtime error occurred:")
        logging.error(traceback.format_exc())
        
        # 异常处理：清理过小的文件（通常是损坏的录像）
        if current_file_name and os.path.exists(current_file_name):
            if os.path.getsize(current_file_name) < 10240:
                os.remove(current_file_name)
    finally:
        release_resources(video, out2f)
        if not need_to_end:
            logging.info("Retrying in 10 seconds...")
            time.sleep(10)

logging.info("Program terminated cleanly.")
