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

# --- 1. 环境与硬件优化 (AMD 5700G 专项) ---
cv2.ocl.setUseOpenCL(True)    # 开启 OpenCL 加速（利用核显计算）
cv2.setNumThreads(8)          # 5700G 是 8 核处理器
logging.info(f"OpenCL Acceleration: {'Enabled' if cv2.ocl.haveOpenCL() else 'Disabled'}")

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

# --- 2. 信号处理与资源管理 ---
need_to_end = False

def signal_handler(sig, frame):
    global need_to_end
    logging.info("Interrupt received, shutting down...")
    need_to_end = True

signal.signal(signal.SIGINT, signal_handler)

def release_resources(video, out2f):
    if video is not None: video.release()
    if out2f is not None: out2f.release()
    cv2.destroyAllWindows()
    gc.collect()

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

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 动态背景模型 (使用 float32 累积)
        avg_background = None 
        motion_list = [0, 0]
        is_in_motion = False
        write_cnt = 0
        skip_frame_cnt = 0

        logging.info("Camera connected. AMD Hardware Acceleration Active.")
        u_avg_abs = cv2.UMat()
        # --- 在循环开始前初始化变量 ---
        last_motion_time = 0      # 上次检测到运动的时间戳
        recording_delay = 10      # 运动停止后延迟录制的时间（秒）
        is_recording = False      # 当前是否处于录制状态

        while not need_to_end:
            check, frame = video.read()
            if not check:
                logging.warning("Frame read failed, attempting reconnect...")
                break

            # 环境适应期
            if skip_frame_cnt < 10:
                skip_frame_cnt += 1
                continue

            # --- 硬件加速处理流程 ---
            # 将帧上传至 GPU (UMat)
            u_frame = cv2.UMat(frame)
            u_gray = cv2.cvtColor(u_frame, cv2.COLOR_BGR2GRAY)
            u_gray = cv2.GaussianBlur(u_gray, (21, 21), 0)

            # 初始化/更新背景
            if avg_background is None:
                # 初始背景同步回 CPU 设置为 float
                avg_background = u_gray.get().astype("float")
                del u_frame, u_gray
                continue

            # 动态调整背景：权重 0.05 意味着背景会缓慢吸收环境光线的变化
            cv2.accumulateWeighted(u_gray.get(), avg_background, 0.05)
            u_avg_abs = cv2.UMat(cv2.convertScaleAbs(avg_background))
            
            # 计算当前帧与动态背景的差异
            u_diff = cv2.absdiff(u_gray,u_avg_abs )
            u_thresh = cv2.threshold(u_diff, 30, 255, cv2.THRESH_BINARY)[1]
            u_thresh = cv2.dilate(u_thresh, None, iterations=2)

            # 只有轮廓检测这种无法简单并行的逻辑切回 CPU (get() 方法)
            cnts, _ = cv2.findContours(u_thresh.get(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion = 0
            for contour in cnts:
                if cv2.contourArea(contour) < 8000: # 根据 5700G 性能和分辨率可微调
                    continue
                motion = 1
                break
            if motion == 1:
                last_motion_time = time.time()

            motion_list.append(motion)
            motion_list = motion_list[-2:]

            # 逻辑 1：触发录制
            # 条件：检测到运动 且 当前没在录制
            if motion == 1 and not is_recording:
                is_recording = True
                current_file_name = os.path.join(storage_path, datetime.now().strftime("%Y%m%d%H%M%S") + '.avi')
                out2f = cv2.VideoWriter(current_file_name, fourcc, fps, (frame_width, frame_height))
                write_cnt = 0
                logging.info(f"Motion detected! Start recording: {current_file_name}")
            # 逻辑 2：维持录制
            # 条件：正在录制中
            if is_recording and out2f:
                out2f.write(frame)
                write_cnt += 1
        
                # 逻辑 3：停止录制判断
                if motion == 0 and (time.time() - last_motion_time) > recording_delay:
                    logging.info(f"No motion for {recording_delay}s. Stop recording.")
                    out2f.release()
                    out2f = None
                    is_recording = False
                    write_cnt = 0
            
                elif write_cnt > 1200: # 强制分段由 600 帧(30s)提升到 1200 帧(1分钟)
                    logging.info("Reaching max segment length. Rolling to next file...")
                    out2f.release()
                    # 立即开启下一个衔接文件
                    current_file_name = os.path.join(storage_path, datetime.now().strftime("%Y%m%d%H%M%S") + '_cont.avi')
                    out2f = cv2.VideoWriter(current_file_name, fourcc, fps, (frame_width, frame_height))
                    write_cnt = 0

            # 显式清理 UMat 引用，确保 GPU 内存回收
            del u_frame
            del u_gray
            del u_diff
            del u_avg_abs
            del u_thresh

            # 内存回收
            if skip_frame_cnt > 1000:
                gc.collect()
                skip_frame_cnt=0
                cv2.ocl.finish()


            if cv2.waitKey(1) & 0xFF == ord('q'):
                need_to_end = True
                break


    except Exception:
        logging.error(f"Runtime error:\n{traceback.format_exc()}")
        if current_file_name and os.path.exists(current_file_name):
            if os.path.getsize(current_file_name) < 10240:
                os.remove(current_file_name)
    finally:
        release_resources(video, out2f)
        if not need_to_end:
            time.sleep(5)

logging.info("Program terminated.")
