# Python program to implement
# Webcam Motion Detector

import configparser
# import pandas
import logging
import os
import signal
import sys
# importing datetime class from datetime library
import time
import traceback
from datetime import datetime

# importing OpenCV, time and Pandas library
import cv2

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


out2f = None
video = None
current_file_name = None
need_to_end = False
while True and not need_to_end:
    try:

        # droidcampass
        config = configparser.ConfigParser()
        config.read_file(open(r'private_config.txt'))
        droidcampass = config.get('cam_setting', 'droidcampass')
        camip = config.get('cam_setting', 'camip')




        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            global need_to_end
            need_to_end = True


        signal.signal(signal.SIGINT, signal_handler, )

        # Assigning our static_back to None
        static_back = None

        # List when any moving object appear
        motion_list = [None, None]

        # Time of movement
        # time = []
        video_show = False

        # Capturing video
        video = cv2.VideoCapture('http://' + droidcampass + '@' + camip + ':4747/video')
        if not video.isOpened():
            raise ValueError(" video is not open1!")
        if not out2f.isOpened():
            raise ValueError(" video is not open2!")

        is_init = True
        skip_frame_cnt = 0
        # out2f = None
        # Infinite while loop to treat stack of image as video
        is_in_motion = False
        in_motion_cnt = 0
        write_cnt = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        current_file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.avi'
        out2f = cv2.VideoWriter(current_file_name, fourcc, 20.0, (640, 480))
        while True:
            # Reading frame(image) from video
            # drop frames every time to lower cpu usage
            video.set(cv2.CAP_PROP_POS_FRAMES, 30)
            check, frame = video.read()

            if skip_frame_cnt < 10:
                # print("----")
                # print(skip_frame_cnt)
                skip_frame_cnt = skip_frame_cnt + 1
                continue

            if is_in_motion:

                if write_cnt < 600:
                    pass
                else:
                    static_back = None
                    out2f.release()
                    write_cnt = 0
                    current_file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.avi'
                    out2f = cv2.VideoWriter(current_file_name, fourcc, 20.0, (640, 480))

                out2f.write(frame)
                write_cnt += 1

            if in_motion_cnt > 0:
                in_motion_cnt += 1

            # Initializing motion = 0(no motion)
            motion = 0

            # Converting color image to gray_scale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Converting gray scale image to GaussianBlur
            # so that change can be find easily
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # In first iteration we assign the value
            # of static_back to our first frame

            if static_back is None:
                static_back = gray
                continue

            # Difference between static background
            # and current frame(which is GaussianBlur)
            diff_frame = cv2.absdiff(static_back, gray)

            # If change in between static background and
            # current frame is greater than 30 it will show white color(255)
            thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            # Finding contour of moving object
            cnts, _ = cv2.findContours(thresh_frame.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(cnts)
            for contour in cnts:
                ctaaa = cv2.contourArea(contour)
                # print(ctaaa)
                if ctaaa < 10000:
                    continue
                motion = 1
                # print(ctaaa)

                (x, y, w, h) = cv2.boundingRect(contour)
                # making green rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Appending status of motion
            motion_list.append(motion)

            motion_list = motion_list[-2:]

            # Appending Start time of motion

            if motion_list[-1] == 1 and motion_list[-2] == 0:
                in_motion_cnt = 1
                # time.append(datetime.now())
                is_in_motion = True

            # Appending End time of motion
            if motion_list[-1] == 0 and motion_list[-2] == 1:
                logging.info(f"in_motion_cnt:{in_motion_cnt}")
                in_motion_cnt = 0
                # time.append(datetime.now())
                is_in_motion = False

            if video_show:
                # Displaying image in gray_scale
                cv2.imshow("Gray Frame", gray)

                # Displaying the difference in currentframe to
                # the staticframe(very first_frame)
                cv2.imshow("Difference Frame", diff_frame)

                # Displaying the black and white image in which if
                # intensity difference greater than 30 it will appear white
                cv2.imshow("Threshold Frame", thresh_frame)

                # Displaying color frame with contour of motion of object
                cv2.imshow("Color Frame", frame)

            key = cv2.waitKey(1)
            # if q entered whole process will stop
            if key == ord('q') or need_to_end:
                # if something is movingthen it append the end time of movement
                if motion == 1:
                    # time.append(datetime.now())
                    is_in_motion = False
                break

        # result_list = []
        # # Appending time of motion in DataFrame
        # for i in range(0, len(time), 2):
        #     result_list.append({"Start": time[i], "End": time[i + 1]})
        # df = pandas.DataFrame(data=result_list)

        # Creating a CSV file in which time of movements will be saved
        # df.to_csv("Time_of_movements.csv")
        if out2f:
            out2f.release()
        video.release()

        # Destroying all the windows
        cv2.destroyAllWindows()
        logging.info("==============end=========")
    except Exception as e:
        traceback.print_exc()
        if out2f:
            out2f.release()
        if video:
            video.release()
        if current_file_name:
            if os.path.getsize(current_file_name) < 10240:
                os.remove(current_file_name)

        logging.info("============may not ready ,wait for retry===================")
        time.sleep(10)
