# Python program to implement
# Webcam Motion Detector

# importing datetime class from datetime library
from datetime import datetime

# importing OpenCV, time and Pandas library
import cv2
import pandas

# Assigning our static_back to None
static_back = None

# List when any moving object appear
motion_list = [None, None]

# Time of movement
time = []



# Capturing video
video = cv2.VideoCapture('http://192.168.8.132:4747/video')

skip_frame_cnt=0
out2f=None
# Infinite while loop to treat stack of image as video
is_in_motion = False
while True:
    # Reading frame(image) from video
    check, frame = video.read()
    if is_in_motion:
        out2f.write(frame)

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

        if skip_frame_cnt < 10:
            print("----")
            print(skip_frame_cnt)
            skip_frame_cnt = skip_frame_cnt + 1
            continue

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
    #print(cnts)
    for contour in cnts:
        ctaaa = cv2.contourArea(contour)
        #print(ctaaa)
        if ctaaa < 10000:
            continue
        motion = 1
        #print(ctaaa)

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Appending status of motion
    motion_list.append(motion)

    motion_list = motion_list[-2:]

    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())
        is_in_motion=True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out2f = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())
        is_in_motion=False

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
    if key == ord('q'):
        # if something is movingthen it append the end time of movement
        if motion == 1:
            time.append(datetime.now())
            is_in_motion=False
        break
result_list = []
# Appending time of motion in DataFrame
for i in range(0, len(time), 2):
    result_list.append({"Start": time[i], "End": time[i + 1]})
df = pandas.DataFrame(data=result_list)

# Creating a CSV file in which time of movements will be saved
df.to_csv("Time_of_movements.csv")
if out2f:
    out2f.release()
video.release()

# Destroying all the windows
cv2.destroyAllWindows()
