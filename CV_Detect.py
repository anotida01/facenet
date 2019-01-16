import cv2
import CompVision
import numpy as np
import time
import os
import random
from Classifier.TRAIN import classify_many_photos

# important variables

cascade_scale_factor = 1.25
cascade_min_neighbour = 5

PATH = os.getcwd()

capture_thres = 40
frame_count = int()
head_frame_count = int()
num_faces = int()
num_windows = int()
current_windows = list()

# Load Cascades
face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')


def read_for_faces(curr_frame):
    """
    :param curr_frame: The frame on which the face is going to be scanned for
    :return: Co-ordinates as well as other dimensions of the face
    """
    face = face_cascade.detectMultiScale(curr_frame, cascade_scale_factor, cascade_min_neighbour)
    return face


def crop_head_frame(curr_frame, rectangle_tuple):
    """
    Compared to the one in CompVision, this one can iterate through many heads
    :param curr_frame: Frame where the head has been detected
    :param rectangle_tuple: Tuple containing Rectangle Details - (x_cord, y_cord, width, height)
    :return: Returns ROI Image Frame of the head.
    """

    num_heads_list = np.arange(num_faces)

    if num_faces > num_windows:
        for window in num_heads_list:
            if str(window) not in current_windows:
                create_new_cv2_window(str(window), (300, 300)) # TODO 1080p version

    if num_faces >= 1:
        for n in range(num_faces):

            for (x, y, w, h) in rectangle_tuple[[n]]:
                cropped = curr_frame[y:y + h, x:x + w]
                cv2.resize(cropped, (128, 128))
                cv2.imshow(str(num_heads_list[n]), cropped)


def create_new_cv2_window(window_name, dimensions):
    global num_windows  # Global vars can only be read. To edit a global var inside of a func, use global keyword
    num_windows = num_windows + 1

    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, dimensions[0], dimensions[1])

    if num_windows <= 5:
        window_x_coord = 1500 + (300 * (num_windows - 1))
        window_y_coord = 0
    elif num_windows <= 10:
        window_x_coord = 1500 + (300 * (num_windows - 6))
        window_y_coord = 450
    elif num_windows <= 15:
        window_x_coord = 1500 + (300 * (num_windows - 11))
        window_y_coord = 850
    elif num_windows <= 20:
        window_x_coord = 1500 + (300 * (num_windows - 16))
        window_y_coord = 1500
    else:
        window_x_coord = 1500 + (300 * num_windows)
        window_y_coord = 400

    cv2.moveWindow(window_name, window_x_coord, window_y_coord)
    if window_name in current_windows:
        pass
    else:
        current_windows.append(window_name)


def destroy_empty_windows():
    global current_windows
    global num_windows

    if len(current_windows) > 0:
        last_most_window = current_windows[-1]

        if num_faces < num_windows:
            cv2.destroyWindow(last_most_window)
            if len(current_windows) == 1:
                current_windows = []
            else:
                current_windows = current_windows[:-1]
            num_windows = num_windows - 1
            print('WINDOW {} DELETED'.format(last_most_window))


def run(scale_factor=float(), min_neighbour=int()):
    web_cam = cv2.VideoCapture(0)

    global cascade_min_neighbour
    global cascade_scale_factor
    global frame_count
    global head_frame_count
    global num_faces

    cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('feed', 1400, 1050) # TODO 4k
    cv2.resizeWindow('feed', 700, 505)  # TODO 1080p
    cv2.moveWindow('feed', 100, 50)

    cascade_scale_factor = scale_factor
    cascade_min_neighbour = min_neighbour

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(get_video_file_name(), fourcc, 20, (640, 480))

    while True:
        frame_count = frame_count + 1
        face_detected = False
        status, frame = web_cam.read()
        _, clean_frame = web_cam.read()
        output.write(clean_frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_tuple = read_for_faces(gray_frame)

        num_faces = len(face_tuple)

        if num_faces > 0:
            face_detected = True

        if face_detected:
            head_frame_count = head_frame_count + 1
            for x, y, w, h in face_tuple:
                CompVision.draw_rectangle(frame, x, y, w, h, color=(155, 206, 101))
                if head_frame_count > capture_thres:
                    crop_head_frame(gray_frame, face_tuple)
                    save_all_frames(gray_frame, face_tuple)
                    head_frame_count = 0
        else:
            head_frame_count = 0
            num_faces = 0

        cv2.imshow('feed', frame)

        if frame_count % 40 == 0:
            destroy_empty_windows()
            print('\n'+str(frame_count))
            print('***************************')
            print('Number Of Faces: {}'.format(num_faces))
            print('Current Windows: {}'.format(current_windows))
            print('***************************' + '\n')

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    web_cam.release()
    cv2.destroyAllWindows()
    classify_many_photos()


def save_all_frames(curr_frame, rectangle_tuple):
    global num_faces
    if num_faces >= 1:
        for n in range(num_faces):

            path = PATH + '/People/Random/Class/' + str(time.time()) + '-' + str(n) + '.png'

            for (x, y, w, h) in rectangle_tuple[[n]]:
                cropped = curr_frame[y:y + h, x:x + w]
                cropped = cv2.resize(cropped, (160, 160))
                # Equalizing is attempting to spread entire grayscale spectrum (0-255) on an image
                cropped = cv2.equalizeHist(cropped)
                cv2.imwrite(path, cropped)
                print(path)


def get_video_file_name():
    PATH = os.getcwd()

    video_path = PATH + '/Recordings/'

    if not os.path.exists(video_path):
        os.mkdir(video_path)

    video_name = video_path + str(time.asctime()) + '.avi'

    if os.path.exists(video_name):
        print('Error, video file:{} already exists'.format(video_name))
        return video_path + 'default{}.avi'.format(random.randint(10, 10000))
    else:
        return video_name