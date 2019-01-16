# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
import CompVision
import CV_Detect
import random
import os

import cv2


import Facenet.face as face

verbose = bool()


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 0, 0), 1)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3],),
                            cv2.QT_FONT_NORMAL, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 76, 231),
                thickness=2, lineType=2)


def main(frame_interval=5):
    CompVision.create_main_cv2_window()
    fps_display_interval = 1  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output = cv2.VideoWriter(get_video_file_name(), fourcc, 20, (640, 480))

    if verbose:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.equalizeHist(frame)

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count = frame_count + 1
        CompVision.render_image('feed', frame)
        output.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    output.release()


def get_video_file_name():
    PATH = os.getcwd()

    video_path = PATH + '/Recordings/Detected/'

    if not os.path.exists(video_path):
        os.mkdir(video_path)

    video_name = video_path + str(time.asctime()) + '.avi'

    if os.path.exists(video_name):
        print('Error, video file:{} already exists'.format(video_name))
        return video_path + 'default{}.avi'.format(random.randint(10, 10000))
    else:
        return video_name


def run(frame_interval=5, _verbose=False):
    global verbose
    verbose = _verbose
    main(frame_interval=frame_interval)