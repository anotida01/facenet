import cv2
import time
import os

capture_threshold = 15
cascade_scale_factor = 1.25
cascade_min_neighbour = 10
face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
num_faces = int()
frame_count = int()
head_frame_count = int()
cv2_font = cv2.QT_FONT_NORMAL


def read_for_one_face():
    web_cam = cv2.VideoCapture(0)
    global head_frame_count
    global num_faces
    head_frame_count = 0

    create_main_cv2_window()

    while True:
        _, frame = web_cam.read()
        frame, roi_face = grab_face(frame)

        if len(roi_face) > 1:
            cropped = True
        else:
            cropped = False

        cv2.imshow('feed', frame)
        key = cv2.waitKey(1)

        if cropped:
            web_cam.release()
            return roi_face
        elif key == ord('q'):
            web_cam.release()
            return []


def create_main_cv2_window():
    cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('feed', 1400, 1050) # TODO 4k
    cv2.resizeWindow('feed', 700, 505) # TODO 1080p
    cv2.moveWindow('feed', 100, 50)


def draw_rectangle(curr_frame, x_cord, y_cord, length, height, color):
    """
    :param curr_frame: Current Frame To Be Drawn On
    :param x_cord: X Co-Ordinate
    :param y_cord: Y Co-Ordinate
    :param length: Frame Length
    :param height: Frame Height
    :param color: Color Values in format: (XXX, XXX, XXX) / BGR
    :return: Void Function
    """

    pt1 = (x_cord, y_cord)
    pt2 = (x_cord + length, y_cord + height)

    cv2.rectangle(curr_frame, pt1=pt1, pt2=pt2, color=color, thickness=2)


def crop_head_frame(curr_frame, rectangle_tuple):
    """

    :param curr_frame: Frame where the head has been detected
    :param rectangle_tuple: Tuple containing Rectangle Details - (x_cord, y_cord, width, height)
    :return: Returns ROI Image Frame of the head.
    """
    cropped = list()

    create_new_cv2_window('crop', (300, 300)) # TODO Changed for 1080p

    for (x, y, w, h) in rectangle_tuple:
        cropped = curr_frame[y:y + h, x:x + w]
        cropped = cv2.equalizeHist(cropped)
        cropped = cv2.resize(cropped, (160, 160))
        cv2.imshow('crop', cropped)

    return cropped


def create_new_cv2_window(window_name, dimensions):

    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, dimensions[0], dimensions[1])

    cv2.moveWindow(window_name, 750, 0)  # TODO 1080p


def take_a_photo():
    create_main_cv2_window()

    face = read_for_one_face()
    face = cv2.resize(face, (160, 160))
    cv2.imshow('crop', face)
    cv2.imwrite('Predict/CLASS1/Predict.png', face)
    cv2.destroyWindow('feed')  # deprecated


def render_image(window_name, frame):
    cv2.imshow(window_name, frame)


def save_image(image_path, image):
    cv2.imwrite(image_path, image)


def read_for_one_face_iter(number_of_iters):
    web_cam = cv2.VideoCapture(0)

    faces = []
    cropped = False
    faces_capped = list()

    create_main_cv2_window()

    while True:
        _, frame = web_cam.read()
        frame, roi_face = grab_face(frame)

        if len(roi_face) > 1:
            cropped = True
        else:
            cropped = False

        cv2.imshow('feed', frame)
        key = cv2.waitKey(1)

        if cropped:
            print('face captured at %s' % time.time())
            faces.append(roi_face)

        faces_capped = len(faces)

        if number_of_iters == faces_capped or key == ord('q'):
            print('Exiting')
            web_cam.release()
            return faces


def grab_face(frame):
    global num_faces
    global head_frame_count
    roi_face = list()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_tuple = face_cascade.detectMultiScale(gray_frame, cascade_scale_factor, cascade_min_neighbour)
    num_faces = len(face_tuple)

    if num_faces > 0:
        head_frame_count = head_frame_count + 1

        for x, y, w, h in face_tuple:
            draw_rectangle(frame, x, y, w, h, color=(155, 206, 101))

            if head_frame_count > capture_threshold and num_faces == 1:
                roi_face = crop_head_frame(gray_frame, face_tuple)
                head_frame_count = 0
            elif num_faces > 1:
                cv2.putText(frame, 'REMEMBER: Only 1 Face at a Time!! :)', (0, 35), cv2_font, 0.7, (250, 10, 200), 1)
                head_frame_count = 0
    else:
        head_frame_count = 0
        cv2.putText(frame, 'No Faces Detected, Try Changing Light Conditions', (0, 35), cv2_font, 0.7, (250, 10, 200), 1)

    percentage_confidence = int((head_frame_count / capture_threshold) * 100)
    confidence_msg = 'HOLD STILL: {}%'.format(percentage_confidence)
    cv2.putText(frame, confidence_msg, (0, 60), cv2_font, 0.7, (250, 10, 200), 1)

    return frame, roi_face


def blur_photo(photo_dir=str()):
    try:
        image = cv2.imread(photo_dir)
        blur = cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=10)
        os.remove(photo_dir)
        cv2.imwrite(photo_dir, blur)
        print('%s was blurred' % photo_dir)
    except Exception as e:
        print(e, 'Blur failed, file may not exist')
