import os
import CV_Detect
import cv2
if not os.getcwd() == '/home/anotidaishe/PycharmProjects/facenet':
    os.chdir('/home/anotidaishe/PycharmProjects/facenet')
from DataProcessing import Saver
from Classifier import TRAIN
import time
from Facenet import real_time_recognition

PATH = os.getcwd() + '/'
# TODO: Adding photos for existing person needs to work


print('*****************************\n'
      '--Facial Recognition System--\n'
      'Uses  FaceNet For TensorFlow\n'
      '*****************************\n')

input_prompt = ('\n\t\t--Input Keys 0 - 10--\n'
                '0. Clear The Screen\n'
                '1. Add a new person to the database\n'
                '2. View people currently in the database\n'
                '3. View number of photos a person has\n'
                '4. Add photos for a person\n\n'
                '\t\t--TENSORFLOW OPS--\n\n'
                '5. Align faces using MTCNN\n'
                '6. Train model\n\n'
                '\t\t  --OPENCV OPS--\n\n'
                '7. Live classification feed\n'
                '8. Single classification feed (Higher Precision)\n'
                '9. Test Face Detection Parameters\n\n'
                '10. FULL OP (Add person & train & predict)\n'
                '11. Load Video & Classify'
                '\n\n98. Close All CV Windows'
                '\n99. Exit the program'
                '\n\nYour Input: ')


def add_photos():
    print('Option 4 - Add Additional Photos')
    while True:
        name = str(input('Person Name: ')).upper()
        if Saver.person_exists(name):
            num_photos = int(input('Number additional photos: '))
            Saver.add_photos(name, num_photos)
            break
        else:
            print('%s was not found.' % name)
            if str(input('Input 99 to Cancel, Press Enter to try again')) == '99':
                break


def get_num_photos():
    print('Option 3 - Obtaining Number Of Photos')
    Saver.update_all_photo_nums()
    person_name = str(input('Input person name: ')).upper()
    num_photos = Saver.get_num_photos(person_name)
    print('{} has {} photos'.format(person_name, num_photos))


def view_entire_database():
    print('--Entire Database--')
    people_dir = PATH + 'People/People.txt'
    os.system('cat {}'.format(people_dir))


def classify_single_image():
    print('Option 8 - Classifying for a single image')
    TRAIN.classify_photo()


def train_model():
    print('Option 6 - Train Model')
    TRAIN.train_model(int(input('Training Batch Size: ')))


def classify_live():
    print('Option 7 - Live Classification')
    real_time_recognition.run(int(input('Classification is performed every how many frames? : ')))


def full_op():
    train_model()
    align_faces()
    real_time_recognition.run()


def align_faces():
    TRAIN.align_faces()


def many_face():
    CV_Detect.run(float(input('Scale Factor: ')), int(input('Minimum Neighbours: ')))


def load_video():
    pass


def clear():
    os.system('clear')


def wait():
    input('Press enter key to continue')


def execute_option(option=int()):
    seconds = time.time()
    show_time = False
    if option == 0:
        clear()
    elif option == 1:
        Saver.add_person(str(input('Person Name: ')))
    elif option == 2:
        view_entire_database()
        wait()
    elif option == 3:
        get_num_photos()
        wait()
    elif option == 4:
        add_photos()
        wait()
    elif option == 5:
        align_faces()
        show_time = True
    elif option == 6:
        train_model()
        show_time = True
    elif option == 7:
        classify_live()
    elif option == 8:
        classify_single_image()
        wait()
    elif option == 9:
        many_face()
        wait()
    elif option == 10:
        Saver.add_person(str(input('Person Name: ')))
        full_op()
    elif option == 98:
        cv2.destroyAllWindows()
        print('Done!')
        wait()
        show_time = False
    else:
        print('Choice invalid, Input 99 to exit.')
        show_time = False
        wait()
    if show_time:
        seconds = time.time() - seconds
        print('Total Processing TIme: {}seconds'.format(seconds))
        wait()


while True:

    input_num = int(input(input_prompt))

    if input_num == 99:
        break
    execute_option(input_num)

    # Clean up for next instruction
    cv2.destroyAllWindows()
    clear()
