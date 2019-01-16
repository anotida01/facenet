import Facenet.classifier as classifier
import CompVision
from Facenet.Align import align_dataset_mtcnn
import os
from DataProcessing import Saver

PATH = os.getcwd()


def train_model(batch_size):
    print('Model Train Started')
    classifier.run('TRAIN', batch_size=batch_size)
    print('Model Training Finished')


def align_faces(image_size=160, margin=32, gpu_memory_fraction=0.1, detect_multi_face=False):
    print('Face Alignment Started')
    align_dataset_mtcnn.run(image_size=image_size, margin=margin, gpu_memory_allocation=gpu_memory_fraction,
                            multi_face=detect_multi_face)
    print('Face Alignment Complete')
    print('Blur Started')
    blur_photos()
    print('Blur Completed')


def classify_photo():
    CompVision.take_a_photo()
    name = classifier.run()[0].upper()
    original_path = PATH + '/Predict/CLASS1/Predict.png'

    if not name == 'UNKNOWN':
        decision = str(input('Was Prediction Correct? [y/n]: ')).upper()
        if decision == 'Y':
            num_photos = Saver.get_num_photos(name)
            os.rename(original_path, PATH + '/People/Raw/{}/{}-{}.png'.format(name, name, num_photos))
            print('file renamed')
        else:
            os.rename(original_path, PATH + '/Predict/CLASS1/UNKNOWN.png')
    else:
        print('file named unknown')
        os.rename(original_path, PATH + '/Predict/CLASS1/UNKNOWN.png')


def classify_many_photos():
    folder = 'People/Random/Class/'
    class_list = classifier.run(_data_dir='People/Random')
    folder_list = os.listdir(folder)
    index = int()

    for name in class_list:
        old_pic = PATH + '/' + folder + folder_list[index]
        new_pic = PATH + '/' + folder + name + '-' + str(index)
        os.rename(old_pic, new_pic)
        index = index + 1

    folder_list = os.listdir(folder)
    for pic in folder_list:
        name = pic[:-2]
        if not name == 'UNKNOWN':
            num_photos = Saver.get_num_photos(name)
            if num_photos == 0:
                temp_name = name[:-1]
                if Saver.get_num_photos(temp_name) > 0:
                    name = temp_name
                else:
                    continue
            new_name = '{}-{}.png'.format(name, num_photos)
            old_dir = folder + pic
            new_dir = 'People/Raw/{}/{}'.format(name, new_name)
            os.rename(old_dir, new_dir)


def blur_photos():

    directory = PATH + '/People/Aligned/'
    people_names = os.listdir(directory)

    for name in people_names:
        person_dir = directory + str(name) + '/'

        try:
            for curr_photo in os.listdir(person_dir):
                photo_path = person_dir + curr_photo
                CompVision.blur_photo(photo_path)
        except NotADirectoryError:
            print('{} CAUGHT. NOT A PERSON DIR'.format(person_dir))
            continue


if __name__ == '__main__':
    print('This module can only be run from the main.py file')
    input()