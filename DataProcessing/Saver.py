import os
import CompVision


def check_database(person_name):
    """
    :param person_name: Persons Name
    :return: True Or False if person found, their index position
    """
    person_name = person_name.upper()
    found = False
    index = int()

    for line in file_list:
        name = line.rstrip()[:-5]
        if name == person_name:
            found = True
            break
        else:
            index = index + 1

    return found, index


def person_exists(person_name):

    person_name = person_name.upper()
    person_path = PATH + '/People/Raw/' + person_name

    if os.path.exists(person_path):
        return True
    else:
        return False


def add_person(person_name):

    person_name = person_name.upper()
    person_path = PATH + '/People/Raw/' + person_name
    string = '{} : XX\n'.format(person_name)

    if person_exists(person_name):
        print('{} Already Exists'.format(person_name))
    else:
        file_list.append(string)
        os.mkdir(person_path)
        print('{} Folder Created'.format(person_name))
        print('Because this is a new user, you must add 10 to 99 new Photos')
        while True:
            num_photos = int(input('How many photos would you like to add: '))
            if num_photos < 10 or num_photos > 99:
                print('Please input a number from 10 to 99 ')
            else:
                break
        add_photos(person_name, num_photos)


def add_photos(person_name, num_photos, mode='LIVE', _faces=list()):
    person_name = person_name.upper()
    count = int()
    try:
        if mode == 'LIVE':
            faces = CompVision.read_for_one_face_iter(num_photos)
        else:
            faces = _faces
        face_index = 0
        for _ in range(num_photos):
            person_path = PATH + '/People/Raw/' + person_name + '/'
            photo_path = person_path + person_name + '-' + str(get_num_photos(person_name)) + '.png'

            image = faces[face_index]
            CompVision.save_image(photo_path, image)
            count = count + 1
            face_index = face_index + 1

        print("Successfully added {} photos for {}".format(num_photos, person_name))

    except Exception as e:
        print(e)
        print('Process Interrupted, Added {}/{} photos for {}'.format(count, num_photos, person_name))
    finally:
        update_photo_numbers(person_name)


def update_photo_numbers(person_name):
    found = False
    index = int()
    person_name = person_name.upper()
    person_path = PATH + '/People/Raw/' + person_name
    person_num_photos = len(os.listdir(person_path))

    string = '{} : {}\n'.format(person_name, person_num_photos)

    for line in file_list:
        name = line.rstrip()[:-5]

        if name == person_name:
            found = True
            file_list[index] = string
            break
        index = index + 1

    if not found:
        file_list.append(string)

    write_to_file()


def write_to_file():
    os.remove(PATH + '/People/People.txt')
    new_file = open(PATH + '/People/People.txt', 'x')
    for line in file_list:
        new_file.write(line)
    new_file.close()


def get_num_photos(person_name):
    person_name = person_name.upper()
    person_path = PATH + '/People/Raw/{}'.format(person_name)

    if person_exists(person_name):
        num_photos = len(os.listdir(person_path))
    else:
        num_photos = 0
        print('%s Doesnt Exist' % person_name)

    return num_photos


def get_names():
    names = os.listdir('People/Raw/')

    return names



def update_all_photo_nums():

    for person in file_list:
        person = person[:-5].rstrip()
        update_photo_numbers(person)


if __name__ == 'DataProcessing.Saver':
    PATH = os.getcwd()

    person_file = open(PATH + '/People/People.txt', 'r+')
    file_list = person_file.readlines()