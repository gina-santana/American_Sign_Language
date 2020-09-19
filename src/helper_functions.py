import os
import random


def search_and_destroy():
    '''
    Finds and deletes unexpected '.ipynb_checkpoints' files in data folders
    '''
    folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    for alpha in folders:
        for file in os.listdir(f'../../../test2/Training/{alpha}'):
            if file == '.ipynb_checkpoints':
                os.rmdir(file)
                print(f'File removed in {alpha}')         
        print(f'Nothing found in {alpha}')


def choose_validation_images():
    folders = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','nothing']
    for alpha in folders:
        for _ in range(900):
            random_number = random.randint(1, 3000)
            random_file = f'{alpha}{random_number}.jpg'
            while os.path.isfile(f'../../../test2/Training/{alpha}/{random_file}') == False:
                random_number = random.randint(1, 3000)
                random_file = f'{alpha}{random_number}.jpg'
            os.rename(f'../../../test2/Training/{alpha}/{random_file}', f'../../../test2/Validation/{alpha}/{random_file}')
            print(f'File moved from ../../../test2/Training/{alpha}/{random_file} to ../../../test2/Validation/{alpha}/{random_file}')         







