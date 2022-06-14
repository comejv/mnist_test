import numpy as np
from pandas import read_csv, DataFrame
from PIL import Image
import cv2
from sys import argv
import os


def test_to_images(test):
    try:
        os.mkdir('images_extraites')
    except FileExistsError:
        pass
    data = read_csv(test, header=None)
    data = np.array(data)[:, 1:]
    data = data.astype(np.uint8)
    data = data.reshape((data.shape[0], 28, 28))
    for i in range(data.shape[0]):
        new_image = Image.fromarray(data[i], mode='L')
        new_image.save(f'images_extraites/image_{i}.png', mode='L')


def images_to_test():
    n = 0
    df = np.array([])
    list_of_files = sorted(filter(lambda x: os.path.isfile(os.path.join(argv[2], x)),
                                  os.listdir(argv[2])))
    for file in list_of_files:
        img = cv2.imread(os.path.join(argv[2], file), 0)
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        img.resize((784))
        df = np.append(df, img)
        n += 1

    return df.reshape((n, 784))


if argv[1] == '-r':
    test_to_images(argv[2])

elif argv[1] == '-w':
    DataFrame(images_to_test()).to_csv('mnist_test_perso.csv', header=False)
