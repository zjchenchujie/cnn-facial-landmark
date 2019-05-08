import os
import random
import h5py
import numpy as np
from skimage import io
import json
# import cv2

def save2hdf5(X, Y, filename):
    with h5py.File(filename, 'w') as f:
        # f['data'] = X
        # f['label'] = Y
        f.create_dataset('data', data=X)
        f.create_dataset('label', data=Y)
        print('having saving a hdf5 file !')
    # f.close()


def convert(source_path, pairlist, savepath, hdf5list, TARGET_WIDHT, TARGET_HEIGHT):
    step = 5000

    fid = open(pairlist)
    lines = fid.read().splitlines()
    fid.close()
    X = np.empty((step, 3, TARGET_WIDHT, TARGET_HEIGHT), dtype=np.float)
    Y = np.empty((step, 136, 1, 1), dtype=np.float)
    i = 0
    t = 1

    # print ("list[0]", lines[0])
    # print ("list[1]", lines[1])

    for line in lines:
        line = line.split('.')[0]
        image_name = line + '.jpg'
        json_url = line + '.json'
        print("processing: %d\n ", i)
        with open(json_url) as json_file:
            points = json.load(json_file)
        img = io.imread(image_name, as_grey=False)
        img = (img - 127.5) / 128  #
        # im=np.array(im)
        img = img.transpose(2, 0, 1)  #
        # im=tf.resize(im,(w,h))
        X[i, :, :, :] = img.astype(np.float32)
        Y[i, :, 0, 0] = points
        i = i + 1

        if i == step:
            #
            # mean = X.mean(axis=0)
            # mean = mean.mean(1).mean(1)
            #
            # for m in range(len(X)):
            #   X[m][0]=X[m][0]-mean[0]
            #  X[m][1]=X[m][1]-mean[1]
            # X[m][2]=X[m][2]-mean[2]

            data_type = pairlist.split('.')[0]
            _, tail = os.path.split(data_type)
            filename = os.path.join(savepath, tail + '_' + str(t) + '.hdf5')
            save2hdf5(X, Y, filename)
            with open(os.path.join(savepath, hdf5list), 'a') as f:
                f.write(filename + '\n')
            i = 0
            t = t + 1

    if i > 0:
        # mean = X.mean(axis=0)
        # mean = mean.mean(1).mean(1)
        # for m in range(len(X)):
        #   X[m][0]=X[m][0]-mean[0]
        #  X[m][1]=X[m][1]-mean[1]
        # X[m][2]=X[m][2]-mean[2]
        data_type = pairlist.split('.')[0]
        _, tail = os.path.split(data_type)
        filename = os.path.join(savepath, tail + str(t) + '.hdf5')
        save2hdf5(X[0:i, :, :, :], Y[0:i, :, :, :], filename)
        with open(os.path.join(savepath, hdf5list), 'a') as f:
            f.write(filename + '\n')


if __name__ == '__main__':
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128
    image_source_path = '/home/aia1/ccj/face_landmark/dataset/data/random-select'
    filelist = '/home/aia1/ccj/face_landmark/dataset/file_list/random-select/train.txt'
    save_path = '/home/aia1/ccj/face_landmark/dataset/HDF5/random-select/'
    hdf5list = '/home/aia1/ccj/face_landmark/dataset/HDF5/random-select/train_hdf5.txt'
    convert(image_source_path, filelist, save_path, hdf5list, TARGET_WIDTH, TARGET_HEIGHT)

    # source_path = '/media/crw/MyBook/MyDataset/FacePoint/test39X39/'
    # save_path = '/media/crw/MyBook/TrainData/HDF5/FacePoint/10000_39X39/test/'
    # hdf5list='/media/crw/MyBook/TrainData/HDF5/FacePoint/10000_39X39/test/test.txt'
    # filelist = '/media/crw/MyBook/MyDataset/FacePoint/test39X39/test.list'
    # convert(source_path,filelist,save_path,hdf5list,w,h)