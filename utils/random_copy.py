"""
this python script is to randomly select a portion of files and copy to another directory.

"""
# -*- coding: UTF-8 -*-

import os
import os.path
import shutil
import random

def moveFile(srcDir, tarDir):
    file_names = os.listdir(srcDir)
    json_filenames = []
    for filename in file_names:
        if filename.split('.')[-1] in ['json']:
            json_filenames.append(filename.split('.')[0])
    json_file_num = len(json_filenames)
    assert(json_file_num*2 == len(file_names))
    rate = 0.1
    picknumber = int(json_file_num * rate)
    sample = random.sample(json_filenames, picknumber)
    # print (sample)
    count = 0
    for name in sample:
        json_name = os.path.split(name)[-1]+'.json'
        img_name = os.path.split(name)[-1] + '.jpg'
        shutil.copy(os.path.join(srcDir, json_name), os.path.join(tarDir, json_name))
        shutil.copy(os.path.join(srcDir, img_name), os.path.join(tarDir, img_name))
        # if count >=10:
        #     break
        count += 1
        print("processing: {}".format(count))
    return


if __name__ == '__main__':
    srcDir = "/home/aia1/ccj/face_landmark/dataset/300vw"
    tarDir = "/home/aia1/ccj/face_landmark/dataset/random-select"
    moveFile(srcDir, tarDir)

