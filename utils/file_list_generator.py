"""
This script shows how to generate a file name list in CSV format
"""
import csv
import os

import random


class ListGenerator:
    """Generate a list of specific files in directory."""

    def __init__(self):
        """Initialization"""
        # The list to be generated.
        self.file_list = []

    def generate_list(self, src_dir, save_dir, format_list=['jpg', 'png'], train_val_ratio=0.9, train_ratio=0.9):
        """Generate the file list of format_list in target_dir

        Args:
            target_dir: the directory in which files will be listed.
            format_list: a list of file extention names.

        Returns:
            a list of file urls.

        """
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.format_list = format_list
        self.train_val_ratio = train_val_ratio
        self.train_ratio = train_ratio

        # Walk through directories and list all files.
        for file_path, _, current_files in os.walk(self.src_dir, followlinks=False):
            current_image_files = []
            for filename in current_files:
                # First make sure the file is exactly of the format we need.
                # Then process the file.
                if filename.split('.')[-1] in self.format_list:
                    current_image_files.append(filename)

            sample_num = len(current_image_files)
            file_list_index = range(sample_num)
            tv = int(sample_num * self.train_val_ratio)
            tr = int(tv * self.train_ratio)
            print("train_val_num: {}\ntrain_num: {}".format(tv, tr))
            train_val = random.sample(file_list_index, tv)
            train = random.sample(train_val, tr)

            ftrain_val = open(self.save_dir + '/trainval.txt', 'w')
            ftest = open(self.save_dir + '/test.txt', 'w')
            ftrain = open(self.save_dir + '/train.txt', 'w')
            fval = open(self.save_dir + '/val.txt', 'w')
            for i in file_list_index:
                name = os.path.join(self.src_dir, current_image_files[i]) + '\n'
                if i in train_val:
                    ftrain_val.write(name)
                    if i in train:
                        ftrain.write(name)
                    else:
                        fval.write(name)
                else:
                    ftest.write(name)

            ftrain_val.close()
            ftrain.close()
            fval.close()
            ftest.close()
            #
            # for filename in current_files:
            #     # First make sure the file is exactly of the format we need.
            #     # Then process the file.
            #     if filename.split('.')[-1] in self.format_list:
            #         # Get file url.
            #         file_url = os.path.join(file_path, filename)
            #         self.file_list.append(file_url)

        return self.file_list

    # def save_list(self, list_name='list.csv'):
    #     """Save the list in csv format.
    #
    #     Args:
    #         list_name: the file name to be written.
    #
    #     """
    #     with open(list_name, 'w', newline='') as csv_file:
    #         writer = csv.DictWriter(csv_file, fieldnames=['file_url'])
    #
    #         # Write the header.
    #         writer.writeheader()
    #
    #         # Write all the rows.
    #         for each_record in self.file_list:
    #             writer.writerow({'file_url': each_record})


def main():
    """MAIN"""
    lg = ListGenerator()
    lg.generate_list(src_dir='/home/aia1/ccj/face_landmark/dataset/random-select',
                     save_dir='/home/aia1/ccj/face_landmark/dataset/file_list/random-select')
    print("Done !!")
    # lg.save_list()


if __name__ == '__main__':
    main()
