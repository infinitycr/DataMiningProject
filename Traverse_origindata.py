# _*_ coding:utf-8 _*_

import os
from builtins import print
import Data_formatting


def traverse_data(dir_path):
    """遍历dir_path目录，按Data_formatting.toSpaceSplit()处理目录下的文件

    :param dir_path: director to be processed
    :return:  return void
    """
    if not os.path.isdir(dir_path):
        print("Not a director!")
        return
        # if os.path.exists(dir_path):
        # print("Director already exists, cannot do it!")
        # return
    new_dir = r"D:\demo4test\dataset"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for direc, sub_dir, files in os.walk(dir_path, topdown=False):
        for file in files:
            absolute_file_path = os.path.join(dir_path, file)
            print("Now executing file: {0} ".format(absolute_file_path))
            destination_file_path = os.path.join(new_dir,
                                                 os.path.basename(absolute_file_path))
            Data_formatting.to_space_split(absolute_file_path,
                                           destination_file_path)
            print("After toSpaceSplit , generate a new file: {0}".format(destination_file_path))
            # test
            # file_path="123456789"
            # print("Now executing file: {0} ".format(file_path))
            # file=r"CH1949BST.txt"
            # new_dir=r"D:\demo4test"
            # destination_file_path=os.path.join(new_dir,file.split('\\')[-1])
            # dir_path=r"F:\数据挖掘\CMA热带气旋最佳数据集"
            # for dir, sub_dir, files in os.walk(dir_path, topdown=False):
            #     for file in files:
            #         absolute_file_path = os.path.join(dir_path, file)
            #         print("Now executing file: {0} ".format(absolute_file_path))
            #         destination_file_path = os.path.join(new_dir, os.path.basename(absolute_file_path))
            #         print("New file name is : {0} ".format(destination_file_path))
            # traverse_data(r"F:\数据挖掘\CMA热带气旋最佳数据集")
