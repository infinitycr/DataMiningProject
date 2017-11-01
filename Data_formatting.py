# _*_ coding:utf-8 _*_
import os

from pyparsing import line


def to_space_split(srcfileName, dstfileName):
    """将文本文件filename处理成每行一条数据，每数据项均以空格隔开

    :param  srcfileName , dstfileName
    :return: return void
    """
    if os.path.exists(dstfileName):
        print("Target file already exists, cannot set up!\n")
        return
    newfile = open(dstfileName, 'w', encoding='utf-8')
    for line in open(srcfileName, encoding='utf-8').readlines():
        # elements = line.split()
        # " ".join(elements)
        # linestr = ""
        # for e in elements:
        #     linestr = linestr + e + ' '
        newfile.write(" ".join(line.split()))
        newfile.write('\n')
        # newfile.write(str(elements))
        # newfile.write('\n')

# test
# file = 'CH1949BST.txt'
# for line in open(file).readlines():
#     elements = line.split()
#     print(elements)
#
# to_space_split(file,'test.txt')
print(to_space_split.__doc__)
