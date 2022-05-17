import fnmatch
import os
import pandas as pd
import random
import numpy as np
import  glob

def changeFile():
    file = open("E:/wangyu_file/rs_Nas/src/data/lists/rs_test.lst", "r", encoding='UTF-8')
    file_list = file.readlines()
    file_name = []
    for i in range(file_list.__len__()):
        a = str(file_list[i].replace('test', 'train')).replace('\n', '')
        file_name.append(a)
    df = pd.DataFrame(file_name, columns=['one'])
    df.to_csv('E:/wangyu_file/rs_Nas/src/data/lists/rs_test1.lst', columns=['one'], index=False, header=False)

    file.close()

def mergeFile():
    file1 = open("E:/wangyu_file/1.lst", "r", encoding='UTF-8')
    file2 = open("E:/wangyu_file/2.lst", "r", encoding='UTF-8')
    file_list1 = file1.readlines()  # 将所有变量读入列表file_list1
    file_list2 = file2.readlines()  # 将所有变量读入列表file_list2
    file_list = []
    for i in range(file_list1.__len__()):
        a = str(file_list1[i])
        a = a.replace('\n', '').replace('\\', '/')
        b = str(file_list2[i])
        b = b.replace('\n', '').replace('\\', '/').replace('goundTruth', 'groundTruth')
        file_list.append(a + '\t'+ b)
    df = pd.DataFrame(file_list, columns=['one'])
    df.to_csv('E:/wangyu_file/rs_Nas/src/data/lists/rs_test.lst', columns=['one'], index=False, header=False)
    # file = open("train_pair.lst", "w")
    # file.writelines(file_list)
    file1.close()
    file2.close()
    # file.close()


def ReadSaveAddr(Stra, Strb):
    df = pd.DataFrame(np.arange(0).reshape(0, 1), columns=['Addr'])
    print(df)
    path = Stra
    for dirpath, dirnames, filenames in os.walk(path):
        filenames_len = filenames.__len__()
        # a_list = fnmatch.filter(os.listdir(dirpath),Strb)
        if filenames_len:
            dft = pd.DataFrame(np.arange(filenames_len).reshape((filenames_len, 1)), columns=['Addr'])
            # for i in range(len(filenames)):
            #     filenames[i] = filenames[i].replace('.tif', '.png')
            dft.Addr = filenames
            dft.Addr = dirpath.replace('E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/', '') + '/' + dft.Addr  # 输出绝对路径
            frames = [df, dft]
            df = pd.concat(frames)
            print(df.shape)
    df.to_csv('E:/wangyu_file/2.lst', columns=['Addr'], index=False, header=False)  # ***.lst即为最终保存的文件名，可修改
    print("Write To Get.lst !")

def ReadFileBytxt(txtfilename):
    txtfile = open(txtfilename, "r", encoding='UTF-8')
    file_list = txtfile.readlines()
    file = []

    for file_name in file_list:
        file_path = "/media/dell/DATA/wy/data/GID-5/4096/image/" + file_name.replace('.tif', '').replace('\n', '') + "/"
        imgs = glob.glob('{}*.tif'.format(file_path))
        if len(imgs) != 56:
            print('drong!')
        for img in imgs:
            img = img.replace('/media/dell/DATA/wy/data/GID-5/4096/', '')
            label = img.replace('image', 'label').replace('img.tif', 'label.png')
            file.append(img + '\t' + label)
    df = pd.DataFrame(file, columns=['one'])
    df.to_csv("/media/dell/DATA/wy/Seg_NAS/data/lists/gid_4096_test.lst", columns=['one'], index=False, header=False)

def ReadFileBypath(path):
    imgs = glob.glob(('{}*.tif'.format(path)))
    file = []
    for img in imgs:
        img = img.replace('/media/dell/DATA/wy/data/', '')
        label = img.replace('img', 'label').replace('320label512', '320label512_map')
        file.append(img + '\t' + label)
    df = pd.DataFrame(file, columns=['one'])
    # df = df.sample(frac=0.05, random_state=1)
    df.to_csv("/media/dell/DATA/wy/Seg_NAS/data/lists/uadataset/map_uad_512_test.lst", columns=['one'], index=False, header=False)

def make_lines(list):
    line_list = []
    for i in range(list.__len__()):
        a = str(list[i])
        a = a.replace('\n', '')
        line_list.append(a)
    return line_list

def split_lst_file(path):
    def data_split(full_list, ratio, shuffle=False):
        """
        数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
        :param full_list: 数据列表
        :param ratio:     子列表1
        :param shuffle:   子列表2
        :return:
        """
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2

    file = open(path, "r", encoding='UTF-8')
    file_list = file.readlines()  # 将所有变量读入列表file_list2
    file_list1, file_list2 = data_split(file_list, 0.5, shuffle=True)
    file_list1 = make_lines(file_list1)
    file_list2 = make_lines(file_list2)

    df1 = pd.DataFrame(file_list1, columns=['one'])
    df1.to_csv('/media/dell/DATA/wy/data/uadataset/mini_uad_512_train_1.lst', columns=['one'], index=False, header=False)
    df2 = pd.DataFrame(file_list2, columns=['one'])
    df2.to_csv('/media/dell/DATA/wy/data/uadataset/mini_uad_512_train_2.lst', columns=['one'], index=False, header=False)



if __name__ == '__main__':
    # InputStra="E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/test/label"#数据存在的路径
    # InputStra = "E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/train/label"
    # InputStrb = "*.png"
    # ReadSaveAddr(InputStra, InputStrb)
    # mergeFile()
    # changeFile()
    # path = "/media/dell/DATA/wy/data/uadataset/320img512/test/"
    # ReadFileBypath(path)
    path = '/media/dell/DATA/wy/data/uadataset/mini_uad_512_train.lst'
    split_lst_file(path)