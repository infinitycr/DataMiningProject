# _*_ coding:utf-8 _*_

import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
import math


def execute_sublist(list):
    """
    去除feature的最后4个samples，去除long和lat的最前4个samples
    :param list:
    :return:feature list，lat list 和 long list
    """
    feature = []
    long = []
    lat = []
    for element in list:
        long.append(element[-3])
        lat.append(element[-4])
        del element[1]
        del element[1]
        feature.append(element[:])
    for i in range(4):
        del feature[-1]
        del long[0]
        del lat[0]
    # print(feature)
    # print(long)
    # print(lat)
    return feature, lat, long


def load_data_as_ndarray(filepath):
    """
    读取路径数据,处理成两个numpy array
    :param filepath: 文件路径
    :return: features array,经度 targets array和纬度 targets array
    """
    # data_list = []
    feature_list = []
    latitude_targets_list = []
    longitude_targets_list = []
    if not os.path.exists(filepath):
        print("Invalid file path!")
        return
    for dir, sub_dir, files in os.walk(filepath, topdown=False):
        for file in files:
            print("Executing file:{0}".format(os.path.basename(file)))
            temp_list = []
            for line in open(os.path.join(filepath, file), 'r', encoding='utf-8').readlines():
                if line.strip().split()[0] == "66666":
                    if len(temp_list) < 5:
                        continue
                    else:
                        temp_fea, temp_lat, temp_long = execute_sublist(temp_list)
                        feature_list.extend(temp_fea)
                        latitude_targets_list.extend(temp_lat)
                        longitude_targets_list.extend(temp_long)
                        # print(temp_fea)
                        # print(len(temp_fea))
                        # print('-------------------------------')
                        # print(temp_long)
                        # print(len(temp_long))
                        # print('-------------------------------')
                        # print(temp_lat)
                        # print(len(temp_lat))
                        # print('-------------------------------')
                        temp_list = []
                        continue
                # print(line.strip().split())
                list = line.strip().split()[-5:]
                # print(list)
                temp_list.append(list)
    print("Executing finished!")
    # feature_list = []
    # longitude_targets_list = []
    # latitude_targets_list = []
    # for element in data_list:
    #     latitude_targets_list.append(element[-3])
    #     longitude_targets_list.append(element[-4])
    #     del element[1]
    #     del element[1]
    #     feature_list.append(element[:])
    # print(feature_list)
    # print('------------------')
    # print(longitude_targets_list)
    # print('------------------')
    # print(latitude_targets_list)
    feature_array = np.array(feature_list)
    latitude_array = np.array(latitude_targets_list)
    longitude_array = np.array(longitude_targets_list)
    # print(feature_array)
    # print('------------------')
    # print(longitude_array)
    # print('----------------------')
    # print(latitude_array)
    return feature_array, latitude_array, longitude_array


def euclidean_distance(lat1, long1, lat2, long2):
    """
    经纬度转换成欧氏距离
    :param lat: 纬度
    :param long: 经度
    :return: 距离
    """
    eu_dist = 0.0
    latitude1 = (math.pi / 180.0) * lat1
    latitude2 = (math.pi / 180.0) * lat2
    longitude1 = (math.pi / 180.0) * long1
    longitude2 = (math.pi / 180.0) * long2
    # 因此AB两点的球面距离为:{arccos[sina*sinx+cosb*cosx*cos(b-y)]}*R  (a,b,x,y)
    # 地球半径
    R = 6378.1
    temp = math.sin(latitude1) * math.sin(latitude2) + math.cos(latitude1) * math.cos(latitude2) * math.cos(
        longitude2 - longitude1)
    if float(repr(temp)) > 1.0:
        temp = 1.0
    eu_dist = math.acos(temp) * R
    return eu_dist


if __name__ == '__main__':
    feature_array, latitude_targets_array, longitude_targets_array = load_data_as_ndarray(r"D:\demo4test\dataset")
    # print(len(feature_array),len(longitude_targets_array),len(latitude_targets_array))
    # length is 39737
    # length = len(feature_array)


    x_train = feature_array[:30000]
    x_test = feature_array[30000:]

    y_longitude_train = longitude_targets_array[:30000]
    y_longitude_test = longitude_targets_array[30000:]

    y_latitude_train = latitude_targets_array[:30000]
    y_latitude_test = latitude_targets_array[30000:]

    longitude_regr = linear_model.LinearRegression()

    latitude_regr = linear_model.LinearRegression()

    longitude_regr.fit(x_train.astype(np.int32), y_longitude_train.astype(np.int32))

    latitude_regr.fit(x_train.astype(np.int32), y_latitude_train.astype(np.int32))

    y_longitude_predict = longitude_regr.predict(x_test.astype(np.int32))

    y_latitude_predict = latitude_regr.predict(x_test.astype(np.int32))

    # print("Longtitude coef: ", longitude_regr.coef_)
    #
    # print("Latitude coef: ", latitude_regr.coef_)
    #
    # print("Longtitude intercept: ", longitude_regr.intercept_)
    #
    # print("Latitude intercept: ", latitude_regr.intercept_)
    #
    # print(len(longitude_regr.coef_))
    long_coef = longitude_regr.coef_
    long_intercept = longitude_regr.intercept_
    lat_coef = latitude_regr.coef_
    lat_intercept = latitude_regr.intercept_
    test_feature, test_lat, test_long = load_data_as_ndarray(r"D:\demo4test\testingset")
    # test_feature = x_test
    # test_long = y_longitude_test
    # test_lat = y_latitude_test
    long_predict = []
    lat_predict = []
    for i in range(len(test_feature)):
        # print('123')
        # print(long_coef[0])
        # print(long_coef[1])
        # print(long_coef[2])
        # print(test_feature[i][0])
        # print(test_feature[i][1])
        # print(test_feature[i][2])
        long_temp = float(long_coef[0]) * float(test_feature[i][0]) + float(long_coef[1]) * float(
            test_feature[i][1]) + float(long_coef[2]) * float(test_feature[i][
                                                                  2]) + float(long_intercept)
        long_predict.append(long_temp)
        lat_temp = float(lat_coef[0]) * float(test_feature[i][0]) + float(lat_coef[1]) * float(
            test_feature[i][1]) + float(lat_coef[2]) * float(test_feature[i][
                                                                 2]) + float(lat_intercept)
        lat_predict.append(lat_temp)
    # plt.scatter(long_predict,test_long,color='black')
    print(lat_predict)
    print(test_lat)

    print(long_predict)
    print(test_long)

    distance_list = []
    for i in range(len(lat_predict)):
        distance_list.append(
            euclidean_distance(float(lat_predict[i]) * 0.1, float(long_predict[i]) * 0.1, float(test_lat[i]) * 0.1,
                               float(test_long[i]) * 0.1))

    #预测与真实的欧几里得距离
    print(distance_list)
    plt.plot(list(range(len(distance_list))), distance_list, 'c', label='distance')

    #预测与真实的纬度比较
    plt.plot(list(range(len(lat_predict))), lat_predict, 'r', label='Latitude_predict')
    plt.plot(list(range(len(test_lat))), test_lat, 'b', label='Latitude_2014_2015')

    #预测与真实的经度比较
    plt.plot(list(range(len(long_predict))), long_predict, 'k', label='Longitude_predict')
    plt.plot(list(range(len(test_long))), test_long, 'm', label='Longitude_2014_2015')

    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()
