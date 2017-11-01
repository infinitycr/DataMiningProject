# _*_ coding:utf-8 _*_

import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures


def print_mean_var(list):
    if list == None:
        return -1
    pos_list = [abs(elem) for elem in list]
    print(pos_list)
    narray = np.array(pos_list)
    mean = narray.mean()
    var = narray.var()
    # for i in range(len(narray)):
    #     if narray[i]>5:
    #         print(i)
    print("mean is: ", mean)
    print("var is: ", var)
    print("max is: ", narray.max())
    print("min is: ", narray.min())
    return


def execute_sublist_v1(list):
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
    return feature, lat, long


def execute_sublist_v2(palist):
    """
    去除feature的最后4个samples，去除long和lat的最前4个samples
    :param list:
    :return:feature list，lat list 和 long list
    """
    feature = []
    long = []
    lat = []
    for element in palist:
        long.append(element[-3])
        lat.append(element[-4])
        feature.append(element[:])
    for i in range(len(palist) - 3):
        tmp = [lat[i + 1], lat[i + 2], lat[i + 3]]
        feature[i].extend(tmp)
        tmp = [long[i + 1], long[i + 2], long[i + 3]]
        feature[i].extend(tmp)
    for i in range(4):
        del feature[-1]
        del long[0]
        del lat[0]
    return feature, lat, long


def execute_sublist_v3(pathlist):
    """
    向前窥探3个时间点的feature
    :param pathlist:
    :return:
    """
    feature = []
    long = []
    lat = []
    for element in pathlist:
        long.append(element[-3])
        lat.append(element[-4])
        feature.append(element[:])
    for i in range(3, len(pathlist) - 3):
        tmp = [lat[i - 3], lat[i - 2], lat[i - 1]]
        feature[i].extend(tmp)
        tmp = [long[i - 3], long[i - 2], long[i - 1]]
        feature[i].extend(tmp)
    for i in range(3):
        del feature[0]
    for i in range(4):
        del feature[-1]
    for i in range(7):
        del lat[0]
        del long[0]
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
                    if len(temp_list) < 7:
                        continue
                    else:
                        temp_fea, temp_lat, temp_long = execute_sublist_v3(temp_list)
                        feature_list.extend(temp_fea)
                        latitude_targets_list.extend(temp_lat)
                        longitude_targets_list.extend(temp_long)
                        temp_list = []
                        continue
                list = line.strip().split()[-5:]
                temp_list.append(list)
            if len(temp_list) >= 7:
                temp_fea, temp_lat, temp_long = execute_sublist_v3(temp_list)
                feature_list.extend(temp_fea)
                latitude_targets_list.extend(temp_lat)
                longitude_targets_list.extend(temp_long)
    print("Executing finished!")
    feature_array = np.array(feature_list)
    latitude_array = np.array(latitude_targets_list)
    longitude_array = np.array(longitude_targets_list)

    return feature_array, latitude_array, longitude_array


def load_data_as_test_ndarray(filepath):
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
                        temp_fea, temp_lat, temp_long = execute_sublist_v1(temp_list)
                        feature_list.extend(temp_fea)
                        latitude_targets_list.extend(temp_lat)
                        longitude_targets_list.extend(temp_long)
                        temp_list = []
                        continue
                list = line.strip().split()[-5:]
                temp_list.append(list)
            temp_fea, temp_lat, temp_long = execute_sublist_v1(temp_list)
            feature_list.extend(temp_fea)
            latitude_targets_list.extend(temp_lat)
            longitude_targets_list.extend(temp_long)
    print("Executing finished!")
    feature_array = np.array(feature_list)
    latitude_array = np.array(latitude_targets_list)
    longitude_array = np.array(longitude_targets_list)

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

    x_train = feature_array[:]

    y_latitude_train = latitude_targets_array[:]
    y_longitude_train = longitude_targets_array[:]

    quadra_feature = PolynomialFeatures(degree=1)
    x_train_quadra = quadra_feature.fit_transform(x_train)

    latitude_regr = linear_model.LinearRegression()
    latitude_regr.fit(x_train_quadra, y_latitude_train)

    longitude_regr = linear_model.LinearRegression()
    longitude_regr.fit(x_train_quadra, y_longitude_train)

    test_feature, test_lat, test_long = load_data_as_ndarray(r"D:\demo4test\testingset")

    test_feature_quadra = quadra_feature.fit_transform(test_feature)
    lat_predict = latitude_regr.predict(test_feature_quadra)
    long_predict = longitude_regr.predict(test_feature_quadra)

    # 数据源是以0.1°N和0.1°E记录的，计算真实值
    lat_infact_predict = [e * 0.1 for e in lat_predict]
    long_infact_predict = [e * 0.1 for e in long_predict]
    test_lat_infact = [e.astype(int) * 0.1 for e in test_lat]
    test_long_infact = [e.astype(int) * 0.1 for e in test_long]

    # print(lat_predict)
    # print(lat_infact_predict)
    # print(test_lat)
    # print(test_lat_infact)
    # print(long_predict)
    # print(long_infact_predict)
    # print(test_long)
    # print(test_long_infact)

    list1 = []
    list2 = []
    for i in range(len(lat_infact_predict)):
        list1.append(abs(lat_infact_predict[i] - float(test_lat_infact[i])))
        list2.append(abs(long_infact_predict[i] - float(test_long_infact[i])))

    print_mean_var(list1)
    print_mean_var(list2)
    # plt.plot(list(range(len(list1))), list1, 'r', label='Lat_diff')
    # plt.plot(list(range(len(list2))), list2, 'b', label='Long_diff')

    # 预测与真实的纬度比较
    # plt.plot(list(range(len(lat_infact_predict))), lat_infact_predict, 'r', label='Latitude_predict')
    # plt.plot(list(range(len(test_lat_infact))), test_lat_infact, 'b', label='Latitude_2014_2015')

    # 预测与真实的经度比较
    # plt.plot(list(range(len(long_predict))), long_predict, 'k', label='Longitude_predict')
    # plt.plot(list(range(len(test_long))), test_long, 'm', label='Longitude_2014_2015')


    # 计算欧氏距离
    distance_list = []
    for i in range(len(lat_infact_predict)):
        distance_list.append(
            euclidean_distance(float(lat_infact_predict[i]), float(long_infact_predict[i]),
                               float(test_lat_infact[i]),
                               float(test_long_infact[i])))
    print(print_mean_var(distance_list))
    count_lessthan_500 = 0
    count_lessthan_400 = 0
    count_lessthan_300 = 0
    count_lessthan_200 = 0
    count_lessthan_100 = 0
    count_lessthan_50 = 0
    for i in range(len(distance_list)):
        if distance_list[i] < 50:
            count_lessthan_50 += 1
        if distance_list[i] < 100:
            count_lessthan_100 += 1
        if distance_list[i] < 200:
            count_lessthan_200 += 1
        if distance_list[i] < 300:
            count_lessthan_300 += 1
        if distance_list[i] < 400:
            count_lessthan_400 += 1
        if distance_list[i] < 500:
            count_lessthan_500 += 1
    print("Sum: ", len(distance_list))
    print("<500: ", count_lessthan_500)
    print("<400: ", count_lessthan_400)
    print("<300: ", count_lessthan_300)
    print("<200: ", count_lessthan_200)
    print("<100: ", count_lessthan_100)
    print("<50: ", count_lessthan_50)

    print("<500: ", count_lessthan_500 / len(distance_list))
    print("<400: ", count_lessthan_400 / len(distance_list))
    print("<300: ", count_lessthan_300 / len(distance_list))
    print("<200: ", count_lessthan_200 / len(distance_list))
    print("<100: ", count_lessthan_100 / len(distance_list))
    print("<50: ", count_lessthan_50 / len(distance_list))
    # for i in range(len(distance_list)):
    #     print(distance_list[i])
    # 预测与真实的欧几里得距离
    plt.plot(list(range(len(distance_list))), distance_list, 'c', label='distance')

    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()
