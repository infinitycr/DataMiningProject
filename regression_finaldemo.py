# _*_ coding:utf-8 _*_

import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import Dense, Dropout


def print_mean_var(list):
    """
    打印出以list构成的np数组的mean和variance
    :param list: 数值列表
    :return: null
    """
    if len(list) == 0:
        return
    pos_list = [abs(elem) for elem in list]
    print(pos_list)
    narray = np.array(pos_list)
    mean = narray.mean()
    var = narray.var()
    print("mean is: ", mean)
    print("var is: ", var)
    return


def execute_sublist_v2(list):
    """
    去除feature的最后4个samples，去除long和lat的最前4个samples
    :param list:
    :return:feature list，lat list 和 long list
    """
    feature = []
    level = []
    paska = []
    long = []
    lat = []
    speed = []
    for element in list:
        level.append(element[-5])
        lat.append(element[-4])
        long.append(element[-3])
        paska.append(element[-2])
        speed.append(element[-1])
        feature.append(element[:])
    for i in range(len(list) - 3):
        tmp = [level[i + 1], lat[i + 1], long[i + 1], paska[i + 1], speed[i + 1], level[i + 2], lat[i + 2], long[i + 2],
               paska[i + 2], speed[i + 2], level[i + 3], lat[i + 3], long[i + 3], paska[i + 3], speed[i + 3]]
        feature[i].extend(tmp)
    for i in range(4):
        del feature[-1]
        del level[0]
        del long[0]
        del lat[0]
        del paska[0]
        del speed[0]
    # print(feature)
    # sys.exit(1)
    return feature, level, lat, long, paska, speed


def sublist_testingset(typh_list):
    fea5_list = []
    for element in typh_list:
        fea5_list.append(element[:])
    return fea5_list


def get_test(filepath):
    """
    获取测试集要用到的lat和long,剔除前七条数据
    :param filepath:
    :return:
    """
    test_lat = []
    test_long = []
    whole_typh_list = []
    for dir, sub_dir, files in os.walk(filepath, topdown=False):
        for file in files:
            each_typh_list = []
            for line in open(os.path.join(filepath, file), 'r', encoding='utf-8').readlines():
                if line.strip().split()[0] == '66666':
                    if len(each_typh_list) < 7:
                        continue
                    else:
                        tmp_latlong = sublist_testingset(each_typh_list)
                        whole_typh_list.append(tmp_latlong)
                        each_typh_list = []
                        continue
                list = line.strip().split()[-4:-2]
                each_typh_list.append(list)
            tmp_latlong = sublist_testingset(each_typh_list)
            whole_typh_list.append(tmp_latlong)
    for elem in whole_typh_list:
        for i in range(7):
            del elem[0]
    for typhoon in whole_typh_list:
        for record in typhoon:
            test_lat.append(float(record[0]) * 0.1)
            test_long.append(float(record[1]) * 0.1)
    # print(test_lat)
    # print(test_long)
    return test_lat, test_long


def load_data_as_ndarray(filepath):
    """
    读取路径数据,处理成两个numpy array
    :param filepath: 文件路径
    :return: features array,纬度 targets array和经度 targets array
    """
    feature_list = []
    level_list = []
    latitude_targets_list = []
    longitude_targets_list = []
    paska_list = []
    speed_list = []
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
                        temp_fea, temp_level, temp_lat, temp_long, temp_paska, temp_speed = execute_sublist_v2(
                            temp_list)
                        feature_list.extend(temp_fea)
                        level_list.extend(temp_level)
                        latitude_targets_list.extend(temp_lat)
                        longitude_targets_list.extend(temp_long)
                        paska_list.extend(temp_paska)
                        speed_list.extend(temp_speed)
                        temp_list = []
                        continue
                list = line.strip().split()[-5:]
                temp_list.append(list)
            temp_fea, temp_level, temp_lat, temp_long, temp_paska, temp_speed = execute_sublist_v2(temp_list)
            feature_list.extend(temp_fea)
            level_list.extend(temp_level)
            latitude_targets_list.extend(temp_lat)
            longitude_targets_list.extend(temp_long)
            paska_list.extend(temp_paska)
            speed_list.extend(temp_speed)
    print("Executing finished!")
    feature_array = np.array(feature_list)
    level_array = np.array(level_list)
    latitude_array = np.array(latitude_targets_list)
    longitude_array = np.array(longitude_targets_list)
    paska_array = np.array(paska_list)
    speed_array = np.array(speed_list)
    return feature_array, level_array, latitude_array, longitude_array, paska_array, speed_array


def euclidean_distance(lat1, long1, lat2, long2):
    """
    根据经纬度计算欧氏距离
    :param lat1: 纬度
    :param long1: 经度
    :param lat2: 纬度
    :param long2: 经度
    :return: 两个经纬度对之间的欧氏距离
    """
    eu_dist = 0.0
    latitude1 = (math.pi / 180.0) * lat1
    latitude2 = (math.pi / 180.0) * lat2
    longitude1 = (math.pi / 180.0) * long1
    longitude2 = (math.pi / 180.0) * long2
    # 地球半径
    R = 6378.1
    # {arccos[sina*sinx+cosb*cosx*cos(b-y)]}*R  (a,b,x,y)
    temp = math.sin(latitude1) * math.sin(latitude2) + math.cos(latitude1) * math.cos(latitude2) * math.cos(
        longitude2 - longitude1)
    if float(repr(temp)) > 1.0:
        temp = 1.0
    eu_dist = math.acos(temp) * R
    return eu_dist


if __name__ == '__main__':
    feature_array, level_targets_array, latitude_targets_array, longitude_targets_array, paska_targets_array, speed_targets_array = load_data_as_ndarray(
        r"D:\demo4test\dataset")

    x_train = feature_array[:]

    y_latitude_train = latitude_targets_array[:]
    y_longitude_train = longitude_targets_array[:]

    print("training...")

    level_regr = linear_model.LinearRegression(normalize=True)
    level_regr.fit(x_train, level_targets_array)

    latitude_regr = linear_model.LinearRegression(normalize=True)
    latitude_regr.fit(x_train, y_latitude_train)

    longitude_regr = linear_model.LinearRegression(normalize=True)
    longitude_regr.fit(x_train, y_longitude_train)

    paska_regr = linear_model.LinearRegression(normalize=True)
    paska_regr.fit(x_train, paska_targets_array)

    speed_regr = linear_model.LinearRegression(normalize=True)
    speed_regr.fit(x_train, speed_targets_array)

    whole_typh_list = []
    for dir, sub_dir, files in os.walk(r"D:\demo4test\testingset", topdown=False):
        for file in files:
            each_typh_list = []
            for line in open(os.path.join(r"D:\demo4test\testingset", file), 'r', encoding='utf-8').readlines():
                if line.strip().split()[0] == '66666':
                    if len(each_typh_list) < 7:
                        continue
                    else:
                        tmp_latlong = sublist_testingset(each_typh_list)
                        whole_typh_list.append(tmp_latlong)
                        each_typh_list = []
                        continue
                list = line.strip().split()[-5:]
                each_typh_list.append(list)
            tmp_latlong = sublist_testingset(each_typh_list)
            whole_typh_list.append(tmp_latlong)

    predict_list = []

    for elem in whole_typh_list.copy():
        temp_each = elem.copy()
        # print(temp_each)
        each_pred = []
        if len(elem) < 7:
            continue
        for i in range(3, len(elem) - 4):
            feature = []
            for j in range(i - 3, i + 1):
                for k in range(5):
                    feature.append(temp_each[j][k])
            np_fea = np.array(feature)
            level6 = level_regr.predict(np_fea.reshape(1, -1).astype(float))
            lat6 = latitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            long6 = longitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            paska6 = paska_regr.predict(np_fea.reshape(1, -1).astype(float))
            speed6 = speed_regr.predict(np_fea.reshape(1, -1).astype(float))
            # print(feature)
            feature.extend([int(level6[0]), int(lat6[0]), int(long6[0]), int(paska6[0]), int(speed6[0])])
            feature = feature[5:]
            # print(feature)
            np_fea = np.array(feature)
            level12 = level_regr.predict(np_fea.reshape(1, -1).astype(float))
            lat12 = latitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            long12 = longitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            paska12 = paska_regr.predict(np_fea.reshape(1, -1).astype(float))
            speed12 = speed_regr.predict(np_fea.reshape(1, -1).astype(float))
            # print(feature)
            feature.extend([int(level12[0]), int(lat12[0]), int(long12[0]), int(paska12[0]), int(speed12[0])])
            feature = feature[5:]
            # print(feature)
            np_fea = np.array(feature)
            level18 = level_regr.predict(np_fea.reshape(1, -1).astype(float))
            lat18 = latitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            long18 = longitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            paska18 = paska_regr.predict(np_fea.reshape(1, -1).astype(float))
            speed18 = speed_regr.predict(np_fea.reshape(1, -1).astype(float))
            feature.extend([int(level18[0]), int(lat18[0]), int(long18[0]), int(paska18[0]), int(speed18[0])])
            # print(feature)
            level24 = level_regr.predict(np_fea.reshape(1, -1).astype(float))
            lat24 = latitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            long24 = longitude_regr.predict(np_fea.reshape(1, -1).astype(float))
            paska24 = paska_regr.predict(np_fea.reshape(1, -1).astype(float))
            speed24 = speed_regr.predict(np_fea.reshape(1, -1).astype(float))

            each_pred.append([float(lat24[0]), float(long24[0])])
        predict_list.append(each_pred)

    # 预测的lat和long乘以0.1
    lat_infact_predict = []
    long_infact_predict = []
    for typh in predict_list.copy():
        for elem in typh.copy():
            lat_infact_predict.append(float(elem[0])*0.1 )
            long_infact_predict.append(float(elem[1])*0.1)
    # print(predlat_to_fact)
    # print(predlong_to_fact)
    test_lat_infact, test_long_infact = get_test(r"D:\demo4test\testingset")

    print(lat_infact_predict)
    print(test_lat_infact)
    print(long_infact_predict)
    print(test_long_infact)
    # print(len(lat_infact_predict))
    # print(len(test_lat_infact))
    # print(len(long_infact_predict))
    # print(len(test_long_infact))


    # list1 纬度预测误差(单位°N)
    list1 = []
    # list2 经度预测误差(单位°E)
    list2 = []
    for i in range(len(lat_infact_predict)):
        list1.append(abs(float(lat_infact_predict[i]) - float(test_lat_infact[i])))
        list2.append(abs(float(long_infact_predict[i]) - float(test_long_infact[i])))

    print_mean_var(list1)
    print_mean_var(list2)
    # plt.plot(np.arange(0,len(lat_infact_predict)), list1, 'r', label='Lat_diff')
    # plt.plot(np.arange(0,len(long_infact_predict)), list2, 'b', label='Long_diff')

    # 预测与真实的纬度比较
    plt.plot(np.arange(0, len(lat_infact_predict)), lat_infact_predict, 'r', label='Latitude_predict')
    plt.plot(np.arange(0, len(test_lat_infact)), test_lat_infact, 'b', label='Latitude_2014_2015')

    # 预测与真实的经度比较
    plt.plot(np.arange(0, len(long_infact_predict)), long_infact_predict, 'k', label='Longitude_predict')
    plt.plot(np.arange(0, len(test_long_infact)), test_long_infact, 'm', label='Longitude_2014_2015')

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
    # 预测与真实的欧几里得距离
    # plt.plot(np.arange(0, len(distance_list)), distance_list, 'c', label='distance')

    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()
