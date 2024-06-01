import os

import numpy as np
import pandas as pd
from numpy import mean

pwd_path = os.getcwd()
month = 1
day = 2
source_scv_path = f"{pwd_path}/order_book_3_2014_{month}_{day}.csv"


def order_book():
    order_book = pd.read_csv(source_scv_path, sep=',')
    print("读取原始数据成功")
    # print(order_book.head(5))
    # 1. 先梳理时间格式
    timestamp = np.array(order_book['Bid_Quantity'][0::4])
    bid_price_1 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Bid'][1::4])))
    bid_price_2 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Bid'][2::4])))
    bid_price_3 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Bid'][3::4])))
    bid_quantity_1 = np.array(list(map(float, order_book['Bid_Quantity'][1::4])))
    bid_quantity_2 = np.array(list(map(float, order_book['Bid_Quantity'][2::4])))
    bid_quantity_3 = np.array(list(map(float, order_book['Bid_Quantity'][3::4])))
    ask_price_1 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Ask'][1::4])))
    ask_price_2 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Ask'][2::4])))
    ask_price_3 = np.array(list(map(lambda x: float(x) / 100.0, order_book['Ask'][3::4])))
    ask_quantity_1 = np.array(list(map(float, order_book['Ask_Quantity'][1::4])))
    ask_quantity_2 = np.array(list(map(float, order_book['Ask_Quantity'][2::4])))
    ask_quantity_3 = np.array(list(map(float, order_book['Ask_Quantity'][3::4])))
    # 打印全部的数据
    # print(f"timestamp: {timestamp}")
    # print(f"bid_price_1: {bid_price_1}")
    # print(f"bid_price_2: {bid_price_2}")
    # print(f"bid_price_3: {bid_price_3}")
    # print(f"bid_quantity_1: {bid_quantity_1}")
    # print(f"bid_quantity_2: {bid_quantity_2}")
    # print(f"bid_quantity_3: {bid_quantity_3}")
    # print(f"ask_price_1: {ask_price_1}")
    # print(f"ask_price_2: {ask_price_2}")
    # print(f"ask_price_3: {ask_price_3}")
    # print(f"ask_quantity_1: {ask_quantity_1}")
    # print(f"ask_quantity_2: {ask_quantity_2}")
    # print(f"ask_quantity_3: {ask_quantity_3}")
    return (timestamp,
            order_book,
            bid_price_1, bid_price_2, bid_price_3,
            bid_quantity_1, bid_quantity_2, bid_quantity_3,
            ask_price_1, ask_price_2, ask_price_3,
            ask_quantity_1, ask_quantity_2, ask_quantity_3)


def time_transform(timestamp_time):
    """ 将时间转换为秒数，但是实现很奇怪。可能源数据格式只能这么处理 """
    time_second_basic = []
    time_second = []
    for i in range(0, len(timestamp_time), 1):
        second = float(timestamp_time[i][11]) * 36000 + float(timestamp_time[i][12]) * 3600 + \
                 float(timestamp_time[i][14]) * 600 + float(timestamp_time[i][15]) * 60 + \
                 float(timestamp_time[i][17]) * 10 + float(timestamp_time[i][18])
        # 32400是因为数据的时间是从9:00开始的，减去9小时的时间戳,但是其实也很难理解
        time_second_basic.append(second - 32400.0)
        time_second.append(second)
    #     time_second 就是秒级时间戳，time_second_basic是减去了32400秒的时间戳
    #  其实time_second 返回后根本没有使用。
    return np.array(time_second), np.array(time_second_basic)


def rise_ask(Ask1, timestamp_time_second, before_time):
    Ask1[Ask1 == 0] = mean(Ask1)
    rise_ratio = []
    index = np.where(timestamp_time_second >= before_time)[0][0]
    # open first before_time mins
    for i in range(0, index, 1):
        rise_ratio_ = round((Ask1[i] - Ask1[0]) * (1.0) / Ask1[0] * 100, 5)
        rise_ratio.append(rise_ratio_)
    for i in range(index, len(Ask1), 1):
        # print np.where(timestamp_time_second[:i] >= timestamp_time_second[i] - before_time)
        # print timestamp_time_second[i],timestamp_time_second[i] - before_time
        index_start = np.where(timestamp_time_second[:i] >= timestamp_time_second[i] - before_time)[0][0]
        rise_ratio_ = round((Ask1[i] - Ask1[index_start]) * (1.0) / Ask1[index_start] * 100, 5)
        rise_ratio.append(rise_ratio_)
    return np.array(rise_ratio)


def weight_pecentage(w1, w2, w3,
                     ask_quantity_1, ask_quantity_2, ask_quantity_3,
                     bid_quantity_1, bid_quantity_2, bid_quantity_3):
    """ 通过权重，在得到两个值，这两个值为啥要这么算？暂时还不理解。我们姑且当做两个指标来看待"""

    weight_ask = (w1 * ask_quantity_1 + w2 * ask_quantity_2 + w3 * ask_quantity_3)
    weight_bid = (w1 * bid_quantity_1 + w2 * bid_quantity_2 + w3 * bid_quantity_3)
    w_ab = weight_ask / weight_bid
    w_a_b = (weight_ask - weight_bid) / (weight_ask + weight_bid)
    return w_ab, w_a_b


def calculate_rise_ratios(ask, time_second_basic):
    before_times = [60.0 * i + j for i in range(6, 21) for j in [0, 30]]
    rise_ratios = []
    for idx, before_time in enumerate(before_times, start=1):
        rise_ratios.append(rise_ask(ask, time_second_basic, before_time))
    return rise_ratios


def calculate_weights(ask_quantity_1, ask_quantity_2, ask_quantity_3,
                      bid_quantity_1, bid_quantity_2, bid_quantity_3):
    weights_config = [
        (100, 0, 0), (0, 100, 0), (0, 0, 100), (90, 10, 0), (80, 20, 0), (70, 30, 0),
        (60, 40, 0), (50, 50, 0), (70, 20, 10), (50, 30, 20), (1, 1, 1), (10, 90, 1),
        (20, 80, 0), (30, 70, 0), (40, 60, 0), (10, 20, 70), (20, 30, 50)
    ]
    weights_results = []
    for idx, (w1, w2, w3) in enumerate(weights_config, start=100):
        weights_results.append(weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                             bid_quantity_1, bid_quantity_2, bid_quantity_3))
    return weights_results


def traded_label_one_second(time1, time2, time_second_basic,
                            bid_price_1, ask_price_1, traded_time,
                            rise_ratio_asks, W_ABs):
    # 初始化列表，用于存储交易标签、索引和各种比例
    global index
    traded = []
    index_ = []
    rise_ratio_seconds = [[] for _ in range(30)]  # 用于存储30个不同比例的列表
    w_divid_seconds = [[] for _ in range(30)]  # 用于存储30个不同W_AB的除数列表
    w_diff_seconds = [[] for _ in range(30)]  # 用于存储30个不同W_AB的差值列表

    # 计算初始索引
    if time1 == 0:
        index_one = np.where(time_second_basic <= 0)[0][-1]
    elif time1 == 14400:
        index_one = np.where(time_second_basic <= 14400)[0][-1]

    # 遍历时间区间
    for i in range(time1, time2, 1):
        # 根据当前时间找到对应的索引数组
        if i == 0 or i == 14400:
            index_array = np.where(time_second_basic <= i)[-1]
        else:
            index_array = np.where((time_second_basic < i + 1) & (time_second_basic >= i))[-1]

        # 如果找到索引
        if len(index_array) > 0:
            index = index_array[-1]
            # 根据时间位置添加到索引列表
            if i == time1:
                index_.append(index)
            if i == time2 - 1:
                index_.append(index)
            if i < 25200 - traded_time:
                index_min = np.where(time_second_basic <= i + traded_time)[0][-1]
                traded_min = ask_price_1[index:index_min]
                if bid_price_1[index] > min(traded_min):
                    traded.append(1)
                else:
                    traded.append(0)
            elif i >= 25200 - traded_time:
                if bid_price_1[index] > ask_price_1[-1]:
                    traded.append(1)
                else:
                    traded.append(0)

            # 添加比例和W_AB值到对应的列表
            # 遍历 rise_ratio_asks
            for item in rise_ratio_asks:
                rise_ratio_seconds.append(item[(index - index_one)])

            for item in W_ABs:
                w_divid_seconds.append(item[0][index_one + (index - index_one)])
                w_diff_seconds.append(item[1][index_one + (index - index_one)])



        else:
            # 如果没有找到索引，复制上一次的交易标签和比例
            if i < 25200 - traded_time:
                index_min = np.where(time_second_basic <= i + traded_time)[0][-1]
                traded_min = ask_price_1[index:index_min]
                if bid_price_1[index] > min(traded_min):
                    traded.append(1)
                else:
                    traded.append(0)
            elif i >= 25200 - traded_time:
                if bid_price_1[index] > ask_price_1[-1]:
                    traded.append(1)
                else:
                    traded.append(0)

            for j in range(30):
                rise_ratio_seconds[j].append(rise_ratio_seconds[j][-1])
                w_divid_seconds[j].append(w_divid_seconds[j][-1])
                w_diff_seconds[j].append(w_diff_seconds[j][-1])

    # 返回所有计算结果
    return (
        traded, index_, *rise_ratio_seconds, *w_divid_seconds, *w_diff_seconds
    )


def data(traded_time):
    # 获取订单簿数据
    timestamp, order_book_, bid_price_1, bid_price_2, bid_price_3, \
        bid_quantity_1, bid_quantity_2, bid_quantity_3, \
        ask_price_1, ask_price_2, ask_price_3, ask_quantity_1, \
        ask_quantity_2, ask_quantity_3 = order_book()

    _, time_second_basic = time_transform(timestamp)

    #  np.where(time_second_basic <= 0.0)[0][-1] 找到时间戳小于等于0的最后一个索引，
    #  然后ask1 是吧ask_price_1的数据从这个索引到最后的数据取出来
    ask1 = ask_price_1[np.where(time_second_basic <= 0.0)[0][-1]:]

    # 计算涨跌比率
    # 这里得到是一个map，key是rise_ratio_ask_N, value是一个risk_ask数组
    rise_ratios = calculate_rise_ratios(ask1, time_second_basic)

    # 计算权重
    # 这里得到是一个mpa ,key 是W_AB_NNN, value是一个元组，元组的第一个元素是w_ab, 第二个元素是w_a_b
    weights = calculate_weights(ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                bid_quantity_1, bid_quantity_2, bid_quantity_3)

    # 创建特征数据集，这里省略了具体实现，假设Feature_DataFrame_UP和Feature_DataFrame_DOWN存在

    data_2014_UP = traded_label_one_second(time1=0, time2=9000,
                                           time_second_basic=time_second_basic,
                                           bid_price_1=bid_price_1, ask_price_1=ask_price_1,
                                           traded_time=traded_time,
                                           rise_ratio_asks=rise_ratios,
                                           W_ABs=weights)

    data_2014_DOWN = traded_label_one_second(time1=14400, time2=25200,
                                             time_second_basic=time_second_basic,
                                             bid_price_1=bid_price_1, ask_price_1=ask_price_1,
                                             traded_time=traded_time,
                                             rise_ratio_asks=rise_ratios,
                                             W_ABs=weights)


traded_time = 600
data(traded_time)
