# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm

def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            # 对于类别x_name中，同一类别的y值求和，和求次数
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

def target_mean_v3(data, y_name, x_name):
    t = pd.concat([data.groupby(x_name)[y_name].transform('count'),data.groupby(x_name)[y_name].transform('sum')],axis=1)
    t.columns = ['count', 'total']
    result = (t.total-data[y_name])/(t['count']-1)
    return result.values


def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    a = time.time()
    result_1 = target_mean_v2(data, 'y', 'x')
    print('最开始的第二种方法        ', time.time() - a)

    a = time.time()
    result_2 = target_mean_v3(data, 'y', 'x')
    print('通过transform写的方法   ', time.time() - a)

    a = time.time()
    result_3 = tm.target_mean_v3(data, 'y', 'x')
    print('王然老师的方法           ', time.time() - a)

    # 可以改的思路：unordered_map, 数据类型:int, float; 该循环 for row from 0 <= row < nrow by 1:
    a = time.time()
    result_4_type_change, value_dict, count_dict = tm.target_mean_v4(data, 'y', 'x')
    # print(result_4, value_dict, count_dict)
    print('改写数据类型的方法        ', time.time() - a)

    a = time.time()
    result_4_unordered_map, value_dict, count_dict = tm.target_mean_v4_unordered_map(data, 'y', 'x')
    print('修改成unordered_map     ', time.time() - a)

    print(np.linalg.norm(result_2 - result_1))
    print(np.linalg.norm(result_3 - result_1))
    print(np.linalg.norm(result_4_type_change - result_1))
    print(np.linalg.norm(result_4_unordered_map - result_1))


if __name__ == '__main__':
    main()
