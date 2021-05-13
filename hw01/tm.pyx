# distutils: language=c++
import numpy as np
cimport numpy as np

import cython
cimport libcpp
from libcpp.unordered_map cimport unordered_map

def hello():
    print("hello")


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

# 王然老师的方法
cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)

# 用unordered_map
cpdef target_mean_v4_unordered_map(data, y_name, x_name):
    cdef:
        long nrow = data.shape[0]
        np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
        np.ndarray[int] y = np.asfortranarray(data[y_name], dtype=np.int32)
        np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)
        unordered_map[int, int] value_dict
        unordered_map[int, int] count_dict

    target_mean_v4_impl_unordered_map(result, y, x, nrow, value_dict, count_dict)
    return result, value_dict, count_dict

cdef void target_mean_v4_impl_unordered_map(double[:] result, int[:] y, int[:] x, const long nrow,unordered_map[int, int] value_dict, unordered_map[int, int] count_dict):
    cdef long i
    for i in range(nrow):
    #for i from 0 <= i < nrow by 1:
        if (value_dict.find(x[i]) != value_dict.end()):
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
        else:

            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
    i=0
    for i in range(nrow):#for i from 0 <= i < nrow by 1:
        result[i] = float(value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)

# 改了下数据类型
cpdef target_mean_v4(data, y_name, x_name):
    cdef:
        long nrow = data.shape[0]
        np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
        np.ndarray[int] y = np.asfortranarray(data[y_name], dtype=np.int32)
        np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)
        dict value_dict = dict()
        dict count_dict = dict()
    target_mean_v4_impl(result, y, x, nrow, value_dict, count_dict)
    return result, value_dict, count_dict

cdef void target_mean_v4_impl(double[:] result, int[:] y, int[:] x, const long nrow, dict value_dict, dict count_dict):
    cdef:
        long i
    for i in range(nrow):
    #for i from 0 <= i < nrow by 1:
        if x[i] in value_dict:
            value_dict[x[i]] = value_dict[x[i]]+ y[i]
            count_dict[x[i]] = count_dict[x[i]] + 1
        else:
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
    i=0
    for i in range(nrow):#for i from 0 <= i < nrow by 1:
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)
