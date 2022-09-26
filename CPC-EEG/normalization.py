import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
# 可读模式读入文本内容
f = open(r"D:/contrastive-predictive-coding-master/result_xyzhuan.txt", 'r')
# 将文本内容以多行方式一次性读入
lines = f.readlines()
res = []
for i in lines:
	# 根据每行各个列之间的空格，返回一个每行3列的列表
    line = i.split(' ')
    res.append(line)    # res 得到一个由若干行组成的列表
#print(res)
# for i in range(len(res)):
# 	# 将 "+" 替换掉
#     res[i][0] = res[i][0].replace('+','')
#     # 对第 i 行，索引为 1 的列，字符串切片，重新赋值给第 i 行第 1 列
#     res[i][1] = res[i][1][2:]
#     res[i][2] = res[i][2][2:]
# # 将 res 转换为 numpy 数组类型，取所有行，不取最后一列，也就是剔除掉最后一列
data = np.array(res)[:,:]
#print(data)
# #data = str.replace('\t(Tab)','\t,',30000)
# data = str.strip(" ")
# #print(data)
# lines = f.readlines()
# for line in lines:
#     line = line.strip("\n")
#     line = line.split(" ")
#     line = [float(x) for x in line]

minMax = MinMaxScaler(data)  # Normalize data
data = np.hstack((minMax.fit_transform(data[:,1:]), data[:, 0].reshape(data.shape[0], 1)))
#data = minMax.fit_transform(data[:, :])
print(data)
data = data.astype(float)
scipy.io.savemat('gyh.mat', {'data': data})
