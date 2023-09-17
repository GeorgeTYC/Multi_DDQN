import matplotlib.pyplot as plt
import numpy as np

import csv

mean1 = np.zeros(5000)
dt1 = np.zeros(5000)
rows = np.zeros(5000)
# 打开 CSV 文件
with open('E:\projects\Multi-DDQN\Multi-DDQN-v1\DDQN\logs_v2.csv', 'r') as file:
    # 创建 CSV 读取器
    reader = csv.reader(file)
    count = 0
    for row in reader:
        if count >= 1:
            # print(count)
            rows[count-1] = count-1
            mean1[count-1] = float(row[2])
            dt1[count-1] = float(row[3])
        count = count + 1

plt.plot(mean1)
plt.show()
