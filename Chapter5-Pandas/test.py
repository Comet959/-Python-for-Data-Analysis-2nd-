import numpy as np
data = np.array([1, 2, 3, [2, 3]])  # 对象类型的object

copy1 = data
copy2 = data.copy()

data[3][0] = 10  # 修改值

print("copy1: ", copy1)
print("copy2", copy2)
