# Numpy
##### 作者：PureFFFmennory

### 对象类型：ndarry

### 4.1 常用Method：
```python
>> np.array()
>> np.arange()
>> np.ones()
>> np.zeros()
>> np.eye()
# 上述方法均有dtype = np.float64或np.int32类型参数，更多数据类型查看书内P109
```

```python
>> np.random.randn()
>> ndarry.shape() # 查看数组形状
>> ndarry.dtype() # 查看数组数据类型
>> ndarry.ndim() # 查看数组维度
>> ndarry.ashape(np.float64) # 转化数组的类型
```
#### ndarry类型数据可以进行切片操作，如：
```python
>> ndarry[5: 8]
```
#### 切片是浅拷贝，用C的话语来讲，切片在赋值时只拷贝了地址，而不是直接复制数据，若修改拷贝后的数据，那么原来的数据也将发生改变，因为他们都只想同一块内存区域。如：
```python
arr = np.arange(10) # 生成0到9的十个数字
arr_slice = arr[5: 8] # 切片操作
arr_slice[1] = 12345 # 给拷贝后的数据赋值
```
#### 结果将导致原数组的arr[5+1]的值也将变成12345，这是由于浅拷贝导致的后果。
#### 如果不想使用浅拷贝，可以使用copy方法，如：
```python
>> arr[5, 8].copy() # 深拷贝
```
#### 与切片不同的是，花式索引总是将数据另复制到新数组当中，而不是浅拷贝。
### 数组的转置：
#### ndarry数组具有转置方法：
```python
>> ndarry.T() # 将数组转置
```
#### 利用转置，我们可以使用dot方法实现矩阵的内部乘法如：
```python
>> arr = np.random.randn(6, 3) # 生成6行3列的正太分布数据矩阵
>> np.dot(arr.T, arr) # 计算矩阵arr的平方
```
### 4.2 快速使用元素数组函数
#### 简单来讲，就是使用Numpy对ndarry数组的构建，让他可以快速使用矩阵计算，如对矩阵内每个元素作加减乘除平方开方运算。如：
```python
>> arr = np.arange(10)
>> np.squt(arr) # 对矩阵每个元素求平方根
>> np.exp(arr) # 令矩阵每个元素作为以自然对数为底的指数
```
#### 上面是一些简单的一元函数，Numpy也提供了操作两个数组的方法，如add，maximum，并返回一个数组作为结果。如：
```python
>> x = np.random.randn(8) # 随机生成8个正态分布的数
>> y = np.random.randn(8)
>>
>> np.maximum(x, y) # 求数组x， y的对应元素的最大值并返回一个最大值数组
```
#### 在这里，numpy.maximum计算了x和y中元素的逐元素的最大值。
#### 尽管不常见，但ufunc(Numpy内置函数对象)可以返回多个数组。modf便是一个示例，它返回浮点数组的分数与整数部分：
```python
>> arr = np.random.randn(7) * 5
>> remainder, whole_part = np.modf(arr) # remainder得到小数部分， whole_part得到整数部分
```
#### 下面是一些”单目运算“函数：
| 函数    |  描述 |
|:---     |  :--- |
|abs, fbs | 计算整数，浮点数或复数的逐元素的绝对值|
|sqrt     | 计算每个元素的平方根|
|square   | 计算每个元素的平方（等价于arr ** 2） |
|exp      | 计算每个元素的指数e<sup>x</sup> |
|log, log10,<br>log2, log1p| 计算自然对数（以e为底），以10为底的对数，<br>以2为底的对数和log(1+x) |
|sign   | 计算每个元素的符号：1（正），0（零），-1（负）|
|ceil    | 每个元素向上取整|
|floor    | 每个元素向下取整 |
|rint     | 将元素舍入到最接近的整数，保留dtype|
|modf     | 讲数组的小数与整数部分作为单独的数组返回|
|isnan    | 返回布尔数组，指示每个值是否为NaN（不是数字）|
|isfinite, isinf| 返回布尔数组，指示每个元素是有限还是无限|
|cos, cosh, <br>sin, sinh,<br>tan, tanh| 计算三角函数和双曲函数|
|arccos, arccosh,<br>arcsin, arcsinh,<br>arctan, arctanh|反三角函数|
|logical_not| 原文：Compute truth value of not x element-wise<br>(equivalent to ~arr)|
#### 还有一些”双目运算“函数：
| 函数 | 描述 |
|:--- | :--- |
|add | 在数组中添加相应的元素 |
|subtract | 从第一个数组中减去第二个数组的元素|
|multiply | 对数组元素做乘法|
|divide,<br>floor_divide | 除法、取整除法 |
|power | 将第一个数组中的元素提升到第二个数组中提示的幂 |
|maximum, famx| 元素最大值；fman忽略NaN |
|minimum, fmin| 元素最小值，fmin忽略NaN |
|mod | 取模运算 |
|copysign | 将第二个参数中值的符号复制到第一个参数的值 |
|greater, greater_equal,<br>less, lee_equal,<br>equal, not_equal| 执行元素比较，生成布尔数组（>, >=, <, <=, =, !=） |
|logical_and,<br>logical_or,<br>logical_xor| 计算逻辑操作的按元素求真值（与，或，异或）|

### 4.3 使用向量计算代替数组
#### 使用Numpy数组，可以将许多类型的数据处理任务表达为简洁的数组表达式，从而免除了复杂的循环逻辑。该种数组表达式替换循环的做法通常被称为向量运算。通常，向量运算操作的速度比纯Python等效运算要快一到两（或更多）个数量级，在处理庞大的数据时，他们的差异是显著的。
