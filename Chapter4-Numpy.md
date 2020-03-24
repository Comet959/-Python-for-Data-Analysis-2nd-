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
>> arr = np.arange(10) # 生成0到9的十个数字
>> arr_slice = arr[5: 8] # 切片操作
>> arr_slice[1] = 12345 # 给拷贝后的数据赋值
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
#### 作为一个简单的示例，假设我们希望在一个常值网格中计算$\sqrt{x^2+y^2}$。np.meshgrid函数被传入用两个一维数组，并生成两个二维数组，对应于两个数组中的(x, y)：
```python
>> points = np.arange(-5, 5, 0.01) # 生成1000个等间距的点
>> xs, ys = np.meshgrid(points, points) # 生成两个二维网格
>> z = np.sqrt(xs ** 2, ys ** 2) # 计算函数表达式
```
#### 我们利用matplotlib（后文将细述）创建该二维数组的可视化效果：
```python
>> import matplotlib.pyplot as plt
>>
>> plt.imshow(z, cmap=plt.cm.gray)
>> plt.colorbar()
# 显示输出：<matplotlib.colorbar.Colorbar at 0x7f715e3fa630>
>> plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# 显示输出：<matplotlib.text.Text at 0x7f715d2de748>
```
#### 在这里，我使用matplotlib函数imshow从二维数组的函数值创建图像绘图。
![avatar](/img/3.png)
### 将条件逻辑表示为数组操作
#### np.where函数是一个接受三个参数的函数，他可以将循环逻辑简化为数组操作，我们以例子说明，假定我们有一个布尔数组和两个数值数组：
```python
>> xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
>> yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
>> cond = np.array([True, False, True, True, False])
```
#### 假定我们现在希望当cond为True时，就取xarr中相应的元素，否则取yarr中相应的元素。执行此项功能的理解类似于：
```python
>> result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
>> print(result)
# 输出：[1.1000000000000001, 2.2000000000000002, 1.3, 1.3999999999999999, 2.5]
```
#### 这存在多个问题。首先，对于大型数组来说，它不是很快（因为所有工作都在解释型Python代码中完成）。其次，它不适用于多维数组，而有了np.where，你可以非常简洁的完成该功能：
```python
>> result = np.where(cond, xarr, yarr)
>> print(result)
# 输出：array([ 1.1, 2.2, 1.3, 1.4, 2.5])
```
#### np.where的第二个和第三个参数不需要数组，其中一个或两个可以是数值标量。在数据分析中一个典型的用处是基于另一个数组生成新的值数组。假设你有一个随机生成的数据的矩阵，并且希望将所有的正值替换为2，所有的负值替换为-2。用np.where将非常容易完成这件事：
```python
>> arr = np.random.randn(4, 4)
>> arr > 0
# 打印将输出一个布尔矩阵
>> np.where(arr > 0, 2, -2)
# 打印将输出一个数组，由2和-2构成，在原数组大于0的位置是2，小于0的位置是-2
```
#### 使用np.where时，可以结合标量和数组。例如，我可以将arr中的所有正值替换为常量2，如下所示：
```python
>> np.where(arr > 0, 2, arr)
# 打印将输出一个矩阵，在原矩阵大于0的地方都变为2
```
#### 传递给np.where的数组可以不仅仅是大小相等的数组或标量。
### 数学和统计方法：


