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
### 数学和统计方法
#### 计算有关整个数组或沿某一行或列的数据统计信息的一组数学函数可以作为数组类的方法访问。你可以通过调用数组实例方法或使用顶级Numpy函数来得到总和、均值和标准差等。
#### 在这里，我生成一些正态分布的随机数据并计算一些统计信息：
```python
>> arr = np.random.randn(5, 4)
>>
>> arr.mean()
>> np.mean(arr)
>> arr.sum()
```
#### mean和sum函数提供了一个关于轴向（行或列）的可选参数用于在给定轴上计算统计数值，生成低维的数组：
```python
>> arr.mean(axis=1)
>> arr.sum(axis=0)
```
#### 此处，arr.mean(1)表示“按列计算平均值”，其中arr.sum(0)表示“计算行的总和”。
#### 其他方法如cumsum，cumprod不会求总和，而是生成一个累计数组，如：
```python
>> arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
>> arr.cumsum()
# 打印输出：array([ 0, 1, 3, 6, 10, 15, 21, 28])
```
#### 我们发现，输出的数组的每一个值，是原数组对应位置前面所有数的总和。
#### 对于多维数组来讲，累计函数（如cumsum）返回大小相同的数组，但根据每个低维切片沿指定轴计算局部总和：
```python
>> arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
>> arr.cumsum(axis=0)
# 打印输出：([[ 0, 1, 2],
#            [ 3, 5, 7],
#            [ 9, 12, 15]])
>> arr.cumprod(axis=1)
# 打印输出：([[ 0, 0, 0],
#            [ 3, 12,60],
#            [ 6, 42, 336]])
```
#### 有关完整列表参阅下表，我们将在后面的章节中看到这些方法的许多示例：
| 方法 | 描述 |
| :--- | :--- |
|sum   |数组或沿轴的所有元素的总和，0长度数组总和为0|
|mean  |算数平均值；0长度数组的算数平均值为NaN|
|std, var|标准差和方差，分别带有可选的自由度调整（默认分母为n）|
|min, max|最小值和最大值|
|argmin,<br>argmax|最小元素和最大元素的下标|
|cumsum| 从0开始的元素的累计总和|
|cumprod| 从1开始元素的累计的乘积|
### 布尔数组方法
#### 布尔值在上述方法中被强制为1(True)和0(False)。因此，sum通常是计算布尔数组中True值的方法：
```python
>> arr = np.random.randn(100)
>> (arr > 0).sum() # 大于0的数量
# 打印输出：数组中大于0的元素的个数
```
#### 有两种附加方法，any和all，特别适用于布尔数组。any检测数组中的一个值或多个值是否为True，而all检测每个值是否为True：
```python
>> bools = np.array([False, False, True, False])
>> bools.any()
# 打印输出：True
>> bools.all()
# 打印输出：False
```
#### 这些方法也适用于非布尔数组，其中非0元素均为True。
### 排序
#### 与Python的内置列表一样，NumPy数组可以使用排序方法就地排序：
```python
>> arr = np.random.randn(6)
>> arr.sort()
# 打印输出：从小到大排序的数组
```
#### 你可以通过传递数组的轴向参数来排序，沿轴向对原多维数组在每一个位置进行排序：
```python
>> arr = np.random.randn(5, 3)
>> arr.sort(1)
# 打印输出：将每一行从小到大进行排序
```
#### 注意！顶级方法np.sort返回数组的排序副本，而不是就地修改数组。一个临时快速的计算数组的分位数（中位数、四分位数等）的方法是排序并选择特定排名的值：
```python
>> large_arr = np.random.randn(1000) # 生成一千个数
>> large_arr.sort()
>> large_arr[int(0.05 * len(large_arr))] # 5%分位数
# 打印输出：输出排名第5%的数字
```
#### 在Pandas中也可以找到与排序相关的几种其他类型的数据操作（例如，按一个或多个列对数据表进行排序）。
### Unique和其他集合操作
#### NumPy具有一维ndarry的一些基本集合操作。常用的一个是np.unnique，它返回数组中排序的唯一的元素（即若数组中有重复的元素，就只返回一个，且将他们排序后输出）：
```python
>> names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
>> np.unique(names)
# 打印输出：array(['Bob', 'Joe', 'Will']), dtype='<U4’)
>> ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
>> np.unique(ints)
# 打印输出：array([1, 2, 3, 4])
```
#### 将纯Python语言与np.unique进行对比：
```python
>> sorted(set(names))
# 打印输出：['Bob', 'Joe', 'Will']
```
#### 另一个函数np.in1d，将检测一个数组的值的成员是否存在于另一个数组，返回布尔数组：
```python
>> values = np.array([6, 0, 0, 3, 2, 5, 6])
>> np.in1d(values, [2, 3, 6])
# 打印输出：array([ True, False, False, True, True, False, True], dtype=boo;)
```
#### 下面列出了一些NumPy中的集合函数：
|方法|描述|
|:---|:---|
|unique(x)|计算并排序x中的唯一的元素|
|intersect1d(x, y)|在x, y中计算并分类出相同的元素|
|union1d(x, y)|计算元素的排序联合|
|in1d(x, y)|计算返回一个布尔数组，指明每个x元素是否包含y元素|
|setdiff1d(x, y)|x - y，元素在x但不在y中|
|setxor1d(x, y)| 设置对称差集，即元素在两个中的任意一个但不在全部中|


### 4.4 数组的文件输入与输出
#### NumPy能够以文本或二进制格式保存和加载数据到磁盘。在本节中，我只讨论Numpy的内置二进制格式，因为绝大多数用户会更喜欢Pandas和其他工具来加载文本或表格数据（有关详细信息，请参阅第6章）。
#### np.save和np.load是两个主力工作函数，用来高效的保存和加载磁盘上的数组数据。默认情况下，数组以未压缩的原始二进制格式保存，文件扩展名为'.npy'。
```python
>> arr = np.arange(10)
>> np.save('some_array', arr)
```
#### 如果文件路径尚未以.npy结尾，则将附加扩展名。然后，磁盘上的数组可以用np.load加载：
```python
>> np.load('some_array')
```
#### 使用np.savez将多个数组保存在未压缩的文档中，并将数组作为关键字参数传递：
```python
>> np.savez('array_archive.npz', a=arr, b=arr)
```
#### 当加载一个.npz文件时，你可以返回一个类似dict的对象，该对象会延迟加载各个数组：
```python
>> arch = np.load('array_archive.npz')
>> arch['b']
# 打印输出：array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
#### 如果数据压缩良好，你可能希望使用numpy.savez_compressed来代替：
```python
>> np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)
```
### 4.5 线性代数
####
