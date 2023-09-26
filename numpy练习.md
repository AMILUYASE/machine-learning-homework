***\*练习一\****

#### **1.** ***\*导入numpy库\****

***\*import numpy as np\****

 

#### ***\*2.建立一个\****[一维数组](https://so.csdn.net/so/search?q=一维数组&spm=1001.2101.3001.7020)***\*a 初始化为[4,5,6], (1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）\****

***\*a = np.array([4,5,6])\****

***\*type(a) # numpy.ndarray\****

***\*a.shape # (3,)\****

***\*a[0] # 4\****

 

#### ***\*3.建立一个\****[二维数组](https://so.csdn.net/so/search?q=二维数组&spm=1001.2101.3001.7020) ***\*b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）\****

***\*b = np.array([[4,5,6],[1,2,3]])\****

***\*b.shape # (2, 3)\****

***\*b[0,0] # 4\****

***\*b[0,1] # 5\**** 

***\*b[1,1] # 2\****

 

#### **4.** ***\*(1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5; (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,大小为 3x2.\****

a = np.zeros((3,3),dtype=int)

b = np.ones((4,5),dtype=int)

c = np.identity(4)

d = np.random.randn(3,2)

 

#### ***\*5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1)打印a; (2)输出 下标为(2,3),(0,0) 这两个数组元素的值\****

a = np.arange(1,13).reshape(3,4)

a

a[2,3]

a[0,0]

 

#### ***\*6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）(1),输出b;(2) 输出b 的（0,0）这个元素的值\****

 b = a[0:2,1:3]

 b

 b[0,0]

 

#### ***\*7. 把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）\****

 c = a[1:3,:]

 c

 c[0][-1]

 

#### **8.** ***\*建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）\****

a = np.array([[1,2],[3,4],[5,6]])

print(a[[0,1,2],[0,1,0]])

 

\9. 建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1) 

a = np.arange(1,13).reshape(4,3)

b = np.array([0,2,0,1])

print(a[[np.arange(4),b]]) # [ 1  6  7 11]

 

10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）

a[[np.arange(4),b]] += 10

a[[np.arange(4),b]] # array([21, 26, 27, 31])

 

\11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型

x = np.array([1,2])

x.dtype # dtype('int32')

 

12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型

x =np.array([1.0,2.0])

x.dtype # dtype('float64')

 

13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

y = np.array([[5, 6], [7, 8]], dtype=np.float64)

x + y

np.add(x,y)

 

 

\14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)

x-y

np.subtract(x,y)

 

\15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有 np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。

x * y # 两个矩阵对应位置元素相乘

 

array([[ 5., 12.],

​    [21., 32.]])

 

np.multiply(x,y) # 两个矩阵对应位置元素相乘

 

array([[ 5., 12.],

​    [21., 32.]])

 

np.dot(x,y) # 矩阵相乘

 

array([[19., 22.],

​    [43., 50.]])

 

\16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())

x / y

 

array([[0.2    , 0.33333333],

​    [0.42857143, 0.5    ]])

 

np.divide(x,y)

 

array([[0.2    , 0.33333333],

​    [0.42857143, 0.5    ]])

 

\17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )

np.sqrt(x)

 

array([[1.     , 1.41421356],

​    [1.73205081, 2.     ]])

 

18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))

print(x.dot(y))

 

 

print(np.dot(x,y))

 

19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)): (2)print(np.sum(x，axis =0 )); (3)print(np.sum(x,axis = 1))

print(np.sum(x)) # 10

print(np.sum(x,axis=0)) # [4. 6.] 两列之和

print(np.sum(x,axis=1)) # [3. 7.] 两行之和

 

20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）

print(np.mean(x))

print(np.mean(x,axis=0))

print(np.mean(x,axis=1))

 

21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）

x.T

print(x.T)

 

22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）

np.exp(x) # 求e的x次方的值

 

array([[ 2.71828183,  7.3890561 ],

​    [20.08553692, 54.59815003]])

 

23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))

print(np.argmax(x))

print(np.argmax(x,axis=0))

print(np.argmax(x,axis=1))

 

24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到 matplotlib.pyplot 库）

import matplotlib.pyplot as plt

x = np.arange(0,100,0.1)

y = x * x

plt.figure(figsize=(6,6))  # 创建画布，并指定画布大小

plt.plot(x,y)  # 在画布上画图

plt.show()  # 展示画图结果

 

 

25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)

x = np.arange(0,3*np.pi,0.1)

y1 = np.sin(x)

y2 = np.cos(x)

plt.figure(figsize=(10,6))

plt.plot(x,y1,color='Red')

plt.plot(x,y2,color='Blue')

plt.legend(['Sin','Cos'])  # 给两条线做标记

plt.show()

 

 



 

 