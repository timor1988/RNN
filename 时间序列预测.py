# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import style
import pandas as pd

# '''读入原始数据并转为list'''

os.listdir()

data = pd.read_csv('AirPassengers.csv')

data

data = data.iloc[:,1].tolist()

'''自定义数据尺度缩放函数'''
def data_processing(raw_data,scale=True):
    if scale == True:
        return (raw_data-np.mean(raw_data))/np.std(raw_data)#标准化
    else:
        return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))#极差规格化


# 数据观察部分：
#
# 　　这一部分，我们需要初步观察到原数据的一些基本特性，以便确定之后的一些参数，如LSTM单元内一个时间步内的递归次数：

data

# +
'''观察数据'''

'''设置绘图风格'''
style.use('ggplot')

plt.plot(data)
# -

# LSTM基本参数设置：
#
# 　　这里我们需要设置的参数有隐层层数，因为数据集比较简单，我们设置为1；隐层神经元个数，这里我随意设置为40个；时间步中递归次数，这里根据上面观察的结论，设置为12；训练轮数，这里也是随意设置的不宜过少，2000；训练批尺寸，这里随意设置为20，表示每次循环从训练集中抽出20组序列样本进行训练：

'''设置隐层神经元个数'''
HIDDEN_SIZE = 40
'''设置隐层层数'''
NUM_LAYERS = 1
'''设置一个时间步中折叠的递归步数'''
TIMESTEPS = 12
'''设置训练轮数'''
TRAINING_STEPS = 2000
'''设置训练批尺寸'''
BATCH_SIZE = 20

# 生成训练集数据：
#
# 　　这里为了将原始的单变量时序数据处理成LSTM可以接受的数据类型（有X输入，有真实标签Y），我们通过自编函数，将原数据（144个）从第一个开始，依次采样长度为12的连续序列作为一个时间步内部的输入序列X，并采样其之后一期的数据作为一个Y，具体过程如下：|

'''样本数据生成函数'''
def generate_data(seq):
    X = []#初始化输入序列X
    Y= []#初始化输出序列Y
    '''生成连贯的时间序列类型样本集，每一个X内的一行对应指定步长的输入序列，Y内的每一行对应比X滞后一期的目标数值'''
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])#从输入序列第一期出发，等步长连续不间断采样
        Y.append([seq[i + TIMESTEPS]])#对应每个X序列的滞后一期序列值
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ## 构造LSTM模型主体：



# 定义LSTM模型
class MyLstm:
    
    #1 .LSTMCell组件，该组件将在训练过程中不但更新参数
    def lstm(self):
        lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE,state_is_tuple=True)
        return lstm_cell
    def lstm_model(self,x,y):
        print(x.shape)  # (?, 1, 12)
        # 只使用一层
        cell = rnn.MultiRNNCell([self.lstm() for _ in range(NUM_LAYERS)])
        '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
        output,_ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32) #  shape 为 [batch_size,序列长度,隐藏层大小]
        print(output.get_shape())  # (?, 1, 40)
        '''根据预定义的每层神经元个数生成隐层每个单元'''
        output = tf.reshape(output,[-1,HIDDEN_SIZE])
        print(output.get_shape()) # (?, 40)
        '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
        predictions = tf.contrib.layers.fully_connected(output,1,None)
        print(predictions.get_shape())
        '''统一预测值与真实值的形状'''
        labels = tf.reshape(y, [-1])
        predictions = tf.reshape(predictions, [-1])

        '''定义损失函数，这里为正常的均方误差'''
        loss = tf.losses.mean_squared_error(predictions, labels)

        '''定义优化器各参数'''
        train_op = tf.contrib.layers.optimize_loss(loss,
                                                   tf.contrib.framework.get_global_step(),
                                                   optimizer='Adagrad',
                                                   learning_rate=0.6)
        '''返回预测值、损失函数及优化器'''
        return predictions, loss, train_op



# +
a = MyLstm()
'''载入tf中仿sklearn训练方式的模块'''
learn = tf.contrib.learn

'''初始化我们的LSTM模型，并保存到工作目录下以方便进行增量学习'''
regressor = SKCompat(learn.Estimator(model_fn=a.lstm_model, model_dir='Model/model_2'))

# +
'''对原数据进行尺度缩放'''
data = data_processing(data)

'''将所有样本来作为训练样本'''
train_X, train_y = generate_data(data)

'''将所有样本作为测试样本'''
test_X, test_y = generate_data(data)

'''以仿sklearn的形式训练模型，这里指定了训练批尺寸和训练轮数'''
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# +
'''利用已训练好的LSTM模型，来生成对应测试集的所有预测值'''
predicted = np.array([pred for pred in regressor.predict(test_X)])

'''绘制反标准化之前的真实值与预测值对比图'''
plt.plot(predicted, label='预测值')
plt.plot(test_y, label='真实值')
plt.title('反标准化之前')
plt.legend()
plt.show()
# -

# 可以看到，预测值与真实值非常的吻合，但这并不是我们需要的形式，我们需要的是反标准化后的真实数值，下面进行相关操作；

# +
'''自定义反标准化函数'''
def scale_inv(raw_data,scale=True):
    '''读入原始数据并转为list'''
    

    data = pd.read_csv('AirPassengers.csv')

    data = data.iloc[:, 1].tolist()

    if scale == True:
        return raw_data*np.std(data)+np.mean(data)
    else:
        return raw_data*(np.max(data)-np.min(data))+np.min(data)


'''绘制反标准化之前的真实值与预测值对比图'''
plt.figure()
plt.plot(scale_inv(predicted), label='预测值')
plt.plot(scale_inv(test_y), label='真实值')
plt.title('反标准化之后')
plt.legend()
plt.show()
# -


