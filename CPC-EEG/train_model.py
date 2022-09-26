'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from data_utils1 import SortedNumberGenerator
import keras as keras
from keras import backend as K
# import tensorflow as tf
# #tf.compat.v1.disable_eager_execution()
# import torch
# from torchvision import transforms
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import scipy.io as scio
# import scipy.ndimage
# from PIL import Image
# import scipy
# import sys
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from skimage import io,transform #skimage模块下的io transform(图像的形变与缩放)模块
# import glob  #glob 文件通配符模块
# import os    #os 处理文件和目录的模块
# from sklearn.utils import shuffle
# import scipy.io as sio
# from os import listdir
from os.path import isfile, join

# import visualize
# import numpy as np
# import scipy.misc

def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings定义将图像映射到嵌入的网络
    卷积核大小：3*3，卷积核个数：64，步长：2，使用线性激活函数
    使用批量标准化对数据进行处理，可以加速训练过程，具有轻微正则化的效果，降低dropout的使用
    由于输入值的分布不同，即输入特征值的scale差异较大，会产生一些偏离较大的差异值，
    会影响后层的训练，所以需要用BN对输入值进行标准化，降低scale的差异到同一个范围内
    ReLU激活函数是一个非线性激活函数，将负值置为0，其余值保持不变
    使用LeakyReLU激活函数，与ReLU不同的是，给所有负值赋给一个非零斜率，这样保留了一些
    负轴的值，使得负轴的信息不会全部丢失'''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    ''' 使用展平层对数据进行展平 '''
    x = keras.layers.Flatten()(x)
    ''' 使用全连接层，神经元个数为256，使用一个线性的激活函数进行激活 '''
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)
    print('x:',x.shape)
    ''' 返回经过网络处理后的x '''
    return x
    #print(x)

def network_autoregressive(x):
    ''' Define the network that integrates information along the sequence
     定义沿着序列整合信息的网络'''
    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    # 使用GRU来整合之前的信息（整合成图中的Ct），只返回最后一个时间步的输出（图中t时刻）
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    return x
#网络的预测结果，predict_terms：应该是预测未来几个时期的Xt+k，如预测未来4个时期，predict_terms=4
#context：自回归得到的上下文信息
def network_prediction(context, code_size, predict_terms):
    ''' Define the network mapping context to multiple embeddings
    定义网络映射上下文到多个嵌入'''
    outputs = []
    for i in range(predict_terms):
        #使用全连接层获取输出结果，将结果放进outputs中进行保存
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))
    if len(outputs) == 1:
        #K.expand_dims(x, axis=1)，用于扩展数组的形状，
        # axis=1代表扩展列，axis=-1代表扩展最后一个参数
        #（维度扩展）如果输出只包含一个参数，那么就对输出结果进行行的扩展
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        #对输出按行进行堆叠，把多个张量堆叠成一个张量
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)
    #返回将处理好的output
    return output


class CPCLayer(keras.layers.Layer):
    ''' 计算真实值和预测值embedding之间的点积，比较相似度（判断模型好坏）'''
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)
    def call(self, inputs):
        # 计算向量间的点积
        preds, y_encoded = inputs
        #mean：求均值，axis=0，压缩行，对各列求均值，返回1*n矩阵，
        # axis=1，压缩列，对行求均值，返回n*1矩阵
        dot_product = K.mean(y_encoded * preds, axis=-1)#（None，4，128）——>（None，4）
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)
        # 在时间维度上求平均（None，4）——>（None，1）
        # Keras损失函数采用概率
        dot_product_probs = K.sigmoid(dot_product)
        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model
     结合了编码和自回归模型的CPC网络'''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)#当phase=1，则为train，当phase=0，则为test

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    #调用上面定义的network_encoder函数，进行数据的编码操作，将图像编码为code_size维的向量
    encoder_output = network_encoder(encoder_input, code_size)
    #保存编码模型
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    #layers.Input：用于构建网络的第一层——输入层，这层会告诉我们输入的尺寸是什么
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    #TimeDistributed共享权重信息，可以实现从二维到三维的过渡
    #对每个时间步输入的图像进行编码，输出的是（time_step, code_size）
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    #定义一个context变量，保存自回归模型整合的之前的所有信息，即最后一个时间步的输出
    context = network_autoregressive(x_encoded)
    #保存模型预测的结果
    preds = network_prediction(context, code_size, predict_terms)
    print('1',preds.shape)
    #scio.savemat(r'D:\contrastive-predictive-coding-master\result_encoder.mat',{'matrix_1':preds})
    #toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    #img = preds
    #pic = toPIL(img)
    #pic.save('train.jpg')
    #preds.save("1.jpg")
    #imsave('img1.jpg', preds)

    #img = Image.open('./lena.jpg')
    #ToTensor_transform = transforms.Compose([transforms.ToTensor()])
    #pic = ToTensor_transform(preds)
    #preds = to_pil_image(pic, mode=None)
    #plt.imshow(preds)
    #plt.savefig('./jieguo.png')

    #preds=np.random.random((None,4,128))
    #scipy.misc.imsave('lll.jpg',preds)
    #pic = preds
    #toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    #img = torch.randn(3, 128, 64)
    #pic = toPIL(img)
    #pic.save('random.jpg')
    #对真实值进行编码，用来和自回归模型的预测进行对比
    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model编译模型
    cpc_model.compile(
        #使用Adam优化器进行优化，Adam：自适应矩估计，梯度下降算法的一种变形，
        # 动态的利用梯度的一阶矩估计和二阶矩估计动态的调整每个参数的学习率，
        # 可控制学习速度，经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        #损失函数是二分类的交叉熵
        loss='binary_crossentropy',
        #指标：二分类的精度
        metrics=['binary_accuracy']
    )
    cpc_model.summary()

    return cpc_model
    #get_11rd_layer_output = K.function([model.layers[0].input],
                                    #[model.layers[10].output])
    #layer_output = get_11rd_layer_output([train_data])[0]
    #print(layer_output[0].shape)

def train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=64, color=False):

    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)
    # Prepares the model(image_size, image_size, 3)高，宽，深度（RGB图像为3）
    model = network_cpc(image_shape=(image_size, image_size, 3), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    # Callbacks回调函数
    #ReduceLROnPlateau：当评价指标不再提升时，减少学习效率
    #monitor：被检测的量；factor：每次减少的学习因子，lr=lr*factor
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
    # Trains the model
    history=model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )
    import pandas as pd

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
    #model.summary
    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'cpc.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder.h5'))




if __name__ == "__main__":

    train_model(
        epochs=100,
        batch_size=16,
        output_dir='models/gai11',
        code_size=128,
        lr=1e-3,
        terms=4,
        predict_terms=4,
        image_size=64,
        color=True,
        #validation_steps=8

    )

