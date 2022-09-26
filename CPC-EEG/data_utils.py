''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import io,transform #skimage模块下的io transform(图像的形变与缩放)模块
import glob  #glob 文件通配符模块
import os    #os 处理文件和目录的模块
from sklearn.utils import shuffle
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np   #多维数据处理模块
import time

class EEGIMAGE(object):

    ''' Provides a convenient interface to manipulate MNIST data
     提供一个方便的界面来操作MNIST数据'''

    def __init__(self):
        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

        # Load Lena image to memory
        self.lena = Image.open('resources/lena.jpg')
      
    # 读取图片+数据处理
    def load_dataset(self):
        def read_img(path):
            w = 64
            h = 64
            c = 3
            # os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
            # os.path.isdir(path)判断path是否是目录
            # b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
            cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
            print(os.listdir(path))
            print(cate)
            imgs = []
            labels = []
            for idx, folder in enumerate(cate):
                # glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表
                for im in glob.glob(folder + '/*.png'):
                    # 输出读取的图片的名称
                    #print('reading the images:%s' % (im))
                    # io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
                    # 读取的图片
                    img = io.imread(im)
                    # skimage.transform.resize(image, output_shape)改变图片的尺寸
                    img = transform.resize(img, (w, h))
                    # 将读取的图片数据加载到imgs[]列表中
                    imgs.append(img)
                    # 将图片的label加载到labels[]中，与上方的imgs索引对应
                    labels.append(idx)
            #print('labels_0:',labels)
            # 将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
            return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

        data, label = read_img('new_3/')
        data, label = shuffle(data, label)
        def normalization(data):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        

        # 调用读取图片的函数，得到图片和labels的数据集
        #for file in dirs:
        #    data, label = read_img(join(path, file) + "/")
        #print(datas.shape)
        #print('1 labels.shape:', labels.shape)
        #data, labels = shuffle(datas, labels)
        data = normalization(data)#data / 255
        # y = np.array(labels)
        # print(X.shape)
        # labels = to_categorical(labels)


        # %%
        # 创建训练与测试集
        X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, test_size=0.2)
        #print(X_train, X_test, y_train, y_test)
        # We can now download and read the training and test set images and labels
        # 我们现在可以下载并读训练和测试集的图像和标签。.
    #    X_train = load_mnist_images('resources/train-images-idx3-ubyte.gz')
    #    y_train = load_mnist_labels('resources/train-labels-idx1-ubyte.gz')
    #    X_test = load_mnist_images('resources/t10k-images-idx3-ubyte.gz')
    #    y_test = load_mnist_labels('resources/t10k-labels-idx1-ubyte.gz')
        #print("X_train",X_train.shape)
        #print("y_train", y_train.shape)
        # We reserve the last 10000 training examples for validation
        # 我们保留最后10000个训练示例以供验证。.
        #X_train, X_val = X_train[:-10], X_train[-10:]
        #y_train, y_val = y_train[:-10], y_train[-10:]

        # We just return all the arrays in order, as expected in main().我们只是按照main（）中的预期顺序返回所有数组
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_test, y_test, X_test, y_test

    def process_batch(self, batch, batch_size, image_size=64, color=False, rescale=True):

        # Resize from 28x28 to 64x64从28x28调整到64x64
        #if image_size == 64:
        #    batch_resized = []
        #    for i in range(batch.shape[0]):
                # resize to 64x64 pixels调整大小至64x64像素
        #        batch_resized.append(scipy.ndimage.zoom(batch[i, :, :, :], 2.3, order=1))
        #    batch = np.stack(batch_resized)

        # Convert to RGB

        batch = batch.reshape((batch_size, 3, image_size, image_size))
        #batch = np.concatenate([batch, batch, batch], axis=1)
        #print('1 batch.shape:', batch.shape)
        # Modify images if color distribution requested 如果需要颜色分布，请修改图像
        ''''''
        if color:

            # Binarize images 二值化图像
            batch[batch >= 0.5] = 1
            batch[batch < 0.5] = 0

            # For each image in the mini batch 对于小批量中的每个图像
            for i in range(batch_size):

                # Take a random crop of the Lena image (background)随机裁剪Lena图像（背景）
                x_c = np.random.randint(0, self.lena.size[0] - image_size)
                y_c = np.random.randint(0, self.lena.size[1] - image_size)
                image = self.lena.crop((x_c, y_c, x_c + image_size, y_c + image_size))
                image = np.asarray(image).transpose((2, 0, 1)) / 255.0

                # Randomly alter the color distribution of the crop
                for j in range(3):
                    image[j, :, :] = (image[j, :, :] + np.random.uniform(0, 1)) / 2.0

                # Invert the color of pixels where there is a number反转有数字的像素的颜色
                image[batch[i, :, :, :] == 1] = 1 - image[batch[i, :, :, :] == 1]
                batch[i, :, :, :] = image
        ''''''
        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1

        # Channel last
        batch = batch.transpose((0, 2, 3, 1))

        return batch

    def get_batch(self, subset, batch_size, image_size=64, color=False, rescale=True):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[np.array(idx), :, :, :].reshape((batch_size, image_size, image_size, 3))

        # Process batch 批处理
        batch = self.process_batch(batch, batch_size, image_size, color, rescale)

        # Image label 图像标签
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')

    def get_batch_by_labels(self, subset, labels, image_size=64, color=False, rescale=True):

        # Select a subset 选择一个子集
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Find samples matching labels查找与标签匹配的样本
        #print('labels:', labels)
        idxs = []
        for i, label in enumerate(labels):

            idx = np.where(y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idxs.append(idx_sel)
        #print('len(idxs):',len(idxs))
        #print('X.shape:',X.shape)
        #print('X[np.array(idxs), :, :, :]:',X[np.array(idxs), :, :, :].shape)
        # Retrieve images 检索图像
        batch = X[np.array(idxs), :, :, :].reshape((len(labels), image_size, image_size, 3))
        #print('0 batch.shape:', batch.shape)
        # Process batch
        batch = self.process_batch(batch, len(labels), image_size, color, rescale)
        #print('2 batch.shape:', batch.shape)
        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len


class EEGGenerator(object):

    ''' Data generator providing MNIST data
    提供MNIST数据的数据生成器'''

    def __init__(self, batch_size, subset, image_size=64, color=False, rescale=True):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize eegimage dataset 初始化
        self.eegimage = EEGIMAGE()
        self.n_samples = self.eegimage.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get data
        x, y = self.eegimage.get_batch(self.subset, self.batch_size, self.image_size, self.color, self.rescale)

        # Convert y to one-hot
        y_h = np.eye(2)[y]

        return x, y_h


class SortedNumberGenerator(object):

    ''' Data generator providing lists of sorted numbers
     提供已排序数字列表的数据生成器'''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=64, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.eegimage = EEGIMAGE()
        self.n_samples = self.eegimage.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences造句
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples设置正样本的有序预测
            seed = np.random.randint(0, 10)
            sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), 2)

            if positive_samples_n <= 0:

                # Set random predictions for negative samples设置负样本的随机预测
                # Each predicted term draws a number from a distribution that excludes itself
                #每个预测项都从一个排除自身的分布中抽取一个数字
                numbers = np.arange(0, 2)
                predicted_terms = sentence[-self.predict_terms:]
                #print('predicted_terms:',predicted_terms)
                #print('enumerate(predicted_terms):',list(enumerate(predicted_terms)))
                for i, p in enumerate(predicted_terms):
                    #print('i, p:', i, p)
                    predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                    #print('predicted_terms[' + str(i) + ']', predicted_terms[i])
                sentence[-self.predict_terms:] = np.mod(predicted_terms, 10)
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images 检索实际图像
        images, _ = self.eegimage.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch 组装批次
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize 随机化
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        #print('idxs:', idxs)
        #print('[x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...:', x_images[idxs, ...].shape,
             # y_images[idxs, ...].shape, sentence_labels[idxs, ...])
        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


class SameNumberGenerator(object):

    #Data generator providing lists of similar numbers 提供类似数字列表的数据生成器'''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=2, image_size=64, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.eegimage = EEGIMAGE()
        self.n_samples = self.eegimage.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set positive samples 设置正样本
            seed = np.random.randint(0, 1)
            sentence = seed * np.ones(self.terms + self.predict_terms)

            if positive_samples_n <= 0:

                # Set random predictions for negative samples 设置负样本的随机预测
                sentence[-self.predict_terms:] = np.mod(sentence[-self.predict_terms:] + np.random.randint(1, 1, self.predict_terms), 2)
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images 检索实际图像
        images, _ = self.eegimage.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


def plot_sequences(x, y, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently
     绘制一个图，在图中可以方便地研究数字序列'''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b, 0])

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


if __name__ == "__main__":

    # Test SortedNumberGenerator 测试数字分类器
    ag = SortedNumberGenerator(batch_size=8, subset='train', terms=4, positive_samples=4, predict_terms=2, image_size=64, color=True, rescale=False)
    for (x, y), labels in ag:
        plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted1.png')
        break

