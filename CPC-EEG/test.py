from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage import io,transform #skimage模块下的io transform(图像的形变与缩放)模块
import glob  #glob 文件通配符模块
import os    #os 处理文件和目录的模块
from sklearn.utils import shuffle
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import pickle
import scipy.io as scio
import numpy as np
def load_dataset():
    def read_img(path):
        w = 64
        h = 64
        c = 3
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        print(os.listdir(path))
        print(cate)
        imgs = []
        labels = []
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder + '/*.png'):
                img = io.imread(im)
                img = transform.resize(img, (w, h))
                imgs.append(img)
                labels.append(idx)
        return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

    data, label = read_img('new_3/')
    data, label = shuffle(data, label)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    data = normalization(data)  # data / 255
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, test_size=0.2)
    return X_test, y_test


from sklearn.manifold import TSNE


def visualize(embedding, value):
    z = TSNE(n_components=2).fit_transform(embedding.detach().cpu().numpy())
    plt.figure(figsize=(16, 12))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=30, c=value, cmap="Set2", zorder=2)
    plt.show()


def eeg_test():
    X_test, y_test = load_dataset()
    output_dir = 'models/gai11'
    encoder = load_model(join(output_dir, 'encoder.h5'))
    x_preds = encoder.predict(X_test)
    #y_preds = encoder.predict(y_test)
    print('x_preds:', x_preds.shape)
    print('y_test:', y_test.shape)
    scio.savemat(r'D:\contrastive-predictive-coding-master\result_xy11tu.mat',
                 {'matrix_1':x_preds,'matrix_2':y_test})
    # pca = PCA(n_components=4)
    # projected_config = pca.fit_transform(x_preds)
    # plt.scatter(projected_config[:, 0], projected_config[:, 1], c=)
    # plt.colorbar()
# model.eval()
# out = model(data.x)
    visualize(x_preds, color = y_test)
eeg_test()
