''' This module evaluates the performance of a trained CPC encoder
本模块评估经过训练的CPC编码器的性能 '''

from data_utils1 import EEGGenerator
from os.path import join, basename, dirname, exists
#import tensorflow.keras as keras
import keras

def build_model(encoder_path, image_shape, learning_rate):
    # 读取编码器
    encoder = keras.models.load_model(encoder_path)
    # 冻结权重
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False
    # 定义分类器
    x_input = keras.layers.Input(image_shape)
    #通过genc对输入的图片进行编码
    x = encoder(x_input)
    #经过一个神经元个数为128的全连接层进行处理
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    #得到x最终输出的embedding
    x = keras.layers.Dense(units=2, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)#看相似度，判断encoder模型的好坏

    # Compile model编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):
    #评估模型
    train_data = EEGGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)
    validation_data = EEGGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)
    # Prepares the model
    model = build_model(encoder_path, image_shape=(image_size, image_size, 3), learning_rate=lr)
    # Callbacks回调函数，在训练阶段使用回调函数，可以查看训练模型的内在状态和统计，
    # patience：没有进步的训练轮数，当经过patience轮后还没有进步，那么训练速率就会被降低
    #min_lr：学习率的下边界
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
    # 训练模型
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    benchmark_model(
        encoder_path='models/gai11/encoder.h5',
        epochs=100,
        batch_size=64,
        output_dir='models/gai11',
        lr=1e-3,
        image_size=64,
        color=True
    )
