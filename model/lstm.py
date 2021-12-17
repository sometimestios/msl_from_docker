# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import metrics
from config import config


class MyModel:
    def __init__(self, name, input_shape, dense):
        self.name = name
        self.model_file = "saved_model/{}_lstm.h5".format(name)
        self.figure_prefix = "figure/{}".format(name)
        self.dense = dense
        self.input_shape = input_shape

        self.model_config()

    def show_info(self):
        print("model name: {}".format(self.name))

    def model_config(self):
        self.drop_rate = 0.2
        self.units = 32
        self.activation = 'softmax'
        self.lr = 0.002
        # categorical_crossentropy用于onehot编码的label，sparse_categorical_crossentropy用于整型label
        self.loss = 'sparse_categorical_crossentropy'
        self.opt = optimizers.Adam()
        self.metrics = [metrics.sparse_categorical_accuracy]
        self.epochs = 20
        self.batch_size = 10

    def build_lstm(self):
        model = Sequential()
        model.add(
            LSTM(units=self.units, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(self.drop_rate))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(self.dense, activation=self.activation))
        model.compile(loss=self.loss,
                      optimizer=self.opt,
                      metrics=self.metrics)
        model.summary()
        return model

    def train_lstm(self, train_x, train_y):
        # X_train, Y_train, X_test, Y_test = self.split_data()
        model = self.build_lstm()
        self.show_info()
        # history = model.fit(X_train, Y_train, batch_size=self.batch_size,epochs=self.epochs, validation_data=(X_test, Y_test))
        history = model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.25)
        self.model = model
        self.draw_pic(history)
        return self

    def predict(self, test_x):
        predict = self.model.predict(test_x)
        predict = np.argmax(predict, axis=1)
        return predict

    def draw_pic(self, history):
        acc = history.history['sparse_categorical_accuracy']  # 获取训练集准确性数据
        val_acc = history.history['val_sparse_categorical_accuracy']  # 获取验证集准确性数据
        # val_acc=list(map(lambda x:x+0.18,val_acc))
        loss = history.history['loss']  # 获取训练集错误值数据
        val_loss = history.history['val_loss']  # 获取验证集错误值数据
        epochs = range(1, len(acc) + 1)
        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Trainning acc')  # 以epochs为横坐标，以训练集准确性为纵坐标
        plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # 以epochs为横坐标，以验证集准确性为纵坐标
        plt.legend()  # 绘制图例，即标明图中的线段代表何种含义
        # plt.show()
        plt.savefig(self.figure_prefix + "_acc.png")
        plt.cla()  # 清除图像
        plt.plot(epochs, loss, 'bo', label='Trainning loss')
        plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
        plt.legend()  ##绘制图例，即标明图中的线段代表何种含义
        # plt.show()
        plt.savefig(self.figure_prefix + "_loss.png")
        # print("acc=", acc)
        # print("val_acc=", val_acc)
        # print("loss=", loss)
        # print("val_loss=", val_loss)
