from tensorflow.keras import  models, layers,Model  # type:ignore
import tensorflow as tf

def cnn_bilstm():
    model = models.Sequential()
    model.add(layers.Input(shape=(None, 32, 32, 1)))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='elu')))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(2, activation='softmax')))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

class CNNBiLSTMModel(Model):
    def __init__(self):
        super(CNNBiLSTMModel, self).__init__()
        self.conv_1=layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))  # type:ignore
        self.max_pool_1=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.conv_2=layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))   # type:ignore
        self.max_pool_2=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.conv_3=layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))   # type:ignore
        self.max_pool_3=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.flatten=layers.TimeDistributed(layers.Flatten())
        self.dense_1=layers.TimeDistributed(layers.Dense(128, activation='relu'))
        self.dropout_1=layers.Dropout(0.5)
        self.bilstm=layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dropout_2=layers.Dropout(0.5)
        self.dense_2=layers.TimeDistributed(layers.Dense(2, activation='softmax'))
        self.cnn_lstm=models.Sequential()
        self.cnn_lstm.add(layers.Input(shape=(None, 32, 32, 1)))
        self.cnn_lstm.add(self.conv_1)
        self.cnn_lstm.add(self.max_pool_1)
        self.cnn_lstm.add(self.conv_2)
        self.cnn_lstm.add(self.max_pool_2)
        self.cnn_lstm.add(self.conv_3)
        self.cnn_lstm.add(self.max_pool_3)
        self.cnn_lstm.add(self.flatten)
        self.cnn_lstm.add(self.dense_1)
        self.cnn_lstm.add(self.dropout_1)
        self.cnn_lstm.add(self.bilstm)
        self.cnn_lstm.add(self.dropout_2)
        self.cnn_lstm.add(self.dense_2)
    def call(self,x):
        x=self.cnn_lstm(x)
        return x