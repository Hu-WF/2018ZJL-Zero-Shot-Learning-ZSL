# -*- coding: utf-8 -*-
"""Build VGG11 model based on Keras,train on ZJL datasets to extract image features"""
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from keras.utils import plot_model
import pandas as pd
import os 
from tool_utils import DataProcessing
    
"""1.VGG11 with Batch Normalization layers and Dropout layers"""
class VGG11():
    def __init__(self,):
        self.model_path="Model/"
    def training(self,x_train,y_train):
        model=Sequential()
        #Block-1
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=(64,64,3)))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Block-2
        model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Block-3
        model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Block-4
        model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Block-5
        model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',))
        model.add(BatchNormalization(axis=1,center=True,scale=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Block-6 FN
        model.add(Flatten(name='feature_2048',))
#        model.add(Dropout(0.5))
#        model.add(BatchNormalization())
        model.add(Dense(1024,name='feature_1024',))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(256,name='feature_256',))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
#        model.add(Dense(230,activation='softmax'))
#        model.add(Dense(30,activation='tanh'))
        model.add(Dense(60,activation='linear'))
#        model.add(Dense(300,activation='tanh'))
        #loss_function
        #adam=Adam(0.001)
        model.compile(loss='mse',optimizer='adam',metrics=['mse'])
#        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse'])
        
        #checkpoints
        filepath="Model/weights.{epoch:02d}-{loss:.2f}.hdf5"
#        if os.path.exists(filepath):
#            model.load_weights(filepath)
#            print("Attention:load checkpoint to continue training!")
        checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min',period=100)
        
        #checkpoints
        vgg11_weight_path="Model/vgg11_weae_weight_breakpoints.h5"
#        vgg11_weight_path="Model/weights.200-0.00.hdf5"
        if os.path.exists(vgg11_weight_path):
            model.load_weights(vgg11_weight_path)
            print("Attention:load checkpoint to continue training!")
        #early_stopping
        earlystopping=EarlyStopping(monitor='loss',patience=30,verbose=2)
        #Pack checkpoint and early stopping
        callbacks=[earlystopping,checkpoint]
#        callbacks=[checkpoint,]
        #fit
        model.fit(x_train,y_train,batch_size=64,epochs=200,shuffle=True,callbacks=callbacks)
        #save and plot models
        #complete model：
        plot_model(model=model,to_file='Model/VGG11_weae.png',show_shapes=True)
        model.save("Model/VGG11_weae_epoch300.model")
        #save breakpoints
        model.save_weights(vgg11_weight_path)
        #2048 model
        feature_2048=Model(inputs=model.input,outputs=model.get_layer("feature_2048").output)
        feature_2048.save("Model/VGG11_weae_fea2048_epoch300.model")
#        plot_model(model=feature_2048,to_file='Model/VGG11_fea2048.png',show_shapes=True)
        #1024 model
        feature_1024=Model(inputs=model.input,outputs=model.get_layer("feature_1024").output)
        feature_1024.save("Model/VGG11_weae_fea1024_epoch300.model")
#        plot_model(model=feature_1024,to_file='Model/VGG11_fea1024.png',show_shapes=True)
        #256 model
        feature_256=Model(inputs=model.input,outputs=model.get_layer("feature_256").output)
        feature_256.save("Model/VGG11_weae_fea256_epoch300.model")
#        plot_model(model=feature_512,to_file='Model/VGG11_fea512.png',show_shapes=True)
        return 0
    
    def prediction(self,model_path,input_pic,save_name):
        model=load_model(model_path)
        pre=model.predict(input_pic,batch_size=64)
        pre=pd.DataFrame(pre)
        pre.to_csv(save_name,header=None,index=None)
        return 0
    

"""Training and converting pictures into 2048features"""
def training():
    #Data
    dp=DataProcessing()
    _,train_pic=dp.read_pictures_to_array(path='DatasetB_20180919/train/')
    train_label=pd.read_csv("Data/train_pic_we.csv",header=None,index_col=None)
#    train_label=pd.read_csv("Data/train_pic_att.csv",header=None,index_col=None)
#    train_label=pd.read_csv("Data/train_pic_onehot.csv",header=None,index_col=None)
    #Training
    model=VGG11()
    model.training(x_train=train_pic,y_train=train_label)
#    model=VGG16()
#    model.training(x_train=train_pic,y_train=train_label)
    return 0

def converting():
    #Data
    dp=DataProcessing()
    _,train_pic=dp.read_pictures_to_array(path='DatasetB_20180919/train/')
    _,test_pic=dp.read_pictures_to_array(path='DatasetB_20180919/test/')
    #Converting train and test pictures into 2048features.
    model=VGG11()
    model.prediction(model_path="Model/VGG11_weae_fea2048_epoch300.model",input_pic=train_pic,save_name="Data/01_train_fea2048_weae.csv")
    model.prediction(model_path="Model/VGG11_weae_fea2048_epoch300.model",input_pic=test_pic,save_name="Data/02_test_fea2048_weae.csv")
    
#    model.prediction(model_path="Model/VGG11_onehot_fea2048_epoch1k.model",input_pic=train_pic,save_name="Data/05_train_fea2048_att.csv")
#    model.prediction(model_path="Model/VGG11_onehot_fea2048_epoch1k.model",input_pic=test_pic,save_name="Data/06_test_fea2048_att.csv")
    
#    model.prediction(model_path="Model/VGG11_onehot_epoch1k.model",input_pic=test_pic,save_name="DataPrediction/vgg11_att_prediction.csv")
    return 0


if __name__=='__main__':
    training()
#    converting()

    

        