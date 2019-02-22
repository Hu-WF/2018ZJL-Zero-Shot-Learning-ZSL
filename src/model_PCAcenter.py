# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tool_utils import DataProcessing
from keras.layers import Input,Dense,Dropout,BatchNormalization,Activation
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

class PCACenter():
    def __init__(self,):
        self.__n_per_class=200
        self.__n_class=230
        self.__n_save_components=200
        
    #将CVAE算法重建出的all_pseudoTrain_fea分类别求各自均值
    def __get_class_mean(self,):
        fea=pd.read_csv("Data/all_pseudoTrain_fea.csv",header=None,index_col=None)
        num=self.__n_per_class
        results=np.zeros(shape=(230,2048))
        for i in range(self.__n_class):
            sum_class=fea.iloc[i*num:(i+1)*num,:].sum()
            sum_class=sum_class/200
            results[i,:]=sum_class
        print(results,results.shape)
        results=pd.DataFrame(results)
        results.to_csv("Data/mean_per_class.csv",header=None,index=None)
        return 0
    
    #将所有类均值中心一起做PCA降维处理
    def __mypca(self,):
        fea=pd.read_csv("Data/mean_per_class.csv",header=None,index_col=None)
        #白化去相关后效果很不好
        pca=PCA(n_components=self.__n_save_components)
#        pca=PCA(n_components=230,)
        fea=pca.fit_transform(fea)
        print(pca.explained_variance_ratio_)
        fea=pd.DataFrame(fea)
        fea.to_csv("Data/std_fea_mean_pca.csv",header=None,index=None)

    #获取Train label        
    def __get_train_fea_mean_pca(self,):
        dp=DataProcessing()
        pic_name , _ =dp.read_pictures_to_array(path='DatasetA_train_20180813/train/')
        dp.get_training_x_with_y(train_pic_name_order=pic_name,y="Data/std_fea_mean_pca.csv")
        print("sucess.")
        return 0
        
    def __dnn(self,train_fea,train_label):
        x_in=Input(shape=(2048,),)
        #Block-1
        x=Dense(1024,)(x_in)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        #Block-2
        x=Dense(512,)(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        #Block-2
        x=Dense(256,)(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        #Block-3
        x=Dropout(0.2)(x)
#        y=Dense(self.__n_save_components,activation='tanh')(x)
        y=Dense(self.__n_save_components,activation='linear')(x)
        #Model
#        adam=Adam(lr=0.0001)
        adam=Adam(lr=0.001)
        model=Model(inputs=x_in,outputs=y)
        model.compile(optimizer=adam,loss='mse')
#        model.compile(optimizer='adam',loss='categorical_crossentropy')
#        model.compile(optimizer=adam,loss='categorical_hinge')
        #Load breakpoint
        weight_path="Model/pcadnn_weight_breakpoints.h5"
        if os.path.exists(weight_path):
            print("load weight from "+weight_path+"...")
            model.load_weights(weight_path)
            print("Sucess!")
        #Early stopping
        earlystopping=EarlyStopping(monitor='loss',patience=20,verbose=2)
        callbacks=[earlystopping,]
        #Train model
        model.fit(x=train_fea,y=train_label,batch_size=128,epochs=50,shuffle=True,callbacks=callbacks)
        #save model
#        plot_model(model=model,to_file='Model/dnn.png',show_shapes=True)
        model.save("Model/pcadnn.model")
        model.save_weights(weight_path)
        return 0
    
    #使用原始train fea进行训练（尝试过再用重建的all_fea作为输入，效果非常差），标签为新建出来的重建均值PCA视觉中心。
    def dnn_training(self,):
        fea=pd.read_csv("Data/03_train_fea2048_att_norm.csv",header=None,index_col=None)
#        fea=pd.read_csv("Data/03_train_features_norm.csv",header=None,index_col=None)
        label=pd.read_csv("Data/train_pic_y.csv",header=None,index_col=None)
        self.__dnn(train_fea=fea,train_label=label)
        return 0
    
    def dnn_prediction(self,):
        model=load_model("Model/pcadnn.model")
#        fea=pd.read_csv("Data/04_test_features_norm.csv",header=None,index_col=None)
        fea=pd.read_csv("Data/04_test_fea2048_att_norm.csv",header=None,index_col=None)
        prediction=model.predict(fea)
        prediction=pd.DataFrame(prediction)
        prediction.to_csv("DataPrediction/pcadnn_prediction.csv",header=None,index=None)
        return 0
    
    def get_std_and_train_pcamean(self,):
        self.__get_class_mean()
        self.__mypca()
        self.__get_train_fea_mean_pca()
        print("Generate std and train mean pca center(fea were created by CVAE model).")
        return 0
        
        
        
    
if __name__=='__main__':
    pc=PCACenter()

#    pc.get_std_and_train_pcamean()
    pc.dnn_training()
#    pc.dnn_prediction()
    
    
        
        
        
    

