# -*- coding: utf-8 -*-
"""建立自编码器对wordembedding和att进行无监督降维"""
from keras.models import Model,load_model
from keras.layers import Dense,Input,BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
#from keras.utils import plot_model
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

def wordembedding():
    we=pd.read_csv("Data/class_wordembeddings.csv",header=None,index_col=None)
    value=we.iloc[:,1:]
    print(value)
    value=np.array(value)
    return value

#AE for wordembeddings
def we_AE_training(x_train,y_train):
    #encoded
    input_layer=Input(shape=(300,),)
    encoded=Dense(256,activation='relu')(input_layer)
    encoded=Dense(60,activation='relu')(encoded)
    decoded=Dense(256,activation='relu')(encoded)
    decoded=Dense(300,activation='tanh')(decoded)
    ##完整自编码器
    autoencoder=Model(inputs=input_layer,outputs=decoded)
    ##编码层
    encoder=Model(inputs=input_layer,outputs=encoded)
    ###训练编码器
    autoencoder.compile(loss='mse',optimizer='adam',)
    earlystopping=EarlyStopping(monitor='loss',patience=50,verbose=2)
    callbacks=[earlystopping,]
    #training,epochs:迭代次数
    autoencoder.fit(x_train,y_train,batch_size=64,epochs=2000,callbacks=callbacks)
    #Draw the network structure diagram
#    plot_model(model=autoencoder,to_file='AE_we_model.png',show_shapes=True)
#    plot_model(model=encoder,to_file='E_we_model.png',show_shapes=True)
    #Save model
#    autoencoder.save("AE_we.model")
    encoder.save("Model/we_AEencoder.model")
    return 0
    
def converting():
    #获取header
    we_old=pd.read_csv("Data/class_wordembeddings.csv",header=None,index_col=None)
    header=we_old.iloc[:,0] 
    we=we_old.iloc[:,1:]
    we=np.array(we)
    #载入模型进行转换
    model=load_model("Model/we_AEencoder.model",)
    we_new=model.predict(we)
    we_new=pd.DataFrame(we_new)
    #we_new.to_csv("encoder_output.csv",header=None,index=None)
    #归一化操作
#    mms=MinMaxScaler()
#    we_new=mms.fit_transform(we_new)
#    we_new=pd.DataFrame(we_new)
    #结合header然后输出：
    we_new=pd.concat([header,we_new],axis=1)
    we_new.to_csv("Data/class_wordembeddings_AE60.csv",header=None,index=None)
    return 0

if __name__=='__main__':
    ###wordembedding自编码
#    we=wordembedding()
#    we_AE_training(x_train=we,y_train=we)
    converting()