# -*- coding: utf-8 -*-
"""CVAE model"""
from keras.layers import Input,Dense,Dropout,Lambda,concatenate,BatchNormalization,Activation
from keras.callbacks import EarlyStopping
from keras.models import Model,load_model
import keras.backend as K
from keras.utils import plot_model
import pandas as pd
import numpy as np
import numpy.random as rng
import os
from keras.optimizers import Adam
#from sklearn import svm
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OneHotEncoder

"""Using CVAE model to create pseudoTrain_features for unseen classes"""
class CVAE():
    def __init__(self,):
        self.__n_fea=2048
        self.__n_we=300
#        self.__n_z=256
#        self.__n_z=2
#        self.__n_z=128
#        self.__n_z=32
        self.__n_z=64
#        self.__n_z=50
        self.__n_create_perclass=200
        self.__n_unseen_class=40
        #Training data
#        self.__train_pic=pd.read_csv("Data/09_train_fea2048_norm.csv",header=None,index_col=None)
        self.__train_pic=pd.read_csv("Data/03_train_fea2048_att_norm.csv",header=None,index_col=None)

        self.__train_we=pd.read_csv("Data/train_pic_we.csv",header=None,index_col=None)
    
    def __model(self,in_pic,in_we,out_pic):
#        n_fea=2048
        n_we,n_z=self.__n_we,self.__n_z
        pic=Input(shape=(2048,),)
        #x=Input(shape=(n_fea+n_we,),)
        we=Input(shape=(300,),)
        #Concatenate input picture feature(2048) and wordembeddings(300).
        x=concatenate(inputs=[pic,we])
        #Block-1   ;hidden layers
        h=Dense(1024,)(x)
        h=BatchNormalization()(h)
        h=Activation('relu')(h)
        
#        h=Dropout(0.5)(h)
        #Block-2
        h=Dense(512,)(h)
#        h=Dense(256,)(h)
        h=BatchNormalization()(h)
        h=Activation('relu')(h)
#        h=Dropout(0.5)(h)
        #Block-3
#        h=Dense(256,)(h)
#        h=BatchNormalization()(h)
#        h=Activation('relu')(h)
        #Block-4
        h=Dense(256,)(h)
        h=BatchNormalization()(h)
        h=Activation('relu')(h)
        h=Dropout(0.5)(h)
        #Block-5
        z_mean=Dense(n_z,activation='linear')(h)
        z_log_var=Dense(n_z,activation='linear')(h)
        #Block-6,sampling
        def sampling(args):
            z_mean,z_log_var=args
            epsilon=K.random_normal(shape=(n_z,),mean=0.,stddev=1.)
            return z_mean+K.exp(z_log_var/2)*epsilon
        #Block-7,Lambda layer
        z=Lambda(sampling)([z_mean,z_log_var])
        #concatenate z and wordembeddings.
        z_conc=concatenate(inputs=[z,we])
        #Block-8,Decoder layer
        decoder_hidden_1=Dense(512,)
        decoder_hidden_2=BatchNormalization()
        decoder_hidden_3=Activation('relu')
        decoder_hidden_4=Dense(1024,)
        decoder_hidden_5=BatchNormalization()
        decoder_hidden_6=Activation('relu')
        decoder_out=Dense(2048,activation='sigmoid')
#        decoder_out=Dense(2048,activation='tanh')
#        decoder_out=Dense(2048,activation='linear')
        #instantiation
        d_h=decoder_hidden_1(z_conc)
        d_h=decoder_hidden_2(d_h)
        d_h=decoder_hidden_3(d_h)
#        d_h=decoder_hidden_4(z_conc)
        d_h=decoder_hidden_4(d_h)
        d_h=decoder_hidden_5(d_h)
        d_h=decoder_hidden_6(d_h)
        xout=decoder_out(d_h)
        #Train model
        vae=Model(inputs=[pic,we],outputs=[xout])
        #Packaging decoder
        d1=Input(shape=(n_z,),)
        d2=Input(shape=(n_we,),)
        d_in=concatenate(inputs=[d1,d2])
        #decoder layers
        decoder_h=decoder_hidden_1(d_in)
        decoder_h=decoder_hidden_2(decoder_h)
        decoder_h=decoder_hidden_3(decoder_h)
#        decoder_h=decoder_hidden_4(d_in)
        decoder_h=decoder_hidden_4(decoder_h)
        decoder_h=decoder_hidden_5(decoder_h)
        decoder_h=decoder_hidden_6(decoder_h)
        d_out=decoder_out(decoder_h)
        #Decoder model
        decoder=Model([d1,d2],d_out)
        #Define VAE loss
        def vae_loss(y_true,y_pred):
            mse_loss=K.mean(K.square(y_pred-y_true),axis=1)
            kl_loss=0.5*K.sum(K.exp(z_log_var)+K.square(z_mean)- 1. -z_log_var,axis=1)
            return mse_loss+kl_loss
        #Training vae model
        adam=Adam(lr=0.001)
#        adam=Adam(lr=0.0005)
        vae.compile(optimizer=adam,loss=vae_loss)
#        vae.compile(optimizer='adam',loss=vae_loss)
        #Load breakpoint
        vae_weight_path="Model/cvae_weight_breakpoints.h5"
        if os.path.exists(vae_weight_path):
            print("Loading weight from "+vae_weight_path+"...")
            vae.load_weights(vae_weight_path)
            print("Sucess!")
        #Early stopping
        earlystopping=EarlyStopping(monitor='loss',patience=20,verbose=2)
        callbacks=[earlystopping,]
        #Training VAE
#        vae.fit(x=[in_pic,in_we],y=[out_pic],shuffle=True,batch_size=128,epochs=100,callbacks=callbacks)
        vae.fit(x=[in_pic,in_we],y=[out_pic],shuffle=True,batch_size=128,epochs=50,callbacks=callbacks)
        #Save model
        vae.save("Model/cvae.model")
        decoder.save("Model/cvae_decoder.model")
        vae.save_weights(vae_weight_path)
        plot_model(model=vae,to_file="Model/cvae.png",show_shapes=True)
        #plot_model(model=decoder,to_file="Model/cvae_decoder.png",show_shapes=True)
        return 0
    
    def training(self,):
        pic=np.asarray(self.__train_pic)
        we=np.asarray(self.__train_we)
        print("Start training CVAE model...")
        self.__model(in_pic=pic,in_we=we,out_pic=pic)
        return 0
        
    def __z_generator(self,):
        n_total_create=self.__n_create_perclass * self.__n_unseen_class
        n_z=self.__n_z
        z=rng.normal(size=(n_total_create,n_z),loc=0,scale=1,)
        return z
    
    def __z_generator_all(self,):
        n_total_create=self.__n_create_perclass * 230
        n_z=self.__n_z
        z=rng.normal(size=(n_total_create,n_z),loc=0,scale=1,)
        return z
        
    def prediction(self,):
        #load model
        decoder=load_model("Model/cvae_decoder.model",)
        #load unseen wordembeddings
        unseen=pd.read_csv("Data/unseen_we_vs_label.csv",header=None,index_col=None)
        unseen_header=np.asarray(unseen.iloc[:,0])
        unseen_we=np.asarray(unseen.iloc[:,1:])
        #calculate about unseen class
        n_unseen_class=unseen_we.shape[0]
        n_create_perclass=self.__n_create_perclass
        n_total_create=n_unseen_class * n_create_perclass
        #Create holder to storage
        pseudoTrain_we=np.zeros(shape=(n_total_create,300))
        pseudoTrain_label=[]
        #Create wordembeddings for prediction input
        for i in range(n_unseen_class):
            for j in range(n_create_perclass):
                pseudoTrain_we[200*i+j,:]=unseen_we[i,:]
                pseudoTrain_label.append(unseen_header[i])
        #print(pseudoTrain_we,pseudoTrain_label)
        #Create z for prediction input
        z=self.__z_generator()
        z=np.asarray(z)
        #Storage train label and wordembeddings
        pseudoTrain_label=pd.DataFrame(pseudoTrain_label)
        pseudoTrain_label.to_csv("Data/pseudoTrain_label.csv",header=None,index=None)
        pseudoTrain_we=pd.DataFrame(pseudoTrain_we)
        pseudoTrain_we.to_csv("Data/pseudoTrain_we.csv",header=None,index=None)
        #Prediction to create pseudoTrain_features:
        pseudoTrain_fea=decoder.predict([z,pseudoTrain_we],batch_size=128)
        #Storage train features
        pseudoTrain_fea=pd.DataFrame(pseudoTrain_fea)
        pseudoTrain_fea.to_csv("Data/pseudoTrain_fea.csv",header=None,index=None)
        return 0
        
    def prediction_seen_unseen(self,):
        #load model
        decoder=load_model("Model/cvae_decoder.model",)
        #load unseen wordembeddings
        std_we=pd.read_csv("Data/std_we.csv",header=None,index_col=None)
        header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
        std_we=np.asarray(std_we)
        header=np.asarray(header)
#        unseen=pd.read_csv("Data/unseen_we_vs_label.csv",header=None,index_col=None)
#        unseen_header=np.asarray(unseen.iloc[:,0])
#        unseen_we=np.asarray(unseen.iloc[:,1:])
        #calculate about unseen class
        n_unseen_class=std_we.shape[0]
        n_create_perclass=self.__n_create_perclass
        n_total_create=n_unseen_class * n_create_perclass
        #Create holder to storage
        pseudoTrain_we=np.zeros(shape=(n_total_create,300))
        print(n_total_create)
        pseudoTrain_label=[]
        #Create wordembeddings for prediction input
        for i in range(n_unseen_class):
            for j in range(n_create_perclass):
                pseudoTrain_we[200*i+j,:]=std_we[i,:]
                pseudoTrain_label.append(header[i])
        #print(pseudoTrain_we,pseudoTrain_label)
        #Create z for prediction input
        z=self.__z_generator_all()
        z=np.asarray(z)
        print("Z",z.shape)
        #Storage train label and wordembeddings
        pseudoTrain_label=pd.DataFrame(pseudoTrain_label)
        pseudoTrain_label.to_csv("Data/all_pseudoTrain_label.csv",header=None,index=None)
        pseudoTrain_we=pd.DataFrame(pseudoTrain_we)
        pseudoTrain_we.to_csv("Data/all_pseudoTrain_we.csv",header=None,index=None)
        #Prediction to create pseudoTrain_features:
        pseudoTrain_fea=decoder.predict([z,pseudoTrain_we],batch_size=128)
        #Storage train features
        pseudoTrain_fea=pd.DataFrame(pseudoTrain_fea)
        print(pseudoTrain_fea.shape)
        pseudoTrain_fea.to_csv("Data/all_pseudoTrain_fea.csv",header=None,index=None)
        return 0


"""Using DNN model to training after the CVAE model create pseudoTrain_features for unseen classes"""
"""Mixing seen and unseen classes features,and treat them as a general classification problem"""
class DNN():
    def __init__(self,):
        #Training data
#        self.__train_fea=pd.read_csv("Data/03_train_features_norm.csv",header=None,index_col=None)
#        self.__train_fea=pd.read_csv("Data/03_train_features_norm.csv",header=None,index_col=None)
#        self.__train_label=pd.read_csv("Data/train_pic_onehot.csv",header=None,index_col=None)
#        self.__train_label=pd.read_csv("Data/train_pic_num.csv",header=None,index_col=None)
#        self.__pseudoTrain_fea=pd.read_csv("Data/pseudoTrain_fea.csv",header=None,index_col=None)
#        self.__pseudoTrain_label=pd.read_csv("Data/pseudoTrain_onehot.csv",header=None,index_col=None)
#        self.__pseudoTrain_label=pd.read_csv("Data/pseudoTrain_num.csv",header=None,index_col=None)
        #Test data

        self.__test_fea=pd.read_csv("Data/04_test_fea2048_att_norm.csv",header=None,index_col=None)
        
#        self.__all_train_fea=pd.read_csv("Data/all_pseudoTrain_fea_norm.csv",header=None,index_col=None)
        self.__all_train_fea=pd.read_csv("Data/all_pseudoTrain_fea.csv",header=None,index_col=None)
        self.__all_train_label=pd.read_csv("Data/all_pseudoTrain_we.csv",header=None,index_col=None)
#        self.__all_train_label=pd.read_csv("Data/all_pseudoTrain_att.csv",header=None,index_col=None)
        
    
    def __model(self,train_fea,train_label):
        x_input=Input(shape=(2048,),)
        #Block-1
        x=Dense(1024,)(x_input)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        #Block-2
        x=Dense(512,)(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        #Block-2
#        x=Dense(256,)(x)
#        x=BatchNormalization()(x)
#        x=Activation('relu')(x)
        #Block-2
#        x=Dense(128,)(x)
#        x=BatchNormalization()(x)
#        x=Activation('relu')(x)
        #Block-3
        x=Dropout(0.2)(x)
        y=Dense(300,activation='tanh')(x)
#        y=Dense(30,activation='sigmoid')(x)
        #Model
#        adam=Adam(lr=0.0001)
        adam=Adam(lr=0.001)
        model=Model(inputs=x_input,outputs=y)
#        model.compile(optimizer='adam',loss='mse')
        model.compile(optimizer=adam,loss='mse')
#        model.compile(optimizer='adam',loss='categorical_crossentropy')
#        model.compile(optimizer=adam,loss='categorical_hinge')
        #Load breakpoint
        weight_path="Model/dnn_weight_breakpoints.h5"
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
        model.save("Model/dnn.model")
        model.save_weights(weight_path)
        return 0
    
    def training(self,):
        train_fea , pseudoTrain_fea=self.__train_fea , self.__pseudoTrain_fea
        train_label , pseudoTrain_label=self.__train_label , self.__pseudoTrain_label
        #concat diffent data for tanining
        fea=pd.concat([train_fea,pseudoTrain_fea],axis=0)
        label=pd.concat([train_label,pseudoTrain_label],axis=0)
        #Training with data
        print("Start training DNN model...")
        self.__model(train_fea=fea,train_label=label)
        return 0
    
    def all_training(self,):
        fea=self.__all_train_fea
        label=self.__all_train_label
        #Training with data
        print("Start training DNN model...")
        self.__model(train_fea=fea,train_label=label)
        return 0
    
#    def svm_training_prediction(self,):
#        train_fea , pseudoTrain_fea=self.__train_fea , self.__pseudoTrain_fea
#        train_label , pseudoTrain_label=self.__train_label , self.__pseudoTrain_label
#        #concat diffent data for tanining
#        fea=pd.concat([train_fea,pseudoTrain_fea],axis=0)
#        label=pd.concat([train_label,pseudoTrain_label],axis=0)
#        #Training with data
#        print("Start training SVM model...")
#        clf=svm.SVC(C=100)
#        clf.fit(fea,label)
#        #Prediction
#        test_fea=np.asarray(self.__test_fea)
#        print("Start prediction with SVM model...")
#        prediction=clf.predict(test_fea)
#        prediction=pd.DataFrame(prediction)
#        prediction.to_csv("DataPrediction/svm_prediction.csv",header=None,index=None)
#        return 0
    
    def prediction(self,):
        model=load_model("Model/dnn.model")
        fea=np.asarray(self.__test_fea)
        print("Start prediction...")
        prediction=model.predict(fea)
        prediction=pd.DataFrame(prediction)
#        prediction.to_csv("Data/dnn_prediction.csv",header=None,index=None)
        prediction.to_csv("DataPrediction/dnn_prediction.csv",header=None,index=None)
        return 0
    

class CVAEData():
    def __init__(self,):
        #For unseen
        self.__train=pd.read_csv("Data/train.csv",header=None,index_col=None)
        self.__header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
        self.__std_we=pd.read_csv("Data/std_we.csv",header=None,index_col=None)
        #for pseudoTrain att
        self.__att_per_class=pd.read_csv("Data/att_per_class.csv",header=None,index_col=None)
        self.__std_pcacenter=pd.read_csv("Data/std_fea_mean_pca.csv",header=None,index_col=None)
#        self.__pseudoTrain_label=pd.read_csv("Data/pseudoTrain_label.csv",header=None,index_col=None)
        
        self.__all_pseudoTrain_label=pd.read_csv("Data/all_pseudoTrain_label.csv",header=None,index_col=None)
        
        
#        self.__std_onehot=pd.read_csv("Data/std_onehot.csv",header=None,index_col=None)
#        self.__std_num=pd.read_csv("Data/std_num.csv",header=None,index_col=None)
    
    def get_unseen_we_vs_label(self,):
        #Get seen label
        train=np.asarray(self.__train)
        seen=np.unique(train)
        #Get unseen label
        unseen=[]
        header=np.asarray(self.__header.iloc[:,0])
        for i in range(header.shape[0]):
            if header[i] not in seen:
                unseen.append(header[i])
        #Get unseen wordembeddings
        num=len(unseen)
        std_we=self.__std_we
        unseen_we=np.zeros(shape=(num,300))
        for i in range(num):
            loc=np.argwhere(unseen[i] == header)[0][0]
            unseen_we[i,:]=std_we.iloc[loc,:]
        #Save
        unseen=pd.DataFrame(unseen)
        unseen_we=pd.DataFrame(unseen_we)
        result=pd.concat([unseen,unseen_we],axis=1)
        result.to_csv("Data/unseen_we_vs_label.csv",header=None,index=None)
        return 0
    
    def creat_pseudoTrain_att(self,):
        header=self.__att_per_class.iloc[:,0]
        #std_att=self.__att_per_class.iloc[:,1:]
        num=self.__pseudoTrain_label.shape[0]
        pseudoTrain_label=np.asarray(self.__pseudoTrain_label)
        atts=np.zeros(shape=(num,30))
        for i in range(num):
            #print(pse_train_label[i][0])
            loc=np.argwhere(pseudoTrain_label[i][0] == header)[0][0]
#            atts[i]=att_per_class.iloc[loc,1:]
            atts[i]=self.__att_per_class.iloc[loc,1:]
        atts=pd.DataFrame(atts)
        atts.to_csv("Data/all_pseudoTrain_att.csv",header=None,index=None)
        return 0
    
    def creat_pseudoTrain_pcacenter(self,):
        header=self.__att_per_class.iloc[:,0]
        #std_att=self.__att_per_class.iloc[:,1:]
        num=self.__all_pseudoTrain_label.shape[0]
        pseudoTrain_label=np.asarray(self.__all_pseudoTrain_label)
        atts=np.zeros(shape=(num,200))
        for i in range(num):
            #print(pse_train_label[i][0])
            loc=np.argwhere(pseudoTrain_label[i][0] == header)[0][0]
#            atts[i]=att_per_class.iloc[loc,1:]
            atts[i]=self.__std_pcacenter.iloc[loc,:]
        atts=pd.DataFrame(atts)
        atts.to_csv("Data/all_pseudoTrain_pcacenter.csv",header=None,index=None)
        return 0
    
    def creat_pseudoTrain_att_all(self,):
        header=self.__att_per_class.iloc[:,0]
        #std_att=self.__att_per_class.iloc[:,1:]
        num=self.__all_pseudoTrain_label.shape[0]
        pseudoTrain_label=np.asarray(self.__all_pseudoTrain_label)
        atts=np.zeros(shape=(num,30))
        for i in range(num):
            #print(pse_train_label[i][0])
            loc=np.argwhere(pseudoTrain_label[i][0] == header)[0][0]
#            atts[i]=att_per_class.iloc[loc,1:]
            atts[i]=self.__att_per_class.iloc[loc,1:]
        atts=pd.DataFrame(atts)
        atts.to_csv("Data/all_pseudoTrain_att.csv",header=None,index=None)
        return 0
    
    def creat_pseudoTrain_onehot(self,):
        header=self.__std_onehot.iloc[:,0]
        #std_onehot=self.__std_onehot.iloc[:,1:]
        num=self.__pseudoTrain_label.shape[0]
        pseudoTrain_label=np.asarray(self.__pseudoTrain_label)
        onehots=np.zeros(shape=(num,230))
        for i in range(num):
            #print(pse_train_label[i][0])
            loc=np.argwhere(pseudoTrain_label[i][0] == header)[0][0]
            onehots[i]=self.__std_onehot.iloc[loc,1:]
        onehots=pd.DataFrame(onehots)
        onehots.to_csv("Data/pseudoTrain_onehot.csv",header=None,index=None)
        return 0
    
    def creat_pseudoTrain_num(self,):
        header=self.__std_num.iloc[:,0]
        #std_onehot=self.__std_onehot.iloc[:,1:]
        num=self.__pseudoTrain_label.shape[0]
        pseudoTrain_label=np.asarray(self.__pseudoTrain_label)
        onehots=np.zeros(shape=(num,1))
        for i in range(num):
            #print(pse_train_label[i][0])
            loc=np.argwhere(pseudoTrain_label[i][0] == header)[0][0]
            onehots[i]=self.__std_num.iloc[loc,1:]
        onehots=pd.DataFrame(onehots)
        onehots.to_csv("Data/pseudoTrain_num.csv",header=None,index=None)
        return 0
    
    def processing(self,):
#        self.creat_pseudoTrain_onehot()
#        self.creat_pseudoTrain_num()
#        self.creat_pseudoTrain_att_all()
#        self.get_unseen_we_vs_label()
        self.creat_pseudoTrain_pcacenter()
        return 0
    
    
if __name__=="__main__":
#    cvae=CVAE()
#    cvae.training()
#    cvae.prediction_seen_unseen()
    """training 之后记得prediction！！！！！"""
#    cvae.prediction()#training 之后记得prediction！！！！！
    
#    cvaed=CVAEData()
#    cvaed.processing()
    
    dnn=DNN()
#    dnn.all_training()
#    dnn.training()
#    dnn.svm_training_prediction()
    dnn.prediction()
    