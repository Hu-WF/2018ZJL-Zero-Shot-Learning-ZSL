# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numpy.random as rng
from keras.layers import Input,Lambda,Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model,load_model
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
import os
from tool_utils import DataProcessing

"""1.Building Pseudo-Siamese Network，PSN"""
class PSN():
#    def __init__(self,):
#        #Model input data
#        self.train_we=pd.read_csv("Data/train_pic_we.csv",header=None,index_col=None)
##        self.train_we=pd.read_csv("Data/fuse_train_att_we.csv",header=None,index_col=None)
#        
#        self.train_pic=pd.read_csv("Data/03_train_features_norm.csv",header=None,index_col=None)
#        self.test_pic=pd.read_csv("Data/04_test_features_norm.csv",header=None,index_col=None)
#        
##        self.train_pic=pd.read_csv("Data/07_preVGG19_train_fea2048_norm.csv",header=None,index_col=None)
##        self.test_pic=pd.read_csv("Data/08_preVGG19_test_fea2048_norm.csv",header=None,index_col=None)
#        
##        self.std_we_cosine_distances=pd.read_csv("Data/std_we_cosine_distances.csv",header=None,index_col=None)
#        #Other data
#        self.all_data=pd.read_csv("Data/all_data.csv",header=None,index_col=None)
##        self.std_we=pd.read_csv('Data/fuse_std_att_we.csv',header=None,index_col=None)
#        self.std_we=pd.read_csv('Data/std_we.csv',header=None,index_col=None)
#        self.header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
    
    #将CVAE重建出的所有46000fea作为输入进行训练
    def __init__(self,):
        #Model input data
        self.train_we=pd.read_csv("Data/all_pseudoTrain_we.csv",header=None,index_col=None)
#        self.train_we=pd.read_csv("Data/train_pic_we.csv",header=None,index_col=None)

        
        self.train_pic=pd.read_csv("Data/all_pseudoTrain_fea.csv",header=None,index_col=None)
        self.test_pic=pd.read_csv("Data/04_test_fea2048_att_norm.csv",header=None,index_col=None)
        
        self.std_we_cosine_distances=pd.read_csv("Data/std_we_cosine_distances.csv",header=None,index_col=None)
        #Other data
        self.all_data=pd.read_csv("Data/all_pseudoTrain_label.csv",header=None,index_col=None)
#        self.all_data=pd.read_csv("Data/all_data.csv",header=None,index_col=None)
#        self.std_we=pd.read_csv('Data/fuse_std_att_we.csv',header=None,index_col=None)
        self.std_we=pd.read_csv('Data/std_we.csv',header=None,index_col=None)
        self.header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
        
        
#    def __init__(self,):
#        #Model input data
#        self.train_we=pd.read_csv("Data/CVAEPSN_train_we.csv",header=None,index_col=None)
#
#        
#        self.train_pic=pd.read_csv("Data/CVAEPSN_train_fea.csv",header=None,index_col=None)
#        self.test_pic=pd.read_csv("Data/04_test_features_norm.csv",header=None,index_col=None)
#        
##        self.std_we_cosine_distances=pd.read_csv("Data/std_we_cosine_distances.csv",header=None,index_col=None)
#        #Other data
##        self.all_data=pd.read_csv("Data/all_data.csv",header=None,index_col=None)
#        self.all_data=pd.read_csv("Data/CVAEPSN_all_data.csv",header=None,index_col=None)
##        self.std_we=pd.read_csv('Data/fuse_std_att_we.csv',header=None,index_col=None)
#        self.std_we=pd.read_csv('Data/std_we.csv',header=None,index_col=None)
#        self.header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)

    
    def create_data_CVAE_PSN(self,):
        #1:Train fea
#        pseudoTrain_fea=pd.read_csv("Data/pseudoTrain_fea_norm.csv",header=None,index_col=None)
        pseudoTrain_fea=pd.read_csv("Data/pseudoTrain_fea.csv",header=None,index_col=None)
        train_fea=pd.read_csv("Data/03_train_features_norm.csv",header=None,index_col=None)
        all_train_fea=pd.concat([train_fea,pseudoTrain_fea],axis=0)
        #2:Train header
        all_data=pd.read_csv("Data/all_data.csv",header=None,index_col=None)
        all_data=all_data.iloc[:,0]
        pseudoHeader=pd.read_csv("Data/pseudoTrain_label.csv",header=None,index_col=None)
        all_header=pd.concat([all_data,pseudoHeader],axis=0)
#        3:Train wordembeddings:
        train_we=pd.read_csv("Data/train_pic_we.csv",header=None,index_col=None)
        pseudoTrain_we=pd.read_csv("Data/pseudoTrain_we.csv",header=None,index_col=None)
        all_train_we=pd.concat([train_we,pseudoTrain_we],axis=0)
        #save
        all_train_fea.to_csv("Data/CVAEPSN_train_fea.csv",header=None,index=None)
        all_header.to_csv("Data/CVAEPSN_all_data.csv",header=None,index=None)
        all_train_we.to_csv("Data/CVAEPSN_train_we.csv",header=None,index=None)
        return 0
        


    #Build PSN
    def model(self,):
        #Input
        left_input=Input(shape=(300,))
        right_input=Input(shape=(2048,))
        #Left-Block-01
#        left=BatchNormalization()(left_input)   
        left=Dense(512,)(left_input)
        left=BatchNormalization()(left)  
        left=Activation('relu')(left)
        #Left-Block-02
        left=Dense(1024,)(left)
        left=BatchNormalization()(left) 
        left=Activation('relu')(left)
        #Left-Block-03
        left=Dense(2048,)(left)
        left=BatchNormalization()(left) 
        left=Activation('relu')(left)
        #Right-Block-04
#        right=BatchNormalization()(right_input)
        right=Dense(2048,)(right_input)
        right=BatchNormalization()(right)
        right=Activation('relu')(right)
        #Merge=Block-05
        L1_layer=Lambda(lambda tensors : K.abs(tensors[0]-tensors[1]))
#        L1_distance=L1_layer([left,right])
        L1_distance=L1_layer([left,right_input])
        
        
        prediction=Dense(1024,)(L1_distance)
        prediction=BatchNormalization()(prediction)
        prediction=Activation('relu')(prediction)
        prediction=Dropout(0.2)(prediction)
        prediction=Dense(1,activation='tanh')(prediction)
#        prediction=Dense(1,activation='sigmoid')(prediction)
        #Output model
        model=Model(inputs=[left_input,right_input],outputs=prediction)
        adam=Adam(lr=0.001)
#        adam=Adam(lr=0.0005)
        model.compile(optimizer=adam,loss='mse')
#        model.compile(optimizer='adam',loss='mse')
#        model.compile(optimizer=adam,loss='binary_crossentropy')
#        model.compile(optimizer='adam',loss='binary_crossentropy')
        print(model.count_params())
        plot_model(model=model,to_file='Model/CVAEPSN2_regression.png',show_shapes=True)
        return model
    
    def __get_batch_data(self,batch_size):
        #n_classes=230
#        n_examples=38221
        n_examples=46000
#        n_examples=46221
        #input data
        train_we=np.asarray(self.train_we)
        train_pic=np.asarray(self.train_pic)
        all_data=np.asanyarray(self.all_data)
        header=np.asarray(self.header)
        #initialize 2 empty arrays for the input attributes and pictures
        pairs=[np.zeros(shape=(batch_size,300)),np.zeros(shape=(batch_size,2048))]
        #initialize vector for the targets 
        targets=np.zeros(shape=(batch_size,))
        for i in range(batch_size):
            idx_1=rng.randint(0,n_examples)
            #Randomly selecter training pictures
            pairs[1][i,:]=train_pic[idx_1,:].reshape(2048,)
            #First half access positive samples          
            if i < batch_size//2:
                pairs[0][i,:]=train_we[idx_1,:].reshape(300,)#positive wordembeddings
                targets[i]=1#positive targets
            #Latter half access negative samples
            else:
                #01.get train pic location
                pic_label=all_data[idx_1,0] 
                pic_loc=np.argwhere(header == pic_label)[0][0]
                #add a random number to the category to ensure difference
                idx_2=(idx_1+rng.randint(1,n_examples)) % n_examples
                #record random wordembeddings as negative sample.
                pairs[0][i,:]=train_we[idx_2,:].reshape(300,)
                #02.find the location of negative wordembeddings
                we_label=all_data[idx_2,0]
                we_loc=np.argwhere(header == we_label)[0][0]
                #Find cosine_distance_value as target according to the location of train_pic and negative wordembeddings
                """01.cosine method"""
                #Way 1 for regression
                targets[i]=self.std_we_cosine_distances.iloc[pic_loc,we_loc]
                """02.Bucket method"""
                #Way 2 for classification
#                if D <= 0.1:
#                    targets[i]=0.0
#                elif D > 0.1 and D <= 0.3:
#                    targets[i]=0.2
#                elif D > 0.3 and D <= 0.5:
#                    targets[i]=0.4
#                elif D > 0.5 and D <= 0.7:
#                    targets[i]=0.6
#                elif D > 0.7 and D <= 0.9:
#                    targets[i]=0.8
#                elif D > 0.9:
#                    targets[i]=1
                """03.0-1 method"""
                #Way 3 for classification
#                targets[i]=0
#                print(targets[i])
        return pairs,targets
    
    def training(self,):
        model=self.model()
        #Save checkpoints manually
#        weight_path="Model/psn5_regression_tanh_mse_breakpoints.h5"
        weight_path="Model/CVAEPSN2_regression_tanh_mse_breakpoints.h5"
        #Load weight if exist:
        if os.path.exists(weight_path):
            print("Load weight from "+weight_path+"...")
            model.load_weights(weight_path)
            print("Sucess！")
        #setting iteration time
#        n_iter=5000
#        n_iter=800000
        
        n_iter=1000000
#        n_iter=50000
        print("Start training...")
        for i in range(1,n_iter):
            (inputs,targets)=self.__get_batch_data(128)
            loss=model.train_on_batch(inputs,targets)
            if i % 50 == 0:
                print("iteration="+str(i)+",loss="+str(loss))
        model.save_weights(weight_path)
#        model.save("Model/PSN5_regression_tanh_mse_epoch75k.model")  
        model.save("Model/CVAEPSN2_regression_tanh_mse_epoch1005k.model")  
        return 0  
    
    def __prediction(self,):

        model=load_model("Model/CVAEPSN2_regression_tanh_mse_epoch1005k.model")
        #Input data
        header=np.asarray(self.header)
        test_pic=np.asarray(self.test_pic)
        std_we=np.asarray(self.std_we)
        #record num
        pic_num=test_pic.shape[0]##14633
        class_num=std_we.shape[0]
        distances=np.zeros(shape=(pic_num,class_num))
        print("start predcting...")
        for i in range(pic_num):
            if i % 10 == 0:
                print("预测完第"+str(i)+"张图片.")
            for j in range(class_num):
                distances[i,j]=model.predict([std_we[j,:].reshape((1,300)),test_pic[i,:].reshape(1,2048)],batch_size=5120)
        predict_dict=pd.DataFrame(distances)
#        predict_dict.to_csv("Data/PSN5_regression_Predict_cosine_distances_epoch75k.csv",header=None,index=None)
        predict_dict.to_csv("DataPrediction/CVAEPSN2_regression_predict_cosine_distances_epoch1005k.csv",header=None,index=None)
        #Find out the location of min distance.
        names=[]
        for i in range(pic_num):
            location=np.argmax(distances[i])
            name=header[location]
            names.append(name)
        return names
    
#    def temp(self,):
#        pic_num=14633
#        header=np.asarray(self.header)
#        distances=pd.read_csv("Data/Predict_cosine_distances_for_check_epoch80_2k.csv",header=None,index_col=None)
#        distances=np.asarray(distances)
#        names=[]
#        for i in range(pic_num):
#            location=np.argmax(distances[i])
#            name=header[location]
#            names.append(name)
#        return names
        
    def __output_submitting(self,predict_class_name):
        dp=DataProcessing()
        #Pic name
        predict_pic_name, _ = dp.read_pictures_to_array(path='DatasetA_test_20180813/test/')
        predict_pic_name=pd.DataFrame(predict_pic_name)
        #Class name
        class_name=pd.DataFrame(predict_class_name)
        #Concat pic name and class name into one file.
        result=pd.concat([predict_pic_name,class_name],axis=1)
        result.to_csv("Submit/submit_180919_CVAEPSN2_epoch1005k.csv",header=None,index=None)
#        result.to_csv("Submit/submit_180910_psn5_regression_epoch75k.csv",header=None,index=None)
        return 0
        
    def predict_and_submit(self,):
        names=self.__prediction()
        self.__output_submitting(predict_class_name=names)
        return 0
    
if __name__=='__main__':
    psn=PSN()
#    psn.training()
#    print("End training!")
    psn.predict_and_submit()
#    psn.create_data_CVAE_PSN()
    
    
        
