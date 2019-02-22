# -*- coding: utf-8 -*-
"""Input trained model and test data,output submit.csv"""
from keras.models import load_model
from tool_utils import DataProcessing
import pandas as pd
import numpy as np
import heapq
from collections import Counter


class Prediction():
    def __init__(self,):
        self.dp=DataProcessing()
        #load predict dataset at begining
        self.predict_pic_name,self.predict_pic=self.dp.read_pictures_to_array(path='DatasetA_test_20180813/test/')
        self.header=pd.read_csv('Data/std_header.csv',header=None,index_col=None)
    
    #output the prediction result of attribute
    def we_prediction(self,):
        #load model
        model=load_model('Model/VGG11.model')
        #predicting
        result=model.predict(self.predict_pic,batch_size=64)
        #print(result.shape)
        result=pd.DataFrame(result)
        #save the prediction result of attribute
        result.to_csv("prediction.csv",header=None,index=None)
        return 0
    
    ##calculate euclidean distance,out put the class name which get the min distance
    def euclidean_distance_classifier(self,std,prediction):
        ####-----------------read in standard att per class----------------####
        std_we=pd.read_csv(std,header=None,index_col=None)
        #Filtering the first column of data
#        std_we=std_we.iloc[:,1:]
        #print(true_att)
        #convert to array
        std_we=np.array(std_we)
        ####-------------------read in predict att-------------------------####
        predict_we=pd.read_csv(prediction,header=None,index_col=None)
        #convert to array
        predict_we=np.array(predict_we)
        ##create an empty array to save distance result
        dist_result=np.zeros(shape=(14633,230))
        ####----------------Calculate distance by numpy--------------------####
        class_num=std_we.shape[0]#230
        predict_num=predict_we.shape[0]#14633
        print(class_num)
        print(predict_num)
        i=j=0
        while i < predict_num:
            while j< class_num:
                #calculate euclidean distance
                dist=np.sqrt(np.sum(np.square(predict_we[i,:]-std_we[j,:])))
                dist_result[i][j]=dist
                j+=1
            j=0
            i+=1
        #print(dist_result)
        dist_result=pd.DataFrame(dist_result)
        dist_result.to_csv("DataPrediction/distance.csv",header=None,index=None)
        dist_result=np.array(dist_result)
        #find out the location of the min distance in each line.
        positions=[]
        for i in range(predict_num):
            num=np.argmin(dist_result[i])
            positions.append(num)
        #print(positions)
        #get the class name
        class_name=[]
        label_per_class=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
#        att_per_class.to_csv("att_per_class.csv")
        for position in positions:
            name=label_per_class.iloc[position,0]
            class_name.append(name)
        return class_name
    
    ##calculate euclidean distance,out put the class name which get the min distance
    def cosine_distance_classifier(self,std,predict):
        ####-----------------read in standard att per class----------------####
#        std_we=pd.read_csv("Data/std_we_cosine_distances.csv",header=None,index_col=None)
        std_we=pd.read_csv(std,header=None,index_col=None)
        #Filtering the first column of data
#        std_we=std_we.iloc[:,1:]
        #print(true_att)
        #convert to array
        std_we=np.array(std_we)
        ####-------------------read in predict att-------------------------####
#        predict_we=pd.read_csv("prediction.csv",header=None,index_col=None)
        predict_we=pd.read_csv(predict,header=None,index_col=None)
        #convert to array
        predict_we=np.array(predict_we)
        ##create an empty array to save distance result
        dist_result=np.zeros(shape=(14633,230))
        ####----------------Calculate distance by numpy--------------------####
        class_num=std_we.shape[0]#230
        predict_num=predict_we.shape[0]#14633
        print(class_num)
        print(predict_num)
        i=j=0
        while i < predict_num:
            while j< class_num:
                #calculate euclidean distance
#                dist=np.sqrt(np.sum(np.square(predict_we[i,:]-std_we[j,:])))
                X=predict_we[i,:]
                Y=std_we[j,:]
                ###计算余弦相似度
                cosine=np.sum(X*Y)/(np.sqrt(np.sum(np.square(X))) * np.sqrt(np.sum(np.square(Y))))
#                print(cosine)
#                print(cosine.shape)
                dist_result[i][j]=cosine
                j+=1
            j=0
            i+=1
        #print(dist_result)
        dist_result=pd.DataFrame(dist_result)
        dist_result.to_csv("DataPrediction/distance.csv",header=None,index=None)
        dist_result=np.array(dist_result)
        #find out the location of the min distance in each line.
        positions=[]
        for i in range(predict_num):
            ###取余弦相似度最大值为最相近！！！
            num=np.argmax(dist_result[i])
            positions.append(num)
        #print(positions)
        #get the class name
        class_name=[]
        label_per_class=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
#        att_per_class.to_csv("att_per_class.csv")
        for position in positions:
            name=label_per_class.iloc[position,0]
            class_name.append(name)
        return class_name
    
    
    ##input pic_name,predict_class_name;output file named submit.csv.
    def output_submitting(self,pic_name,predict_class_name):
#        predict_pic_name, _ =dp.read_pictures_to_array(path='DatasetA_test_20180813/test/') 
        predict_pic_name=pd.DataFrame(pic_name)
        class_name=pd.DataFrame(predict_class_name)
        ##concat into one file.
        result=pd.concat([predict_pic_name,class_name],axis=1)
        result.to_csv("Submit/submit.csv",header=None,index=None)
        return 0
    
    
    def combine_predict(self,distance_1,distance_2):
        s1=pd.read_csv(distance_1,header=None,index_col=None)
        s2=pd.read_csv(distance_2,header=None,index_col=None)
        s=(s1+s2)/2
        s.to_csv("DataPrediction/distance_combine.csv",header=None,index=None)
        s=np.array(s)
        #find out the location of the min distance in each line.
        positions=[]
        for i in range(14633):
            ###取余弦相似度最大值为最相近！！！
            num=np.argmax(s[i])
            positions.append(num)
        #print(positions)
        #get the class name
        class_name=[]
        label_per_class=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
#        att_per_class.to_csv("att_per_class.csv")
        for position in positions:
            name=label_per_class.iloc[position,0]
            class_name.append(name)
        return class_name       
    

if __name__=='__main__':
    pre=Prediction()
#    pre.we_prediction()
#    predict_class_name=pre.euclidean_distance_classifier(std="Data/std_we.csv",prediction="DataPrediction/dnn_prediction.csv")

    predict_class_name=pre.cosine_distance_classifier(std="Data/std_we.csv",predict="DataPrediction/dnn_prediction.csv")
    
#    predict_class_name=pre.cosine_distance_classifier(std="Data/std_fea_mean_pca.csv",predict="DataPrediction/pcadnn_prediction.csv")
#    predict_class_name=pre.euclidean_distance_classifier(std="Data/std_fea_mean_pca.csv",prediction="DataPrediction/pcadnn_prediction.csv")
    
#    predict_class_name=pre.cosine_distance_classifier()
#    predict_class_name=pre.ensemble_knn_classifier(fea_n=1,k=2)
    pre.output_submitting(pic_name=pre.predict_pic_name,predict_class_name=predict_class_name)   
    

#if __name__=='__main__':
#    pre=Prediction()
#
#    predict_class_name=pre.combine_predict(distance_1="DataPrediction/distance_1.csv",distance_2="DataPrediction/distance_2.csv")
#
#    pre.output_submitting(pic_name=pre.predict_pic_name,predict_class_name=predict_class_name)   

    
    

       
   

    

    
    
