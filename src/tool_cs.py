# -*- coding: utf-8 -*-
"""Adjusting predicion cosine distance using calibration stacking algorithm"""
"""Input predict cosine distances from model_PSN.py ,calculate with CS and output submit.csv"""
import pandas as pd
import numpy as np
from tool_utils import DataProcessing

class CalibratedStacking():
    def __init__(self,):
        #Input raw predict cosine distances.
#        self.__input_predictions=pd.read_csv("DataPrediction/dnn_prediction.csv",header=None,index_col=None)
#        self.__input_predictions=pd.read_csv("DataPrediction/dnn_prediction.csv",header=None,index_col=None)
#        self.__input_predictions=pd.read_csv("DataPrediction/CVAEPSN_classification_predict_cosine_distances_epoch1005k.csv",header=None,index_col=None)
        self.__input_predictions=pd.read_csv("DataPrediction/distance.csv",header=None,index_col=None)
        self.__output_predictions_name="DataPrediction/CS_cosine_distances.csv"
        #Load train data to find out the location of seen class in training
        self.__header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
        self.__train_data=pd.read_csv("Data/train.csv",header=None,index_col=None)
        
    def __converting(self,rate=0.1):
        distances=self.__input_predictions
        header=np.asarray(self.__header)
        #Record the location of seen class label in train pictures.
        train=np.asarray(self.__train_data.iloc[:,1])
        train=np.unique(train)
        print(train.shape,train)
        num=train.shape[0]
        locs=[]
        for i in range(num):
            loc=np.argwhere(header == train[i])[0][0]
            locs.append(loc)
        print("Seen class locations:",locs)
        #Start converting old into new cosine distance prediction.
        for i in range(num):
            loc=locs[i]
            distances.iloc[:,loc]=distances.iloc[:,loc] - rate 
        distances.to_csv(self.__output_predictions_name,header=None,index=None)
        return 0

    def __max_distance_names(self,):
        pic_num=14633
        header=np.asarray(self.__header)
        distances=pd.read_csv(self.__output_predictions_name,header=None,index_col=None)
        distances=np.asarray(distances)
        names=[]
        for i in range(pic_num):
            location=np.argmax(distances[i])
            name=header[location]
            names.append(name)
        return names
        
    def __output_submitting(self,predict_class_name):
        dp=DataProcessing()
        #Pic name
        predict_pic_name, _ = dp.read_pictures_to_array(path='DatasetA_test_20180813/test/')
        predict_pic_name=pd.DataFrame(predict_pic_name)
        #Class name
        class_name=pd.DataFrame(predict_class_name)
        #Concat pic name and class name into one file.
        result=pd.concat([predict_pic_name,class_name],axis=1)
        result.to_csv("Submit/submit_rate0_1.csv",header=None,index=None)
#        result.to_csv("Submit/submit_180913_PSN7_rate0_1.csv",header=None,index=None)
        return 0
    
    def convert_and_submit(self,rate):
        self.__converting(rate=rate)
        names=self.__max_distance_names()
        self.__output_submitting(predict_class_name=names)
        return 0
    

if __name__=="__main__":
    cs=CalibratedStacking()
    cs.convert_and_submit(rate=0.1)
    
    
    
