# -*- coding: utf-8 -*-
"""Used to analyze submit.csv"""
"""Input submit.csv and label name(such as ZJL1),output pictures in folder named 'testing'"""
import pandas as pd
from PIL import Image
import os
class ManualTesting():
    def __init__(self,submit_file,label_name):
        #input submit file name and label name for testing
        self.submit_file=submit_file
        self.label_name=label_name
        self.test_pic_path="DatasetA_test_20180813/test/"
        self.save_folder="testing/"
        self.label_list="DatasetA_test_20180813/label_list.txt"
        self.train="DatasetA_train_20180813/train.txt"
    #Create empty folder and remove pictures in folder everytime
    def __create_empty_folder(self,folder_name="testing/"):
        #create new file folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        #Empty pictures in folder named testing 
        else:
            for i in os.listdir(folder_name):
                os.remove(folder_name+i)
        return 0
    #Search pictures
    def __search_pic_through_label(self,label_name):
        submit=pd.read_csv(self.submit_file,)
        num=submit.shape[0]
        record=[]
        for i in range(num):
            if submit.iloc[i,1] == str(label_name):
                record.append(submit.iloc[i,0])
        print(str(len(record))+" pictures named "+label_name+" in submit_csv.")
        return record
    #Save picture to folder named testing
    def __save_pic_to_folder(self,pic_name_settings):
        for i in pic_name_settings:
            im=Image.open(self.test_pic_path+i)
            im.save(self.save_folder+i,)
        return 0
    #Output class name
    def __output_label_name(self,):
        label_list=pd.read_csv(self.label_list)
        num=label_list.shape[0]
        for i in range(num):
            if self.label_name in label_list.iloc[i,0]:
                print(label_list.iloc[i,0])
        return 0
    #Judge whether in the training set
    def __judge_whether_in_train_set(self,):
        train_set=pd.read_table(self.train,header=None,index_col=None)
        num=train_set.shape[0]
        for i in range(num):
            if self.label_name in train_set.iloc[i,1]:
                print("Exist in training set,",i)
                break
        
    #Testing  
    def testing(self,):
        self.__create_empty_folder()
        pic_name_settings=self.__search_pic_through_label(self.label_name)
        self.__output_label_name()
        self.__save_pic_to_folder(pic_name_settings)
        self.__judge_whether_in_train_set()
        return 0
    
    #Fast evaluation
    def __fast_testing(self,submit_file,label_list):
        seen_label=label_list
        submit=pd.read_csv(submit_file,header=None,index_col=None)
        test_num=submit.shape[0]
#        label_num=seen_label.shape[0]
        label_num=len(seen_label)
        nums=num=0
        for i in range(label_num):
            for j in range(test_num):
                if submit.iloc[j,1] == str(seen_label[i]):
                    num+=1
            print("There have "+str(num)+" "+str(seen_label[i])+" in submit")
            nums=nums+num
            num=0
        print("Totally have "+str(nums)+" "+str(seen_label)+" in submit")
        return 0
    
    def calibrated_stacking_evaluation(self,submit_file):
        seen=['ZJL160','ZJL163','ZJL169','ZJL172','ZJL176','ZJL178','ZJL182','ZJL183','ZJL184','ZJL196','ZJL197']
        
        unseen=['ZJL200','ZJL201','ZJL202','ZJL203','ZJL204','ZJL205','ZJL206',
                'ZJL207','ZJL208','ZJL209','ZJL210','ZJL211','ZJL212','ZJL213',
                'ZJL214','ZJL215','ZJL216','ZJL217','ZJL218','ZJL219','ZJL220',
                'ZJL221','ZJL222','ZJL223','ZJL224','ZJL225','ZJL226','ZJL227',
                'ZJL228','ZJL229','ZJL230','ZJL231','ZJL232','ZJL233','ZJL234',
                'ZJL235','ZJL236','ZJL237','ZJL238','ZJL239','ZJL240']
        #part of Seen label
        self.__fast_testing(submit_file=submit_file,label_list=seen)
        #Unseen label
        self.__fast_testing(submit_file=submit_file,label_list=unseen)
                
    
if __name__=='__main__':

    mt=ManualTesting("Submit/submit.csv","ZJL172")
#    mt=ManualTesting("Submit/submit_rate0_1.csv","ZJL234")
#    mt.testing()
    mt.calibrated_stacking_evaluation(submit_file="Submit/submit.csv")
#    mt.calibrated_stacking_evaluation(submit_file="Submit/submit_rate0_1.csv")

    
            
