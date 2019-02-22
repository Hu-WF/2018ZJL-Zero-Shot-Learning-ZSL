# -*- coding: utf-8 -*-
"""For data processing"""
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import Normalizer
import shutil
from scipy.spatial.distance import cosine

"""1.Load raw data together before processing"""
class LoadRawData():
    def __init__(self,):
        #folder name
        self.folder='Data/'
        #txt file path
        self.__path_attribute_list='DatasetB_20180919/attribute_list.txt'
        self.__path_attributes_per_class='DatasetB_20180919/attributes_per_class.txt'
        self.__path_label_list='DatasetB_20180919/label_list.txt'
        self.__path_train='DatasetB_20180919/train.txt'
        self.__path_class_wordembeddings='DatasetB_20180919/class_wordembeddings.txt'
        #train picture file path
        self.__path_train_pic='DatasetB_20180919/train/'
        #Saved as Dataframe
        self.att_list=pd.read_table(self.__path_attribute_list,header=None,index_col=None)
        self.att_per_class=pd.read_table(self.__path_attributes_per_class,header=None,index_col=None)
        self.label_list=pd.read_table(self.__path_label_list,header=None,index_col=None)
        self.train=pd.read_table(self.__path_train,header=None,index_col=0)
        self.class_wordembeddings=pd.read_table(self.__path_class_wordembeddings,header=None,index_col=None)

    def __create_empty_folder(self,):
        folder_name=self.folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        return 0
    
    def __rawdata_to_csv(self,):
        self.att_list.to_csv(self.folder+"att_list.csv",header=None,index=None)
        self.att_per_class.to_csv(self.folder+"att_per_class.csv",header=None,index=None)
        self.label_list.to_csv(self.folder+"label_list.csv",header=None,index=None)
        #Attention: train must preserve index(index should not equals to None)
        self.train.to_csv(self.folder+"train.csv",header=None)
        return 0
    #Processing inital wordembedding data
    def __wordembedding_to_csv(self,):
        with open (self.__path_class_wordembeddings,'r') as f:
            lines=f.readlines()
        result=[]
        for line in lines:
            data=line.split()
            result.append(data)
        result=pd.DataFrame(result)
        result.to_csv(self.folder+"class_wordembeddings.csv",header=None,index=None)
        return 0
    
    def __wordembedding_to_csv_and_scale(self,):
        with open (self.__path_class_wordembeddings,'r') as f:
            lines=f.readlines()
        result=[]
        for line in lines:
            data=line.split()
            result.append(data)
        result=pd.DataFrame(result)
        #Normalize
        header=result.iloc[:,0]
        we=result.iloc[:,1:]
        nm=Normalizer()
        we_scaled=nm.fit_transform(we)
        we_scaled=pd.DataFrame(we_scaled)
        result=pd.concat([header,we_scaled],axis=1)
#        result.to_csv(self.folder+"class_wordembeddings_norm.csv",header=None,index=None)
        result.to_csv(self.folder+"class_wordembeddings.csv",header=None,index=None)
        return 0
    #Load all datas together
    def load_data_to_csv(self,):
        self.__create_empty_folder()
        self.__rawdata_to_csv()
#        self.__wordembedding_to_csv()
        self.__wordembedding_to_csv_and_scale()
        return 0
    
    
class DataProcessing():
    """2.Process training and testing data"""
    def __init__(self,):
        self.folder='Data/'
        self.__train_pic_path='DatasetB_20180919/train/'
        self.train=pd.read_csv(self.folder+"train.csv",header=None,index_col=0)
        self.label_list=pd.read_csv(self.folder+"label_list.csv",header=None,index_col=None)
    #load pictures and picture names in file_path as two array.
    def read_pictures_to_array(self,path,):
        #Alphabetical order
        imgs=os.listdir(path)
        num=len(imgs)
        pic_array=np.empty((num,64,64,3),dtype='float32')
        pic_name=[]
        for i in range(num):
            img=Image.open(path+imgs[i])
            #convert to an array
            arr=np.asarray(img,dtype='float32')
            #对于单通道图像,则手动扩充为3通道图像(暂定缺失值处理方式).
            if arr.shape == (64,64):
                arr=np.expand_dims(arr,axis=2)
            #save picture array
            arr=(arr/255)#Scale
            pic_array[i]=arr
            #save picture name
            pic_name.append(imgs[i])
            #draw for test
            #pylab.imshow(arr/255)  
        return pic_name,pic_array
    #get the training y with training pictures
    def get_training_x_with_y(self,train_pic_name_order,y="pca.csv"):
        train_pic_name=pd.DataFrame(train_pic_name_order)
        result=pd.merge(train_pic_name,self.train,left_on=0,right_on=0,left_index=True)
        #how='left'
        y=pd.read_csv(y,header=None,index_col=None)
        #先补齐
        header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
        y=pd.concat([header,y],axis=1)
        y.to_csv("y.csv",header=None,index=None)
        y=pd.read_csv("y.csv",header=None,index_col=None)
        print(y)
        result=pd.merge(result,y,left_on=1,right_on=0,how='left',left_index=True)
        #Save the part of attributes
        result=result.iloc[:,4:]
        result.to_csv(self.folder+"train_pic_y.csv",header=None,index=None)
        result=np.array(result)
        return result
    #get the training attributes with training pictures
    def get_training_x_att_y(self,train_pic_name_order,att="Data/att_per_class.csv"):
        train_pic_name=pd.DataFrame(train_pic_name_order)
        result=pd.merge(train_pic_name,self.train,left_on=0,right_on=0,left_index=True)
        #how='left'
        att=pd.read_csv(att,header=None,index_col=None)
        result=pd.merge(result,att,left_on=1,right_on=0,how='left',left_index=True)
        #Save the part of attributes
        result=result.iloc[:,4:]
        result.to_csv(self.folder+"train_pic_att.csv",header=None,index=None)
        result=np.array(result)
        return result
    #Get the training wordembedding with training pictures
    ##train_pic_name_order>>>train>>>label_list>>>wordembeddings
    def get_training_x_wordembedding_y(self,train_pic_name_order,we="class_wordembeddings.csv"):
        #fusing four DF:
        train_pic_name_order=pd.DataFrame(train_pic_name_order)
        train=self.train
        #print(train)
        label_list=self.label_list
        class_word_embeddings=pd.read_csv(self.folder+we,header=None,index_col=None)
        result=pd.merge(train_pic_name_order,train,left_on=0,right_on=0,how='left',left_index=True)
        #print(result)
        result=pd.merge(result,label_list,left_on=1,right_on=0,how='left',left_index=True)
        result.columns=['a','b','c','d','e']
        result.convert_objects(convert_numeric=True)
        result=pd.merge(result,class_word_embeddings,left_on='e',right_on=0,how='left',left_index=True)
        #output all data for checking：
        result.to_csv("Data/all_data.csv",header=None,index=None)        
        #left=result.iloc[:,1]
        #Output wordembedding
        right=result.iloc[:,6:]
        #print(right)
        right.to_csv(self.folder+"train_pic_we.csv",header=None,index=None)
        return right
    #get 230 wordembeddings
    def get_std_we(self,we="class_wordembeddings.csv"):
        label_list=self.label_list
        wordembedding=pd.read_csv(self.folder+we,header=None,index_col=None)
        result=pd.merge(label_list,wordembedding,left_on=1,right_on=0,how='left',left_index=True)
        header=result.iloc[:,1]
        header.to_csv(self.folder+"std_header.csv",header=None,index=None)
        result=result.iloc[:,4:]
        print(result)
        result.to_csv(self.folder+"std_we.csv",header=None,index=None)
        return 0
    #get 230 attributes
    def get_std_att(self,):
        att=pd.read_csv("Data/att_per_class.csv",header=None,index_col=None)
        att=att.iloc[:,1:]
        att.to_csv("Data/std_att.csv",header=None,index=None)
        return 0
    #fusing train attributes and wordembeddings.
    def fuse_train_att_and_we(self,att="Data/train_pic_att.csv",we="Data/train_pic_we.csv"):
        att=pd.read_csv(att,header=None,index_col=None)
        we=pd.read_csv(we,header=None,index_col=None)
        fuse=pd.concat([att,we],axis=1)
        fuse.to_csv(self.folder+"fuse_train_att_we.csv",header=None,index=None)
        return 0
    #fusing 230 train attributes and wordembeddings
    #att_per_class>>>label_list>>>wordembedding
    def fuse_std_att_and_we(self,att="Data/att_per_class.csv",we="Data/class_wordembeddings.csv"):
        att=pd.read_csv(att,header=None,index_col=None)
        label_list=self.label_list
        we=pd.read_csv(we,header=None,index_col=None)
        result=pd.merge(att,label_list,left_on=0,right_on=0,how='left',left_index=True)
        #print(result)
        result=pd.merge(result,we,left_on='1_y',right_on=0,how='left',left_index=True)
        #print(result)
        result.to_csv("fuse_std_att_we.csv",header=None,index=None)
        left=result.iloc[:,1:31]
        right=result.iloc[:,33:]
        result=pd.concat([left,right],axis=1)
        result.to_csv(self.folder+"fuse_std_att_we.csv",header=None,index=None)
        return 0
    #Data normalizer for all kinds of data.
    def data_normalizer(self,data,save_name):
        data=pd.read_csv(data,header=None,index_col=None)
        nm=Normalizer()
        data=nm.fit_transform(data)
        data=pd.DataFrame(data)
        data.to_csv(save_name,header=None,index=None)
        return 0
    #Data normalizer for train and test data together.
    def data_normalizer_train_test_together(self,train_data,test_data,train_save,test_save):
        #Data
        train=pd.read_csv(train_data,header=None,index_col=None)
        test=pd.read_csv(test_data,header=None,index_col=None)
        nm=Normalizer()
        #transform,fit with train data,then transform it to test data!
#        train=nm.fit_transform(train)
#        test=nm.transform(test)
        test=nm.fit_transform(test)
        train=nm.transform(train)
        #save
        train=pd.DataFrame(train)
        test=pd.DataFrame(test)
        train.to_csv(train_save,header=None,index=None)
        test.to_csv(test_save,header=None,index=None)
        return 0
    #Data normalizer for data with header
    def data_normalizer_with_header(self,data,save_name):
        data=pd.read_csv(data,header=None,index_col=None)
        header=data.iloc[:,0]
        data=data.iloc[:,1:]
        nm=Normalizer()
        data=nm.fit_transform(data)
        data=pd.DataFrame(data)
        data=pd.concat([header,data],axis=1)
        data.to_csv(save_name,header=None,index=None)
        return 0
    #For PSN
    def calculate_std_cosine_distances(self,data,save_name):
        we=pd.read_csv(data,header=None,index_col=None)
        we=np.asarray(we)
        num=we.shape[0]
        distances=np.zeros(shape=(num,num))
        for i in range(num):
            for j in range(num):
                #result=1-cosine,result=1 means that they are totally the same.
                distances[i,j]=1-cosine(we[i,:],we[j,:])
        distances=pd.DataFrame(distances)
        distances.to_csv(save_name,header=None,index=None)
        return 0
    #For PSN
    def calculate_std_cosine_distances_using_we(self,):
        we=pd.read_csv('Data/std_we.csv',header=None,index_col=None)
        we=np.asarray(we)
        num=we.shape[0]
        distances=np.zeros(shape=(num,num))
        for i in range(num):
            for j in range(num):
                #result=1-cosine,result=1 means that they are totally the same.
                distances[i,j]=1-cosine(we[i,:],we[j,:])
        distances=pd.DataFrame(distances)
        distances.to_csv("Data/std_we_cosine_distances.csv",header=None,index=None)
        return 0
    #For PSN
    def calculate_std_cosine_distances_using_att(self,):
        we=pd.read_csv('Data/std_att.csv',header=None,index_col=None)
        we=np.asarray(we)
        num=we.shape[0]
        distances=np.zeros(shape=(num,num))
        for i in range(num):
            for j in range(num):
                #result=1-cosine,result=1 means that they are totally the same.
                distances[i,j]=1-cosine(we[i,:],we[j,:])
        distances=pd.DataFrame(distances)
        distances.to_csv("Data/std_att_cosine_distances.csv",header=None,index=None)
        return 0
    #Create onehot label
    def get_std_onehot(self,):
        I=np.eye(230,k=0)
        print(I)
        I=pd.DataFrame(I)
        I.to_csv("Data/std_onehot_without_header.csv",header=None,index=None)
        #import class header
        header=pd.read_csv("Data/label_list.csv",header=None,index_col=None)
        header=header.iloc[:,0]
        #concat I and header
        I=pd.concat([header,I],axis=1)
        I.to_csv("Data/std_onehot.csv",header=None,index=None)
        return 0
    #create onehot train
    def get_training_x_onehot_y(self,train_pic_name_order,onehot="Data/std_onehot.csv"):
        train_pic_name=pd.DataFrame(train_pic_name_order)
        result=pd.merge(train_pic_name,self.train,left_on=0,right_on=0,left_index=True)
        #how='left'
        onehot=pd.read_csv(onehot,header=None,index_col=None)
        result=pd.merge(result,onehot,left_on=1,right_on=0,how='left',left_index=True)
        #Save the part of attributes
        result=result.iloc[:,4:]
        result.to_csv(self.folder+"train_pic_onehot.csv",header=None,index=None)
        result=np.array(result)
        return result
    #Create 0,1,2... label
    def get_std_num(self,):
        num=list(range(230))
        num=pd.DataFrame(num)
        #print(num)
        num.to_csv("Data/std_num_without_header.csv",header=None,index=None)
        #import class header
        header=pd.read_csv("Data/label_list.csv",header=None,index_col=None)
        header=header.iloc[:,0]
        #concat I and header
        num=pd.concat([header,num],axis=1)
        num.to_csv("Data/std_num.csv",header=None,index=None)
        return 0
    #create 0,1,2,3... train
    def get_training_x_num_y(self,train_pic_name_order,num="Data/std_num.csv"):
        train_pic_name=pd.DataFrame(train_pic_name_order)
        result=pd.merge(train_pic_name,self.train,left_on=0,right_on=0,left_index=True)
        #how='left'
        num=pd.read_csv(num,header=None,index_col=None)
        result=pd.merge(result,num,left_on=1,right_on=0,how='left',left_index=True)
        #Save the part of attributes
        result=result.iloc[:,4:]
        result.to_csv(self.folder+"train_pic_num.csv",header=None,index=None)
        result=np.array(result)
        return result
        

        
        
        

 
if __name__=='__main__':
##    #Load raw data
#    lrd=LoadRawData()
#    lrd.load_data_to_csv()
#    #Data processing
    dp=DataProcessing()
    pic_name,pic_array=dp.read_pictures_to_array(path='DatasetB_20180919/train/')
#    dp.get_training_x_with_y(train_pic_name_order=pic_name,y="Data/std_pca.csv")
#    dp.get_training_x_wordembedding_y(train_pic_name_order=pic_name,)  
#    dp.get_training_x_att_y(train_pic_name_order=pic_name,)
#    dp.get_std_we()
#    dp.get_std_att()
###    dp.fuse_std_att_and_we()
###    dp.fuse_train_att_and_we()
#    dp.calculate_std_cosine_distances_using_we()
###    dp.calculate_std_cosine_distances_using_att()
##    dp.get_std_num()
##    dp.get_training_x_num_y(train_pic_name_order=pic_name)
#    
    dp.get_std_onehot()
    dp.get_training_x_onehot_y(train_pic_name_order=pic_name)
    
    
    
#if __name__=='__main__':
#    dp=DataProcessing()
#    dp.data_normalizer(data="Data/all_pseudoTrain_fea.csv",save_name="Data/all_pseudoTrain_fea_norm.csv")
#    dp.data_normalizer(data="Data/pseudoTrain_fea.csv",save_name="Data/pseudoTrain_fea_norm.csv")
    
    
    
#    dp.data_normalizer(data="Data/01_train_fea2048_att.csv",save_name="Data/03_train_fea2048_att_norm.csv")
#    dp.data_normalizer(data="Data/02_test_fea2048_att.csv",save_name="Data/04_test_fea2048_att_norm.csv")
    
#    dp.data_normalizer(data="Data/05_train_fea2048.csv",save_name="Data/09_train_fea2048_norm.csv")
#    dp.data_normalizer(data="Data/06_test_fea2048.csv",save_name="Data/10_test_fea2048_norm.csv")
#    
#    dp.data_normalizer(data="Data/05_train_fea2048_att.csv",save_name="Data/07_train_fea2048_att_norm.csv")
#    dp.data_normalizer(data="Data/06_test_fea2048_att.csv",save_name="Data/08_test_fea2048_att_norm.csv")
    
#    dp.data_normalizer_train_test_together(train_data="Data/01_train_features.csv",
#                                           test_data="Data/02_test_features.csv",
#                                           train_save="Data/05_train_fea2048_norm_together.csv",
#                                           test_save="Data/06_test_fea2048_norm_together.csv")

    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    