# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def processing():
    train=pd.read_csv("Data/train.csv",header=None,index_col=None)
    train=np.asarray(train.iloc[:,1])
    seen=np.unique(train)
    print(seen.shape,seen)

    unseen=[]
    header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
    header=np.asarray(header.iloc[:,0])
    for i in range(header.shape[0]):
        if header[i] not in seen:
            unseen.append(header[i])
    print(unseen)
    
    std_we=pd.read_csv("Data/std_we.csv",header=None,index_col=None)
#    std_we=np.asarray(std_we)
    num=len(unseen)
    unseen_we=np.zeros(shape=(num,300))
    print(std_we)
    for i in range(num):
        loc=np.argwhere(unseen[i] == header)[0][0]
        unseen_we[i,:]=std_we.iloc[loc,:]
        
    unseen=pd.DataFrame(unseen)
    unseen_we=pd.DataFrame(unseen_we)
    unseen_output=pd.concat([unseen,unseen_we],axis=1)
    unseen_output.to_csv("Data/unseen_we_vs_label.csv",header=None,index=None)   
    
    return 0

#def create_onehot_label():
#    header=pd.read_csv("Data/std_header.csv",header=None,index_col=None)
#    
#    num=header.shape[0]
#    
#    label=np.eye(num)
#    print(label)

def creat_pseudotrain_att():
    att_per_class=pd.read_csv("Data/att_per_class.csv",header=None,index_col=None)
    pse_train_label=pd.read_csv("pseudotrainlabel.csv",header=None,index_col=None)
    
    header=att_per_class.iloc[:,0]
    std_att=att_per_class.iloc[:,1:]
    print(header,std_att)
    
    num=pse_train_label.shape[0]
    pse_train_label=np.asarray(pse_train_label)
    atts=np.zeros(shape=(num,30))
    for i in range(num):
        print(pse_train_label[i][0])
#        loc=np.argwhere(pse_train_label[i] == att_per_class.iloc[:,0])[0][0]
        loc=np.argwhere(pse_train_label[i][0] == header)[0][0]
        atts[i]=att_per_class.iloc[loc,1:]
    atts=pd.DataFrame(atts)
    atts.to_csv("pseudoTrainAtt.csv",header=None,index=None)
    return 0


def fuse_pseudotrain():
    pass


if __name__=='__main__':
#    processing()
    
    creat_pseudotrain_att()
    
    
    