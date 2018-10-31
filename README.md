## 2018ZJL-Zero-Shot-Learning-ZSL
### 2018-TIANCHI Zero-shot Learning Competition.

**1.URL**  
https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.6acd33afWW2JsH&raceId=231677

**2.Requirements**  
    - python >= 3.6  
    -keras >= 2.2.4  
    scikit-learn >= 0.19.1  
    opencv-python >= 3.2.0.6  
    gensim >= 3.6.0  
    pandas  
    numpy  
    scipy  
    PIL  

**3.Modules**  
    `tool_utils.py` :for all data processing;  
    `tool_wordembeddings.py`:loading trained word-embeddings of label-list,source：https://github.com/xgli/word2vec-api  
    `model_AE.py`:auto-encoder for class-wordembeddings;  
    `model_VGG.py`:image features extraction；  
    `model_PSN.py`:Pseudo-Siamese Network(PSN),using cosine-distance of class-wordembeddings as output label;  
    `model_SAE.py`:Semantic Auto-encoder(SAE);  
    `model_CVAE.py`:Conditional Variational Auto-encoders(CVAE);  
    `model_AttClassifiers.py`:Attributes classifiers;  
    `tool_distance.py`:Input std_we and prediction_we,calculating cosine distances;  
    `model_CS.py`:Calibration stacking algorithm(CS),input 'predict-cosine-distances.csv',output 'submit.csv';  
    `tool_testing.py`:for testing.  
    
    
    

