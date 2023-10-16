#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:41:36 2019

@author: malabonline
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
# import imblearn
import os
import scipy.io
from tensorflow.keras import optimizers
# from attention_layer_paper import attention_Layer
import cnn_classifier as CNN
from tensorflow.keras.models import load_model, Model




def LoadFeatMatFile(FileName):

    # File = Path + '/' + FileName
    
    FeatMatrix = scipy.io.loadmat(FileName)
    keyNam = list(FeatMatrix.keys())[3]
    # print('keyNam :', keyNam)

    Feat = FeatMatrix[keyNam]

    return Feat


def get_files_list(path):

    file_list = {}
    
    # className = os.listdir(path)
    className = [f.path for f in os.scandir(path) if f.is_dir()]
    
    for clasi in range(len(className)):
        className[clasi] = className[clasi].split('/')[-1]
        path_cls = path + '/' +className[clasi] + '/'
        

        file_list[className[clasi]] = []
        # test_files[className[clasi]] = []  
        
        path_cls = [f.path for f in os.scandir(path_cls) if f.is_dir()]
        # --- for train data
        for foldi in range(np.size(path_cls)):
            trn_path =  path_cls[foldi] 
            # gender = os.listdir(trn_path)
            
            # for gendi in range(np.size(gender)):
                # gender_path = trn_path + '/' + gender[gendi]
            files = np.array(os.listdir(trn_path))
            np.random.shuffle(files)
            
            fileName = np.array([trn_path+ '/' + files[i] for i in range(np.size(files))])
            file_list[className[clasi]].extend(fileName)


    return file_list             


def data_norm(batchData_f1,mean_f1, std_f1, max_f1, min_f1, PARAMS):
    
    batchData_f1 = np.squeeze(batchData_f1)
    # print('batchData shape:', batchData.shape)
        
    temp_mean_f1 = np.expand_dims(mean_f1, axis = 2)
    temp_mean_f1 = np.repeat(temp_mean_f1, PARAMS['CNN_patch_size'], axis = 2)             
    
    # print('temp_mean shape:', temp_mean.shape)
    batchData_f1 = batchData_f1 - np.repeat(temp_mean_f1, batchData_f1.shape[0], axis = 0)
                    
    
    if PARAMS['norm_method'] == 'mean_max':
        range1_f1 =  np.expand_dims(max_f1-min_f1, axis = 2)
        range1_f1 = np.repeat(range1_f1, PARAMS['CNN_patch_size'], axis = 2)

    
        batchData_f1 = batchData_f1/np.repeat(range1_f1, batchData_f1.shape[0], axis = 0)
    
    elif PARAMS['norm_method'] == 'zscore':
        temp_std_f1 = np.expand_dims(std_f1, axis = 2)
        temp_std_f1 = np.repeat(temp_std_f1, PARAMS['CNN_patch_size'], axis = 2)
        
        batchData_f1 = batchData_f1/np.repeat(temp_std_f1, batchData_f1.shape[0], axis = 0)

    batchData_f1 = np.expand_dims(batchData_f1, axis = 3)
    return batchData_f1 
 

def get_embedding(Train_Params, data):
    # learning_rate = 0.0001
    # optimizer = optimizers.Adam(lr=learning_rate)
    
    
    layer_name = 'dense'
    

    # embedding: feat1 
    model_f1 = Train_Params['model']
    model_f1.trainable = False
    intermediate_layer_model_f1 = Model(inputs=model_f1.input,
                                     outputs=model_f1.get_layer(layer_name).output)
    #intermediate_layer_model_f1.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    embeddings_f1 = intermediate_layer_model_f1.predict(data)
    print('embeddings_f1: ', np.shape(embeddings_f1))

    

    # # embedding: feat2
    # model_f2 = Train_Params['trained_model_f2']
    # model_f2.trainable = False
    # intermediate_layer_model_f2 = Model(inputs=model_f2.input,
    #                                   outputs=model_f2.get_layer(layer_name).output)
    # # intermediate_layer_model_f2.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    # embeddings_f1 = intermediate_layer_model_f2.predict(data)
    # # print('embeddings_f2: ', np.shape(embeddings_f2))
        
    # fin_embedding = np.append(embeddings_f1, embeddings_f2, axis = 1)
    return embeddings_f1



def get_embedding_both_feat(Train_Params_f1, Train_Params_f2, data, data2):
    # learning_rate = 0.0001
    # optimizer = optimizers.Adam(lr=learning_rate)
    
    
    layer_name = 'dense'
    

    # embedding: feat1 
    model_f1 = Train_Params_f1['model']
    model_f1.trainable = False
    intermediate_layer_model_f1 = Model(inputs=model_f1.input,
                                     outputs=model_f1.get_layer(layer_name).output)
    #intermediate_layer_model_f1.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    embeddings_f1 = intermediate_layer_model_f1.predict(data)
    print('embeddings_f1: ', np.shape(embeddings_f1))

 
    # embedding: feat2
    model_f2 = Train_Params_f2['model']
    model_f2.trainable = False
    intermediate_layer_model_f2 = Model(inputs=model_f2.input,
                                      outputs=model_f2.get_layer(layer_name).output)
    # intermediate_layer_model_f2.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    embeddings_f2 = intermediate_layer_model_f2.predict(data2)
    # print('embeddings_f2: ', np.shape(embeddings_f2))
        
    fin_embedding = np.append(embeddings_f1, embeddings_f2, axis = 1)
    
    return fin_embedding



def get_embedding_dnn_both_feat(PARAMS, embeddings):
    # learning_rate = 0.0001
    # optimizer = optimizers.Adam(lr=learning_rate)


    PARAMS['modelName_comb'] = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '.' + PARAMS['modelName_comb'].split('.')[-1]    
    weightFile_comb = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '.h5'
    # architechtureFile_comb = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '.json'    
    paramFile_comb = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '_params.npz'    
    # logFile_comb = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '_log.csv'    
    # arch_file_comb = '.'.join(PARAMS['modelName_comb'].split('.')[:-1]) + '_summary.txt'
        
    # embedding_shape = (2*168,)

    if os.path.exists(paramFile_comb):

        PARAMS['epochs'] = np.load(paramFile_comb)['epochs']
        PARAMS['batch_size'] = np.load(paramFile_comb)['batch_size']
        PARAMS['input_shape'] = np.load(paramFile_comb)['input_shape']
        learning_rate_dnn = np.load(paramFile_comb)['lr']
        trainingTimeTaken_dnn = np.load(paramFile_comb)['trainingTimeTaken']
        optimizer_dnn = optimizers.Adam(lr=learning_rate_dnn)
        
        #with open(architechtureFile_f1, 'r') as f: # Model reconstruction from JSON file
        #    model_f1 = model_from_json(f.read())
        #model_f1.load_weights(weightFile_f1) # Load weights into the new model
        print('weightFile_comb:', weightFile_comb)
        # attention_Layer = attention_Layer(attention_dim=1, name='attention_Layer')
        model_dnn =load_model(weightFile_comb, custom_objects={"attention_Layer": attention_Layer})
        
        print('model summary:', model_dnn.summary())
        model_dnn.compile(loss='binary_crossentropy', optimizer=optimizer_dnn, metrics=['accuracy'])

        print('CNN model exists! Loaded. Training time required=',trainingTimeTaken_dnn)

    
    layer_name = 'dropout'
    

    # embedding: feat1 
    # model_f1 = Train_Params['model_f1']
    # model_f1.trainable = False
    intermediate_layer_model_dnn = Model(inputs=model_dnn.input,
                                     outputs=model_dnn.get_layer(layer_name).output)
    #intermediate_layer_model_f1.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    embeddings_dnn = intermediate_layer_model_dnn.predict(embeddings)
    print('embeddings_dnn: ', np.shape(embeddings_dnn))

 
    # embedding: feat2

    return embeddings_dnn



                

def load_data(data_list, inx, numDim_numFrames_flag, mean_f1, std_f1, max_f1, min_f1, PARAMS, Train_Params_f1):
    DataAll = []
    print(inx)
    print(len(inx))
    for i in range(len(inx)):
        if data_list[i].split('.')[-1] == 'npy':
                # data = LoadFeatMatFile(data_list[i])
                data=np.load(data_list[i])
                print(data.shape)
                print("data loaded")
                data = CNN.get_feature_patches(data, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape_f1'], PARAMS)
                print(data.shape)
                data = np.expand_dims(data, axis=3)
                print("dim expaned")
                
                # if PARAMS['manual_Normalize_flag']:
                #     data = data_norm(data,mean_f1, std_f1, max_f1, min_f1, PARAMS)
                    
                embeddings_f1 = get_embedding(Train_Params_f1, data)
                print(embeddings_f1)
                if numDim_numFrames_flag:
                    data = data.T
                    
        if i == 0:
            DataAll = embeddings_f1
        else:
            DataAll = np.append(DataAll, embeddings_f1, axis = 0)
            
    return DataAll
    



def load_data_bothFeat(data_list, inx, numDim_numFrames_flag, PARAMS, Train_Params_f1, Train_Params_f2):
    DataAll = []
    for i in range(len(inx)):
        if data_list[i].split('.')[1] == 'mat':
                f1_file = data_list[i]
                
                datapath = '/'.join(f1_file.split('/')[:-6])
                f2_file = datapath + '/' + '_'.join(PARAMS['FeatName_f2'].split('_')[:1]) + '/'+'/'.join(f1_file.split('/')[-4:])
                
                data_f1 = LoadFeatMatFile(f1_file)
                data_f2 = LoadFeatMatFile(f2_file)
                
                data_f1 = CNN.get_feature_patches(data_f1, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape_f1'], PARAMS)
                data_f2 = CNN.get_feature_patches(data_f2, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape_f2'], PARAMS)
                
                data_f1 = np.expand_dims(data_f1, axis=3)
                data_f2 = np.expand_dims(data_f2, axis=3)
                
                if PARAMS['manual_Normalize_flag']:
                    data_f1 = data_norm(data_f1,PARAMS['trnData_mean_f1'], PARAMS['trnData_std_f1'], PARAMS['trnData_max_f1'], PARAMS['trnData_min_f1'], PARAMS)
                    data_f2 = data_norm(data_f2,PARAMS['trnData_mean_f2'], PARAMS['trnData_std_f2'], PARAMS['trnData_max_f2'], PARAMS['trnData_min_f2'], PARAMS)
                    
                embeddings = get_embedding_both_feat(Train_Params_f1, Train_Params_f2, data_f1, data_f2)
                embeddings_dnn = embeddings
                
                # embeddings_dnn = get_embedding_dnn_both_feat(PARAMS, embeddings)
                
                # if numDim_numFrames_flag:
                #     data = data.T
                    
        if i == 0:
            DataAll = embeddings_dnn
        else:
            DataAll = np.append(DataAll, embeddings_dnn, axis = 0)
            
    return DataAll

    


# ------ Main
# path = '/media/eee/DEB6DECBB6DEA377/Linux_Backup/Phd/Others_material/Moa/DID_excitation_feat/excitationFeb2021/spectrum_features/passage/ilpr_mel_spec_log/1st_session/'

# file_list = get_files_list(path)

# tot_changki_files = np.size(file_list['changki'])
# tot_mongsen_files = np.size(file_list['mongsen'])
# tot_chungli_files = np.size(file_list['chungli'])

# numDim_numFrames_flag = 1
# num_tsne_plots = 1

# changki_files = int(np.round(tot_changki_files/num_tsne_plots))
# mongsen_files = int(np.round(tot_mongsen_files/num_tsne_plots))
# chungli_files = int(np.round(tot_chungli_files/num_tsne_plots))

# featName = 'ILPR Log Mel Spectrogram'





# for i in range(num_tsne_plots):
#     endi_changki = 0
#     endi_mongsen = 0
#     endi_chungli = 0
#     if i == 0:
#         init_changki = i
#         init_mongsen = i
#         init_chungli = i
#     else:
#         init_changki = endi_changki
#         init_mongsen = endi_mongsen
#         init_chungli = endi_chungli
        
#     endi_changki = init_changki + changki_files
#     endi_mongsen = init_mongsen + mongsen_files
#     endi_chungli = init_chungli + chungli_files
    
#     inx = list(range(init_changki,endi_changki))
#     changki_data = load_data(file_list['changki'], inx, numDim_numFrames_flag)
    
#     inx = list(range(init_mongsen, endi_mongsen))
#     mongsen_data = load_data(file_list['mongsen'], inx, numDim_numFrames_flag)

#     inx = list(range(init_chungli, endi_chungli))
#     chungli_data = load_data(file_list['chungli'], inx, numDim_numFrames_flag)


#     # feat_dim = [0,1,5,7,19,20,21,22,23]
#     # single_data2 = single_data[:,feat_dim]
#     # ovrlp_data2 = ovrlp_data[:,feat_dim]

#     data = np.append(changki_data[:1000,:], mongsen_data[:1000,:], axis = 0)
#     # Norm_data = preprocessing.scale(data, axis=0)
#     label = np.append(np.zeros((np.shape(single_data)[0],1)), np.ones((np.shape(ovrlp_data)[0],1)), axis = 0)

#     DIM = np.array(list(range(0,np.shape(data)[1])))

#     data2 = data[:, DIM]

#     tsne = TSNE(
#          n_components=2, 
#          perplexity=30.0, 
#          early_exaggeration=12.0, 
#          learning_rate=200.0, 
#          n_iter=1000, 
#          n_iter_without_progress=300, 
#          min_grad_norm=1e-07, 
#          metric='euclidean', 
#          init='random', 
#          verbose=1, 
#          random_state=None, 
#     #     method='exact', 
#          method='barnes_hut',
#          angle=0.5
#          )


#     numPts = np.shape(data)[0]

#     data_embedded = tsne.fit_transform(data)
    
    
#     inx = list(range(0,np.shape(data)[0]))
#     inx1 = inx[:np.shape(single_data)[0]]
#     inx2 = inx[np.shape(single_data)[0]:]
    
#     plt.figure()
#     plt.scatter(x=data_embedded[inx2,0], y=data_embedded[inx2,1], c='m', label='Overlapped Speech')
#     plt.scatter(x=data_embedded[inx1,0], y=data_embedded[inx1,1], c='b', label='Single Speaker Speech')
#     plt.legend()
#     #plt.xlabel('Frequency bins')
#     #plt.ylabel('Likelihood')
#     plt.title('TSNE plot of proposed '+featName+' feature (random '+str(2*numPts)+' points)')
#     plt.show()






'''
Tot_point_s = np.shape(data_s1)[0]
Tot_point_n = np.shape(data_n1)[0]


featName = 'DCTILPR + RMFCC + MFCC'

data = np.append(Data_Sh, Data_Nor, axis = 0)
Norm_data = preprocessing.scale(data, axis=0)

label = np.append(np.ones((np.shape(data_s1)[0],1)), np.zeros((np.shape(data_n1)[0],1)), axis = 0)


DIM = np.array(list(range(0,np.shape(data)[1])))

data2 = data[:, DIM]


tsne = TSNE(
         n_components=2, 
         perplexity=30.0, 
         early_exaggeration=12.0, 
         learning_rate=200.0, 
         n_iter=1000, 
         n_iter_without_progress=300, 
         min_grad_norm=1e-07, 
         metric='euclidean', 
         init='random', 
         verbose=1, 
         random_state=None, 
    #     method='exact', 
         method='barnes_hut',
         angle=0.5
         )

numPts = np.shape(data)[0]

data_embedded = tsne.fit_transform(Norm_data)


inx = list(range(0,np.shape(data)[0]))
inx1 = inx[:np.shape(data_s1)[0]]
inx2 = inx[np.shape(data_s1)[0]:]

plt.figure()
plt.scatter(x=data_embedded[inx2,0], y=data_embedded[inx2,1], c='m', label='Normal Speech')
plt.scatter(x=data_embedded[inx1,0], y=data_embedded[inx1,1], c='b', label='Shouted Speech')
plt.legend()
#plt.xlabel('Frequency bins')
#plt.ylabel('Likelihood')
plt.title('TSNE plot of proposed '+featName+' feature (random '+str(2*numPts)+' points)')
plt.show()
'''