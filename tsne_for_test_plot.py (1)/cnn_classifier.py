#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:17:42 2020

@author: shikha
"""
import os, sys
import numpy as np
import math
import scipy.io.wavfile as wav
from tensorflow.keras.models import load_model
# from attention_layer_paper import attention_Layer
# from cnn_architecture import get_model_cnn_attn_gru, get_model_cnn_attn_gru_melSpec

# sys.path.insert(0,'/home/shikha/Shikha_Linux/PhD/Multi_Speaker/Overlapped_Speech_Analysis_23Jan_2020/Python/16August2020/baseline_2017_IEEETrans')
# sys.path.insert(0,'/home/eee/Shikha_Linux/Multi_Speaker/Overlapped_Speech_Analysis_23Jan_2020/Python/16August2020/baseline_2017_IEEETrans')

# from equalErrorRate import processDataTable2 as EER
# from model_sincnet import get_model_sincnet


# from tensorflow.keras.optimizers import SGD

# import random
# import statistics 
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import Model
# from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.image import PatchExtractor
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from tensorflow.keras.models import model_from_json

from tensorflow.keras import optimizers
# import misc as misc
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger





def get_feature_patches(FV, patch_size, patch_shift, input_shape, PARAMS):

    # # FV should be of the shape (nFeatures, nFrames)
    if not PARAMS['feat_nfeatXnFrame_flag']:
        # print('FV shape:', np.shape(FV))
        FV = FV.T

       
    patches = np.empty([])

    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)

    startTime = time.process_time()
    numPatches = int(np.ceil((np.shape(FV)[1]-patch_size)/patch_shift)) + 1


    patches = np.empty([], dtype=np.float32)
    for patch_i in range(numPatches):
        frmStart = patch_i*patch_shift
        frmEnd = np.min([patch_i*patch_shift + patch_size, np.shape(FV)[1]])
        if frmEnd-frmStart<patch_size:
            frmStart = frmEnd-patch_size
        fv_temp = np.expand_dims(FV[:,frmStart:frmEnd], axis=0)
        if np.size(patches)<=1:
            patches = fv_temp
        else:
            patches = np.append(patches, fv_temp, axis=0)


    return patches









def generator(PARAMS, file_list, batchSize):
    batch_count = 0
    # np.random.shuffle(file_list['Single_Speaker'])
    # np.random.shuffle(file_list['Overlapped'])
    
    class_name = [PARAMS['classes'][i] for i in range(len(PARAMS['classes']))]
    for clsi in class_name:
        np.random.shuffle(file_list[clsi])
        

    file_list_changki_temp = file_list['changki'].copy()
    file_list_mongsen_temp = file_list['mongsen'].copy()
    file_list_chungli_temp = file_list['chungli'].copy()
    

    batchData_changki = np.empty([], dtype=float)
    batchData_mongsen = np.empty([], dtype=float)
    batchData_chungli = np.empty([], dtype=float)
    
    balance_changki = 0
    balance_mongsen = 0
    balance_chungli = 0
    
    while 1:
        batchData = np.empty([], dtype=float)
        batchLabel = np.empty([], dtype=float)
        
        # ---- changki class
        while balance_changki<batchSize:
            if not file_list_changki_temp:
                file_list_changki_temp = file_list['changki'].copy()
            changki_fName_path = file_list_changki_temp.pop()
            if not os.path.exists(changki_fName_path):
                continue

            if changki_fName_path.split('.')[-1] == 'mat':
                fv_changki = misc.LoadFeatMatFile(changki_fName_path)
            else:
                fv_changki = np.load(changki_fName_path, allow_pickle=True)
            fv_changki = fv_changki.astype(np.float32)


            fv_changki = get_feature_patches(fv_changki, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'], PARAMS)

            if PARAMS['Noise_flag_data_aug']: # Adding very low energy random noise from N(0,1) for data augmentation
                fv_changki_noise = fv_changki + np.random.normal(size = np.shape(fv_changki))*1e-7
                fv_changki = np.append(fv_changki,fv_changki_noise, axis=0)
                changki_inx = list(range(np.shape(fv_changki)[0]))
                np.random.shuffle(changki_inx)
                fv_changki = fv_changki[changki_inx,:,:]
                

            if balance_changki==0:
                batchData_changki = fv_changki
            else:
                batchData_changki = np.append(batchData_changki, fv_changki, axis=0)
            balance_changki += np.shape(fv_changki)[0]

        
        # ---- chungli class
        while balance_chungli<batchSize:
            if not file_list_chungli_temp:
                file_list_chungli_temp = file_list['chungli'].copy()
            chungli_fName_path = file_list_chungli_temp.pop()
            if not os.path.exists(chungli_fName_path):
                continue

            if chungli_fName_path.split('.')[-1] == 'mat':
                fv_chungli = misc.LoadFeatMatFile(chungli_fName_path)               
            else:
                fv_chungli = np.load(chungli_fName_path, allow_pickle=True)
            fv_chungli = fv_chungli.astype(np.float32)
            
            # if not PARAMS['trn_segDurLevel'] == 'frame':
            fv_chungli = get_feature_patches(fv_chungli, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'], PARAMS)

            if PARAMS['Noise_flag_data_aug']: # Adding very low energy random noise from N(0,1) for data augmentation
                fv_chungli_noise = fv_chungli + np.random.normal(size = np.shape(fv_chungli))*1e-7
                fv_chungli = np.append(fv_chungli,fv_chungli_noise, axis=0)
                chungli_inx = list(range(np.shape(fv_chungli)[0]))
                np.random.shuffle(chungli_inx)
                fv_chungli = fv_chungli[chungli_inx,:,:]

            if balance_chungli==0:
                batchData_chungli = fv_chungli
            else:
                batchData_chungli = np.append(batchData_chungli, fv_chungli, axis=0)
            balance_chungli += np.shape(fv_chungli)[0]


        # ---- mongsen class
        while balance_mongsen<batchSize:
            if not file_list_mongsen_temp:
                file_list_mongsen_temp = file_list['mongsen'].copy()
            mongsen_fName_path = file_list_mongsen_temp.pop()
            if not os.path.exists(mongsen_fName_path):
                continue
            # print('mu_fName_path: ', mu_fName_path)
            if mongsen_fName_path.split('.')[-1] == 'mat':
                fv_mongsen = misc.LoadFeatMatFile(mongsen_fName_path)               
            else:
                fv_mongsen = np.load(mongsen_fName_path, allow_pickle=True)
            fv_mongsen = fv_mongsen.astype(np.float32)
           
            # if not PARAMS['trn_segDurLevel'] == 'frame':
            fv_mongsen = get_feature_patches(fv_mongsen, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'], PARAMS)

            if PARAMS['Noise_flag_data_aug']: # Adding very low energy random noise from N(0,1) for data augmentation
                fv_mongsen_noise = fv_mongsen + np.random.normal(size = np.shape(fv_mongsen))*1e-7
                fv_mongsen = np.append(fv_mongsen,fv_mongsen_noise, axis=0)
                mongsen_inx = list(range(np.shape(fv_mongsen)[0]))
                np.random.shuffle(mongsen_inx)
                fv_mongsen = fv_mongsen[mongsen_inx,:,:]


            if balance_mongsen==0:
                batchData_mongsen = fv_mongsen
            else:
                batchData_mongsen = np.append(batchData_mongsen, fv_mongsen, axis=0)
            balance_mongsen += np.shape(fv_mongsen)[0]

        
        # ---- Batch data         
        batchData = batchData_changki[:batchSize, :]
        batchData = np.append(batchData, batchData_mongsen[:batchSize, :], axis=0)        
        batchData = np.append(batchData, batchData_chungli[:batchSize, :], axis=0)

        # normalization of batch data 
        if PARAMS['manual_Normalize_flag']:
            temp_mean = np.expand_dims(PARAMS['trnData_mean_f1'], axis = 2)
            temp_mean = np.repeat(temp_mean, PARAMS['CNN_patch_size'], axis = 2)
            temp_mean = np.repeat(temp_mean, 3*PARAMS['batch_size'], axis = 0) 
            batchData = np.subtract(batchData, temp_mean)                       
            
            if PARAMS['norm_method'] == 'mean_max':
                range1 =  np.expand_dims(PARAMS['trnData_max_f1']-PARAMS['trnData_min_f1'], axis = 2)
                range1 = np.repeat(range1, PARAMS['CNN_patch_size'], axis = 2)
                range1 = np.repeat(range1, 3*PARAMS['batch_size'], axis = 0)
                batchData = np.divide(batchData, range1++1e-10)
                
            elif PARAMS['norm_method'] == 'zscore':
                temp_std = np.expand_dims(PARAMS['trnData_std_f1'], axis = 2)
                temp_std = np.repeat(temp_std, PARAMS['CNN_patch_size'], axis = 2)
                temp_std = np.repeat(temp_std, 3*PARAMS['batch_size'], axis = 0)                
                batchData = np.divide(batchData, temp_std+1e-10) 


        batchData = np.expand_dims(batchData, axis=3)
        
        # --- keep track of remaining data
        balance_changki -= batchSize
        balance_chungli -= batchSize
        balance_mongsen -= batchSize
        
        batchData_changki = batchData_changki[batchSize:, :]
        batchData_chungli = batchData_chungli[batchSize:, :] 
        batchData_mongsen = batchData_mongsen[batchSize:, :]            

        # 
        class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        batchLabel = np.ones(3*batchSize)
        batchLabel[:batchSize] = class_labels['changki']
        batchLabel[batchSize:2*batchSize] = class_labels['mongsen']
        batchLabel[2*batchSize:] = class_labels['chungli']
        
        if not PARAMS['OHE_flag']:
            OHE_batchLabel = batchLabel
        else:
            OHE_batchLabel = to_categorical(batchLabel, num_classes=3)
        
                
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData))
        
        yield batchData, OHE_batchLabel




def train_model(PARAMS, model, weightFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    # es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, min_delta=0.01, patience=5)
    # mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    
    # To load custom layer "attention_Layer" from saved model, then save_weights_only=False should be used.
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    logFile = '.'.join(weightFile.split('.')[:-1]) + '_log_fold' + str(PARAMS['fold']) + '.csv'
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.process_time()

    SPE = PARAMS['train_steps_per_epoch']
    SPE_val = PARAMS['val_steps']
    print('SPE: ', SPE, SPE_val)
    
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files_f1'].keys():
        # files = PARAMS['train_files_f1'][classname]
        # np.random.shuffle(files)
        # idx = int(len(files)*0.7)
        train_files[classname] = PARAMS['train_files_f1'][classname]
        val_files[classname] = PARAMS['val_files_f1'][classname]
    
    # Train the model
    History = model.fit_generator(
            generator(PARAMS, train_files, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, val_files, PARAMS['batch_size']), 
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp],
            shuffle=True,
            )
    


    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History

        
def train_model_nd_save_ModelParams(PARAMS, weightFile_f1, architechtureFile_f1, paramFile_f1):
    # train and save different attributes of the model
    
    # load CNN-GRU-attention model architecture
    if PARAMS['which_model'] == 'cnn_gru_attn':
        model_f1, learning_rate_f1 = get_model_cnn_attn_gru(PARAMS['classes'], PARAMS['input_shape'],PARAMS)
            
    # model training
    model_f1, trainingTimeTaken_f1, History_f1 = train_model(PARAMS, model_f1, weightFile_f1)
        
    # save different attriutes of trained model
    if PARAMS['save_flag']:
        model_f1.save(weightFile_f1)                # save weights of trained model
        #model_f1.save_weights(weightFile_f1)       # Save the weights
        with open(architechtureFile_f1, 'w') as f:  # Save the model architecture
            f.write(model_f1.to_json())
        np.savez(paramFile_f1, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], input_shape=PARAMS['input_shape'], lr=learning_rate_f1, trainingTimeTaken=trainingTimeTaken_f1)


    train_param_temp = {'model': model_f1,
                        'trainingTimeTaken': trainingTimeTaken_f1,
                        'paramFile': paramFile_f1,
                        'architechtureFile': architechtureFile_f1,
                        'learning_rate': learning_rate_f1,
                        'trainingTimeTaken' : trainingTimeTaken_f1,
                        'weightFile':weightFile_f1}
    return train_param_temp
    



def load_PreTrained_Model(paramFile_f1, weightFile_f1, PARAMS):
    
    # load pre-trained model
    learning_rate_f1 = np.load(paramFile_f1)['lr']
    trainingTimeTaken_f1 = np.load(paramFile_f1)['trainingTimeTaken']
    optimizer_f1 = optimizers.Adam(lr=learning_rate_f1)
    
    # load weight file
    print('weightFile_f1:', weightFile_f1)
    model_f1 = load_model(weightFile_f1, custom_objects={"attention_Layer": attention_Layer})
    print('model summary:', model_f1.summary())
    
    # compile the model
    if len(PARAMS['classes']) == 2:
        model_f1.compile(loss = "binary_crossentropy", optimizer = optimizer_f1, metrics=['accuracy'])  
    else:
        model_f1.compile(loss = "categorical_crossentropy", optimizer = optimizer_f1, metrics=['accuracy'])        
    

    # print('Feature 1: CNN model exists! Loaded. Training time required=',trainingTimeTaken_f1)
    
    train_param_temp = {'model': model_f1,
                    'trainingTimeTaken': trainingTimeTaken_f1,
                    'paramFile': paramFile_f1,
                    'learning_rate': learning_rate_f1,
                    'trainingTimeTaken' : trainingTimeTaken_f1,
                    'optimizer': optimizer_f1,
                    'weightFile': weightFile_f1}
    
    return train_param_temp


def train_cnnlstm(PARAMS):
    PARAMS['modelName_f1'] = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '.' + PARAMS['modelName_f1'].split('.')[-1]    # model name with path
    weightFile_f1 = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '.h5'            # weight file name with path
    architechtureFile_f1 = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '.json'   # architecture file name with path
    paramFile_f1 = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '_params.npz'     # parameters file name with path

    # logFile_f1 = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '_log.csv'          # log file name with path
    # arch_file_f1 = '.'.join(PARAMS['modelName_f1'].split('.')[:-1]) + '_summary.txt'    # architecture summary file with path
        
    ##########################################################################  
    # ---- check for Feat1 model
    # if paramFile_f1 does not exist, train the model
    if not os.path.exists(paramFile_f1):
        
        train_param_f1 = train_model_nd_save_ModelParams(PARAMS, weightFile_f1, architechtureFile_f1, paramFile_f1)
        print('CNN model trained.')
        
    # if paramFile_f1 exists, load the pre-trained model
    else:
        PARAMS['epochs'] = np.load(paramFile_f1)['epochs']
        PARAMS['batch_size'] = np.load(paramFile_f1)['batch_size']
        PARAMS['input_shape'] = np.load(paramFile_f1)['input_shape']
        
        # load pre-trained model
        train_param_f1 = load_PreTrained_Model(paramFile_f1, weightFile_f1, PARAMS)
        print('Feature 1: CNN model exists! Loaded. Training time required=',train_param_f1['trainingTimeTaken'])

    # train parameters
    Train_Params = {
                'model_f1': train_param_f1['model'],
                'trainingTimeTaken_f1': train_param_f1['trainingTimeTaken'],
                'epochs_f1': PARAMS['epochs'],
                'batch_size_f1': PARAMS['batch_size'],
                'learning_rate_f1': train_param_f1['learning_rate'],
                'paramFile_f1': train_param_f1['paramFile'],
                'architechtureFile_f1': architechtureFile_f1,
                'weightFile_f1': train_param_f1['weightFile'],
                } 

    ##########################################################################
    # if more than one features are used
    if PARAMS['Num_Feat_forComb'] > 1:
        
        # file names for feature 2
        PARAMS['modelName_f2'] = '.'.join(PARAMS['modelName_f2'].split('.')[:-1]) + '.' + PARAMS['modelName_f2'].split('.')[-1]  # model name with path
        weightFile_f2 = '.'.join(PARAMS['modelName_f2'].split('.')[:-1]) + '.h5'                    # weight file name with path
        architechtureFile_f2 = '.'.join(PARAMS['modelName_f2'].split('.')[:-1]) + '.json'           # architecture file name with path
        paramFile_f2 = '.'.join(PARAMS['modelName_f2'].split('.')[:-1]) + '_params.npz'             # parameters file name with path

        # ---- check for Feat2 model
        # if paramFile_f2 does not exist, train the model
        if not os.path.exists(paramFile_f2):
            
            train_param_f2 = train_model_nd_save_ModelParams(PARAMS, weightFile_f2, architechtureFile_f2, paramFile_f2)
            print('CNN model trained.')
            
        # if paramFile_f2 exists, load the pre-trained model
        else:
            PARAMS['epochs'] = np.load(paramFile_f2)['epochs']
            PARAMS['batch_size'] = np.load(paramFile_f2)['batch_size']
            PARAMS['input_shape'] = np.load(paramFile_f2)['input_shape']

            # load pre-trained model
            train_param_f2 = load_PreTrained_Model(paramFile_f2, weightFile_f2, PARAMS)
            print('Feature 2: CNN model exists! Loaded. Training time required=',train_param_f2['trainingTimeTaken'])
        
        
        # update the existing Train parameters
        Train_Params.update({
                    'model_f2': train_param_f2['model'],
                    'trainingTimeTaken_f2': train_param_f2['trainingTimeTaken'],
                    'epochs_f2': PARAMS['epochs'],
                    'batch_size_f2': PARAMS['batch_size'],
                    'learning_rate_f2': train_param_f2['learning_rate'],
                    'paramFile_f2': train_param_f2['paramFile'],
                    'architechtureFile_f2': architechtureFile_f2,
                    'weightFile_f2': train_param_f2['weightFile'],            
                    })
    
    ##########################################################################
    # if more than two features are considered
    if PARAMS['Num_Feat_forComb'] > 2:
        
        # file's name for feature 3 
        PARAMS['modelName_f3'] = '.'.join(PARAMS['modelName_f3'].split('.')[:-1]) + '.' + PARAMS['modelName_f3'].split('.')[-1]
        weightFile_f3 = '.'.join(PARAMS['modelName_f3'].split('.')[:-1]) + '.h5'
        architechtureFile_f3 = '.'.join(PARAMS['modelName_f3'].split('.')[:-1]) + '.json'
        paramFile_f3 = '.'.join(PARAMS['modelName_f3'].split('.')[:-1]) + '_params.npz'
        

        # ---- check for Feat2 model
        if not os.path.exists(paramFile_f3):
            
            train_param_f3 = train_model_nd_save_ModelParams(PARAMS, weightFile_f3, architechtureFile_f3, paramFile_f3)
            print('CNN model trained.')           
        else:
            PARAMS['epochs'] = np.load(paramFile_f3)['epochs']
            PARAMS['batch_size'] = np.load(paramFile_f3)['batch_size']
            PARAMS['input_shape'] = np.load(paramFile_f3)['input_shape']

            # load pre-trained model
            train_param_f3 = load_PreTrained_Model(paramFile_f3, weightFile_f3, PARAMS)
            print('Feature 3: CNN model exists! Loaded. Training time required=',train_param_f3['trainingTimeTaken'])

        
        Train_Params.update( {'model_f3': train_param_f3['model'],'trainingTimeTaken_f3': train_param_f3['trainingTimeTaken'], 'epochs_f3': PARAMS['epochs'], 'batch_size_f3': PARAMS['batch_size']})  
        Train_Params.update( {'learning_rate_f3': train_param_f3['learning_rate'],'paramFile_f3': train_param_f3['paramFile'],'architechtureFile_f3': architechtureFile_f3,'weightFile_f3': train_param_f3['weightFile']})  

   
    return Train_Params







def load_PreTrained_Model_v2(PARAMS, modelName_f1):
    
    modelName_f1 = '.'.join(modelName_f1.split('.')[:-1]) + '.' + modelName_f1.split('.')[-1]    # model name with path
    weightFile_f1 = '.'.join(modelName_f1.split('.')[:-1]) + '.h5'            # weight file name with path
    architechtureFile_f1 = '.'.join(modelName_f1.split('.')[:-1]) + '.json'   # architecture file name with path
    paramFile_f1 = '.'.join(modelName_f1.split('.')[:-1]) + '_params.npz'     # parameters file name with path
    
    
    
    # load pre-trained model
    learning_rate_f1 = np.load(paramFile_f1)['lr']
    trainingTimeTaken_f1 = np.load(paramFile_f1)['trainingTimeTaken']
    optimizer_f1 = optimizers.Adam(lr=learning_rate_f1)
    
    # load weight file
    print('weightFile_f1:', weightFile_f1)
    model_f1 = load_model(weightFile_f1, custom_objects={"attention_Layer": attention_Layer})
    print('model summary:', model_f1.summary())
    
    # compile the model
    if len(PARAMS['classes']) == 2:
        model_f1.compile(loss = "binary_crossentropy", optimizer = optimizer_f1, metrics=['accuracy'])  
    else:
        model_f1.compile(loss = "categorical_crossentropy", optimizer = optimizer_f1, metrics=['accuracy'])        
    

    print('Feature 1: CNN model exists! Loaded. Training time required=',trainingTimeTaken_f1)
    
    train_param_temp = {'model': model_f1,
                    'trainingTimeTaken': trainingTimeTaken_f1,
                    'paramFile': paramFile_f1,
                    'learning_rate': learning_rate_f1,
                    'trainingTimeTaken' : trainingTimeTaken_f1,
                    'optimizer': optimizer_f1,
                    'weightFile': weightFile_f1}
    
    return train_param_temp






###############################################################################
# save results into csv file
def print_results(opDir, fName_suffix, res_dict):
    if fName_suffix!='':
        opFile = opDir + '/Performance_' + fName_suffix + '.csv'
    else:
        opFile = opDir + '/Performance.csv'
        
    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = ''
    values = ''
    for i in range(len(res_dict)):
        heading = heading + np.squeeze(res_dict[str(i)]).tolist().split(':')[0] + '\t'
        values = values + np.squeeze(res_dict[str(i)]).tolist().split(':')[1] + '\t'

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()






###############################################################################
# --- Model Testing 

def generator_test(PARAMS, file_name, clNum, input_shape):

    if file_name.split('.')[-1] == 'mat':
        batchData = misc.LoadFeatMatFile(file_name)
    else:
        batchData = np.load(file_name, allow_pickle=True)
    
    batchData = batchData.astype(np.float32)

    batchData = get_feature_patches(batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], input_shape, PARAMS)
    
    batchData = np.expand_dims(batchData, axis=3)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel



def postProcessing_predCls(PARAMS, pred, batchLabel):
    tot_frame =  np.shape(pred)[0]
    count = 0
    SegDur_shift = int(np.floor(PARAMS['test_frmNumSeg']/PARAMS['test_segShft']))
    for i in range(0, tot_frame, SegDur_shift):
        temp = np.mean(pred[i:i+SegDur_shift,:], axis = 0)
        if count == 0:
            pred_fin = np.expand_dims(temp, axis = 0)
            count += 1
        else:
            pred_fin = np.append(pred_fin, np.expand_dims(temp, axis = 0) , axis = 0)
            
    batchLabel_fin = np.ones((np.shape(pred_fin)[0],))*np.unique(batchLabel)
    return pred_fin, batchLabel_fin
 

def postProcessing_majorityVote_modified(PARAMS, pred_lab, batchLabel_f1):
    tot_frame =  np.shape(pred_lab)[0]
    count = 0
    SegDur_shift = int(np.floor(PARAMS['test_frmNumSeg']/PARAMS['test_segShft']))
    for i in range(0, tot_frame, SegDur_shift):
        temp_lab = pred_lab[i:i+SegDur_shift]
        tot_class0 = np.size(np.squeeze(np.where(temp_lab==0)))
        tot_class1 = np.size(np.squeeze(np.where(temp_lab==1)))
        tot_class2 = np.size(np.squeeze(np.where(temp_lab==2)))
        
        if not (tot_class0 == tot_class1 and tot_class1 == tot_class2):
            fin_lab = np.argmax([tot_class0, tot_class1, tot_class2])

        
        if count == 0:
            pred_lab_fin = np.expand_dims(fin_lab, axis = 0)
            count += 1
        else:
            pred_lab_fin = np.append(pred_lab_fin, np.expand_dims(fin_lab, axis = 0) , axis = 0)
            
    batchLabel_fin = np.ones((np.shape(pred_lab_fin)[0],))*np.unique(batchLabel_f1)
    return pred_lab_fin, batchLabel_fin

  
 
def data_norm(batchData_f1, PARAMS, trnData_mean, trnData_max, trnData_min, trnData_std):
    
    batchData_f1 = np.squeeze(batchData_f1)
    # print('batchData shape:', batchData.shape)
        
    temp_mean_f1 = np.expand_dims(trnData_mean, axis = 2)
    temp_mean_f1 = np.repeat(temp_mean_f1, PARAMS['CNN_patch_size'], axis = 2)             
    temp_mean_f1 = np.repeat(temp_mean_f1, batchData_f1.shape[0], axis = 0) 
    batchData_f1 = np.subtract(batchData_f1, temp_mean_f1)    
    # print('temp_mean shape:', temp_mean.shape)

                    
    
    if PARAMS['norm_method'] == 'mean_max':
        range1_f1 =  np.expand_dims(trnData_max-trnData_min, axis = 2)
        range1_f1 = np.repeat(range1_f1, PARAMS['CNN_patch_size'], axis = 2)
        range1_f1 = np.repeat(range1_f1, batchData_f1.shape[0], axis = 0)
    
        batchData_f1 = np.divide(batchData_f1, range1_f1+1e-10)
    
    elif PARAMS['norm_method'] == 'zscore':
        temp_std_f1 = np.expand_dims(trnData_std, axis = 2)
        temp_std_f1 = np.repeat(temp_std_f1, PARAMS['CNN_patch_size'], axis = 2)
        temp_std_f1 = np.repeat(temp_std_f1, batchData_f1.shape[0], axis = 0)                
        batchData_f1 = np.divide(batchData_f1, temp_std_f1+1e-10) 

    batchData_f1 = np.expand_dims(batchData_f1, axis = 3)
    return batchData_f1 
 
    
 
    
def test_model(PARAMS, Train_Params):
    # start = time.clock()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    # startTime = time.clock()
    
    item_not_found_cntr = 0
    sentNum = 0 # added for baselines
    
    for classname in PARAMS['test_files_f1'].keys():
        clNum = class_labels[classname]
        files_f1 = PARAMS['test_files_f1'][classname]

        if PARAMS['Num_Feat_forComb'] > 1:
            files_f2 = PARAMS['test_files_f2'][classname]
            
            if PARAMS['Num_Feat_forComb'] > 2:
                files_f3 = PARAMS['test_files_f3'][classname]

  
        # print('test_files: ', files)

        for fl in files_f1:
            ##################################################################
            # ---- feature 1 files
            fName_f1 = fl

            count += 1
            sentNum += 1    # added for baselines
            
            # ---- load feat 1 batch data
            batchData_f1, batchLabel_f1 = generator_test(PARAMS, fName_f1, clNum, PARAMS['input_shape'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f1 = data_norm(batchData_f1, PARAMS, PARAMS['trnData_mean_f1'], PARAMS['trnData_max_f1'], PARAMS['trnData_min_f1'], PARAMS['trnData_std_f1'])

            # ---- prediction for feat1 
            print('batchData_f1 shape:', np.shape(batchData_f1))
            pred_f1 = Train_Params['model_f1'].predict(x=batchData_f1)
            pred = pred_f1

            ##################################################################
            # if more than one feat
            if PARAMS['Num_Feat_forComb'] > 1:
                
                path_f2 = '/'.join(fName_f1.split('/')[:-6])
                fName_f2 = path_f2 + '/' + '_'.join(PARAMS['FeatName_f2'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
                
                
                if not os.path.exists(fName_f1) or not os.path.exists(fName_f2):
                    continue
                
                batchData_f2, batchLabel_f2 = generator_test(PARAMS, fName_f2, clNum)
                if PARAMS['manual_Normalize_flag']:
                    batchData_f2 = data_norm(batchData_f2, PARAMS, PARAMS['trnData_mean_f2'], PARAMS['trnData_max_f2'], PARAMS['trnData_min_f2'], PARAMS['trnData_std_f2'])
                                  
                # ---- prediction for feat2 
                print('batchData_f2 shape:', np.shape(batchData_f2))
                pred_f2  = Train_Params['model_f2'].predict(x=batchData_f2)
                
                # adding prediction obtained from feature 1 and feature 2
                if PARAMS['alpha_flag']:
                    pred = PARAMS['alpha']*pred_f1 + (1-PARAMS['alpha'])*pred_f2
                    # pred = (1-PARAMS['alpha'])*pred_f1 + (PARAMS['alpha'])*pred_f2                    
                else:
                    pred = pred_f1 + pred_f2

            ##################################################################
            # if more than two feat
            if PARAMS['Num_Feat_forComb'] > 2:
                
                path_f3 = '/'.join(fName_f1.split('/')[:-6])
                fName_f3 = path_f3 + '/' + '_'.join(PARAMS['FeatName_f3'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
                
                
                if not os.path.exists(fName_f3):
                    continue
                
                batchData_f3, batchLabel_f3 = generator_test(PARAMS, fName_f3, clNum)
                if PARAMS['manual_Normalize_flag']:
                    batchData_f3 = data_norm(batchData_f3, PARAMS, PARAMS['trnData_mean_f3'], PARAMS['trnData_max_f3'], PARAMS['trnData_min_f3'], PARAMS['trnData_std_f3'])
                                  
                # ---- prediction for feat3 
                print('batchData_f3 shape:', np.shape(batchData_f3))
                pred_f3  = Train_Params['model_f3'].predict(x=batchData_f3)
                
                # adding prediction obtained from feature 1, feature 2 and feature 3
                if PARAMS['alpha_flag']:
                    pred = PARAMS['alpha1']*pred + (1-PARAMS['alpha1'])*pred_f3
                else:
                    pred = pred + pred_f3


            ##################################################################

                
            # if train seg dur and test seg dur are not equal
            if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg'] and not PARAMS['majority_vote_flag']:
                pred, batchLabel = postProcessing_predCls(PARAMS, pred, batchLabel_f1)  # using mean predicted scores of a test segment

            # --- condition for num of output node
            if PARAMS['numNodeOtpt'] == 1:
                pred_lab = np.zeros(np.shape(pred))
                index = pred > 0.5
                pred_lab[index] = 1
            else:
                pred_lab = np.argmax(pred, axis=1)
                
            if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg']  and PARAMS['majority_vote_flag']:
                print('majority vote')
                pred_lab, batchLabel_f1 = postProcessing_majorityVote_modified(PARAMS, pred_lab, batchLabel_f1) 
                
                
            
                
            
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel_f1.tolist())            

            if not np.shape(batchLabel_f1)[0] == np.shape(batchData_f1)[0]:
                print('batch size and num of batch labels are not equal')

            if not np.shape(pred)[0] == np.shape(batchData_f1)[0]:
                print('batch size and num of predictions are not equal')            

            if not np.shape(pred_lab)[0] == np.shape(batchData_f1)[0]:
                print('batch size and num of pred_lab are not equal')
                                        
            if not np.size(PtdLabels) == np.size(GroundTruth):
                print('PtdLabels and GroundTruth have different size')
                # print('Filenmae:',fName_f1)
                break
                
                
            # print('pred_lab:', pred_lab)
            print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2))
            print('ground_truth: ', np.sum(batchLabel_f1==0), np.sum(batchLabel_f1==1), np.sum(batchLabel_f1==2))
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            print(PARAMS['classes'][clNum], fl, np.shape(batchData_f1), ' acc=', np.round(np.sum(pred_lab==batchLabel_f1)*100/len(batchLabel_f1), 2))

    ConfMat, fscore, precision, recall, accuracy = misc.getPerformance(PtdLabels, GroundTruth)
    print('fscore shape: ', np.shape(fscore))
    
    print('item_not_found_cntr:',item_not_found_cntr)
    return ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth




'''
This function is the API that is called for CNN testing. Pass in a dictionary object "PARAMS" containing the required parameters. Also
pass the object returned by the train_cnn() function.
'''
def test_cnn(PARAMS, Train_Params):
    
    ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth = test_model(PARAMS, Train_Params)

    Test_Params = {
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels,
        'Predictions_test': Predictions,
        'GroundTruth_test': GroundTruth,
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall
        }

    return Test_Params



def test_model_v2(PARAMS, Train_Params_f1, Train_Params_f2):
    # start = time.clock()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    # startTime = time.clock()
    
    item_not_found_cntr = 0
    sentNum = 0 # added for baselines
    
    for classname in PARAMS['test_files_f1'].keys():
        clNum = class_labels[classname]
        files_f1 = PARAMS['test_files_f1'][classname]



  
        # print('test_files: ', files)

        for fl in files_f1:
            ##################################################################
            # ---- feature 1 files
            fName_f1 = fl

            count += 1
            sentNum += 1    # added for baselines
            
            # ---- load feat 1 batch data
            batchData_f1, batchLabel_f1 = generator_test(PARAMS, fName_f1, clNum, PARAMS['input_shape_f1'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f1 = data_norm(batchData_f1, PARAMS, PARAMS['trnData_mean_f1'], PARAMS['trnData_max_f1'], PARAMS['trnData_min_f1'], PARAMS['trnData_std_f1'])

            # ---- prediction for feat1 
            print('batchData_f1 shape:', np.shape(batchData_f1))
            pred_f1 = Train_Params_f1['model'].predict(x=batchData_f1)
            pred = pred_f1

            ##################################################################

                
            path_f2 = '/'.join(fName_f1.split('/')[:-6])
            fName_f2 = path_f2 + '/' + '_'.join(PARAMS['FeatName_f2'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
            
            
            if not os.path.exists(fName_f1) or not os.path.exists(fName_f2):
                continue
            
            batchData_f2, batchLabel_f2 = generator_test(PARAMS, fName_f2, clNum, PARAMS['input_shape_f2'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f2 = data_norm(batchData_f2, PARAMS, PARAMS['trnData_mean_f2'], PARAMS['trnData_max_f2'], PARAMS['trnData_min_f2'], PARAMS['trnData_std_f2'])
                              
            # ---- prediction for feat2 
            print('batchData_f2 shape:', np.shape(batchData_f2))
            pred_f2  = Train_Params_f2['model'].predict(x=batchData_f2)
            
            # adding prediction obtained from feature 1 and feature 2
            if PARAMS['alpha_flag']:
                pred = PARAMS['alpha']*pred_f1 + (1-PARAMS['alpha'])*pred_f2
                # pred = (1-PARAMS['alpha'])*pred_f1 + (PARAMS['alpha'])*pred_f2                    
            else:
                pred = pred_f1 + pred_f2


            ##################################################################

                
            # # if train seg dur and test seg dur are not equal
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg'] and not PARAMS['majority_vote_flag']:
            #     pred, batchLabel = postProcessing_predCls(PARAMS, pred, batchLabel_f1)  # using mean predicted scores of a test segment

            # --- condition for num of output node
            if PARAMS['numNodeOtpt'] == 1:
                pred_lab = np.zeros(np.shape(pred))
                index = pred > 0.5
                pred_lab[index] = 1
            else:
                pred_lab = np.argmax(pred, axis=1)
                
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg']  and PARAMS['majority_vote_flag']:
            #     print('majority vote')
            #     pred_lab, batchLabel_f1 = postProcessing_majorityVote_modified(PARAMS, pred_lab, batchLabel_f1) 
                
                
            
                
            
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel_f1.tolist())            

            # if not np.shape(batchLabel_f1)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of batch labels are not equal')

            # if not np.shape(pred)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of predictions are not equal')            

            # if not np.shape(pred_lab)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of pred_lab are not equal')
                                        
            # if not np.size(PtdLabels) == np.size(GroundTruth):
            #     print('PtdLabels and GroundTruth have different size')
            #     # print('Filenmae:',fName_f1)
            #     break
                
                
            # print('pred_lab:', pred_lab)
            print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2))
            print('ground_truth: ', np.sum(batchLabel_f1==0), np.sum(batchLabel_f1==1), np.sum(batchLabel_f1==2))
            
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            
            print(PARAMS['classes'][clNum], fl, np.shape(batchData_f1), ' acc=', np.round(np.sum(pred_lab==batchLabel_f1)*100/len(batchLabel_f1), 2))

    ConfMat, fscore, precision, recall, accuracy = misc.getPerformance(PtdLabels, GroundTruth)
    print('fscore shape: ', np.shape(fscore))
    
    print('item_not_found_cntr:',item_not_found_cntr)
    return ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth


def test_model_v3(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3):
    # start = time.clock()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    # startTime = time.clock()
    
    item_not_found_cntr = 0
    sentNum = 0 # added for baselines
    
    for classname in PARAMS['test_files_f1'].keys():
        clNum = class_labels[classname]
        files_f1 = PARAMS['test_files_f1'][classname]



  
        # print('test_files: ', files)

        for fl in files_f1:
            ##################################################################
            # ---- feature 1 files
            fName_f1 = fl

            count += 1
            sentNum += 1    # added for baselines
            
            # ---- load feat 1 batch data
            batchData_f1, batchLabel_f1 = generator_test(PARAMS, fName_f1, clNum, PARAMS['input_shape_f1'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f1 = data_norm(batchData_f1, PARAMS, PARAMS['trnData_mean_f1'], PARAMS['trnData_max_f1'], PARAMS['trnData_min_f1'], PARAMS['trnData_std_f1'])

            # ---- prediction for feat1 
            print('batchData_f1 shape:', np.shape(batchData_f1))
            pred_f1 = Train_Params_f1['model'].predict(x=batchData_f1)
            pred = pred_f1

            ##################################################################

                
            path_f2 = '/'.join(fName_f1.split('/')[:-6])
            fName_f2 = path_f2 + '/' + '_'.join(PARAMS['FeatName_f2'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
            
            
            if not os.path.exists(fName_f1) or not os.path.exists(fName_f2):
                continue
            
            batchData_f2, batchLabel_f2 = generator_test(PARAMS, fName_f2, clNum, PARAMS['input_shape_f2'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f2 = data_norm(batchData_f2, PARAMS, PARAMS['trnData_mean_f2'], PARAMS['trnData_max_f2'], PARAMS['trnData_min_f2'], PARAMS['trnData_std_f2'])
                              
            # ---- prediction for feat2 
            print('batchData_f2 shape:', np.shape(batchData_f2))
            pred_f2  = Train_Params_f2['model'].predict(x=batchData_f2)
            
            # adding prediction obtained from feature 1 and feature 2
            if PARAMS['alpha_flag']:
                pred = PARAMS['alpha']*pred_f1 + (1-PARAMS['alpha'])*pred_f2
                # pred = (1-PARAMS['alpha'])*pred_f1 + (PARAMS['alpha'])*pred_f2                    
            else:
                pred = pred_f1 + pred_f2


            ##################################################################
            ##################################################################
            # if more than two feat
            if PARAMS['Num_Feat_forComb'] > 2:
                
                path_f3 = '/'.join(fName_f1.split('/')[:-6])
                fName_f3 = path_f3 + '/' + '_'.join(PARAMS['FeatName_f3'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
                
                
                if not os.path.exists(fName_f3):
                    continue
                
                batchData_f3, batchLabel_f3 = generator_test(PARAMS, fName_f3, clNum, PARAMS['input_shape_f3'])
                if PARAMS['manual_Normalize_flag']:
                    batchData_f3 = data_norm(batchData_f3, PARAMS, PARAMS['trnData_mean_f3'], PARAMS['trnData_max_f3'], PARAMS['trnData_min_f3'], PARAMS['trnData_std_f3'])
                                  
                # ---- prediction for feat3 
                print('batchData_f3 shape:', np.shape(batchData_f3))
                pred_f3  = Train_Params_f3['model'].predict(x=batchData_f3)
                
                # adding prediction obtained from feature 1, feature 2 and feature 3
                if PARAMS['alpha_flag']:
                    pred = PARAMS['alpha1']*pred + (1-PARAMS['alpha1'])*pred_f3
                else:
                    pred = pred + pred_f3


            ##################################################################
                
            # # if train seg dur and test seg dur are not equal
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg'] and not PARAMS['majority_vote_flag']:
            #     pred, batchLabel = postProcessing_predCls(PARAMS, pred, batchLabel_f1)  # using mean predicted scores of a test segment

            # --- condition for num of output node
            if PARAMS['numNodeOtpt'] == 1:
                pred_lab = np.zeros(np.shape(pred))
                index = pred > 0.5
                pred_lab[index] = 1
            else:
                pred_lab = np.argmax(pred, axis=1)
                
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg']  and PARAMS['majority_vote_flag']:
            #     print('majority vote')
            #     pred_lab, batchLabel_f1 = postProcessing_majorityVote_modified(PARAMS, pred_lab, batchLabel_f1) 
                
                
            
                
            
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel_f1.tolist())            

            # if not np.shape(batchLabel_f1)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of batch labels are not equal')

            # if not np.shape(pred)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of predictions are not equal')            

            # if not np.shape(pred_lab)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of pred_lab are not equal')
                                        
            # if not np.size(PtdLabels) == np.size(GroundTruth):
            #     print('PtdLabels and GroundTruth have different size')
            #     # print('Filenmae:',fName_f1)
            #     break
                
                
            # print('pred_lab:', pred_lab)
            print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2))
            print('ground_truth: ', np.sum(batchLabel_f1==0), np.sum(batchLabel_f1==1), np.sum(batchLabel_f1==2))
            
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            
            print(PARAMS['classes'][clNum], fl, np.shape(batchData_f1), ' acc=', np.round(np.sum(pred_lab==batchLabel_f1)*100/len(batchLabel_f1), 2))

    ConfMat, fscore, precision, recall, accuracy = misc.getPerformance(PtdLabels, GroundTruth)
    print('fscore shape: ', np.shape(fscore))
    
    print('item_not_found_cntr:',item_not_found_cntr)
    return ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth
    
def test_model_v4(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3, Train_Params_f4):
    # start = time.clock()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    # startTime = time.clock()
    
    item_not_found_cntr = 0
    sentNum = 0 # added for baselines
    
    for classname in PARAMS['test_files_f1'].keys():
        clNum = class_labels[classname]
        files_f1 = PARAMS['test_files_f1'][classname]



  
        # print('test_files: ', files)

        for fl in files_f1:
            ##################################################################
            # ---- feature 1 files
            fName_f1 = fl

            count += 1
            sentNum += 1    # added for baselines
            
            # ---- load feat 1 batch data
            batchData_f1, batchLabel_f1 = generator_test(PARAMS, fName_f1, clNum, PARAMS['input_shape_f1'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f1 = data_norm(batchData_f1, PARAMS, PARAMS['trnData_mean_f1'], PARAMS['trnData_max_f1'], PARAMS['trnData_min_f1'], PARAMS['trnData_std_f1'])
                # batchData_f1 = np.transpose(batchData_f1, (0, 2, 1)) 
            # ---- prediction for feat1 
            print('batchData_f1 shape:', np.shape(batchData_f1))
            pred_f1 = Train_Params_f1['model'].predict(x=batchData_f1)


            ##################################################################

                
            path_f2 = '/'.join(fName_f1.split('/')[:-6])
            fName_f2 = path_f2 + '/' + '_'.join(PARAMS['FeatName_f2'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
            
            
            if not os.path.exists(fName_f1) or not os.path.exists(fName_f2):
                continue
            
            batchData_f2, batchLabel_f2 = generator_test(PARAMS, fName_f2, clNum, PARAMS['input_shape_f2'])
            if PARAMS['manual_Normalize_flag']:
                batchData_f2 = data_norm(batchData_f2, PARAMS, PARAMS['trnData_mean_f2'], PARAMS['trnData_max_f2'], PARAMS['trnData_min_f2'], PARAMS['trnData_std_f2'])
                # if np.shape(batchData_f2)[0] > np.shape(batchData_f1)[0]:
                #     batchData_f2 = batchData_f2[:np.shape(batchData_f1)[0],:]                               
            # ---- prediction for feat2 
            print('batchData_f2 shape:', np.shape(batchData_f2))
            pred_f2  = Train_Params_f2['model'].predict(x=batchData_f2)
            
            # # adding prediction obtained from feature 1 and feature 2
            if PARAMS['alpha_flag']:
                pred = PARAMS['alpha']*pred_f1 + (1-PARAMS['alpha'])*pred_f2
                # pred = (1-PARAMS['alpha'])*pred_f1 + (PARAMS['alpha'])*pred_f2                    
            else:
                pred = pred_f1 + pred_f2


            ##################################################################
            ##################################################################
            # if more than two feat
            if PARAMS['Num_Feat_forComb'] > 2:
                
                path_f3 = '/'.join(fName_f1.split('/')[:-6])
                fName_f3 = path_f3 + '/' + '_'.join(PARAMS['FeatName_f3'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
                
                
                if not os.path.exists(fName_f3):
                    continue
                
                batchData_f3, batchLabel_f3 = generator_test(PARAMS, fName_f3, clNum, PARAMS['input_shape_f3'])
                if PARAMS['manual_Normalize_flag']:
                    batchData_f3 = data_norm(batchData_f3, PARAMS, PARAMS['trnData_mean_f3'], PARAMS['trnData_max_f3'], PARAMS['trnData_min_f3'], PARAMS['trnData_std_f3'])
                # if np.shape(batchData_f3)[0] > np.shape(batchData_f1)[0]:
                #     batchData_f3 = batchData_f3[:np.shape(batchData_f1)[0],:]                                  
                # ---- prediction for feat3 
                print('batchData_f3 shape:', np.shape(batchData_f3))
                pred_f3  = Train_Params_f3['model'].predict(x=batchData_f3)
                
                # adding prediction obtained from feature 1, feature 2 and feature 3
                if PARAMS['alpha_flag']:
                    pred = PARAMS['alpha1']*pred + (1-PARAMS['alpha1'])*pred_f3
                else:
                    pred = pred + pred_f3


            ##################################################################
            
            
            # if more than two feat
            if PARAMS['Num_Feat_forComb'] > 3:
                
                path_f4 = '/'.join(fName_f1.split('/')[:-6])
                fName_f4 = path_f4 + '/' + '_'.join(PARAMS['FeatName_f4'].split('_')[:-3]) + '/' + '/'.join(fName_f1.split('/')[-4:])
                
                
                if not os.path.exists(fName_f4):
                    continue
                
                batchData_f4, batchLabel_f4 = generator_test(PARAMS, fName_f4, clNum, PARAMS['input_shape_f4'])
                if PARAMS['manual_Normalize_flag']:
                    batchData_f4 = data_norm(batchData_f4, PARAMS, PARAMS['trnData_mean_f4'], PARAMS['trnData_max_f4'], PARAMS['trnData_min_f4'], PARAMS['trnData_std_f4'])
                # if np.shape(batchData_f4)[0] > np.shape(batchData_f1)[0]:
                #     batchData_f4 = batchData_f4[:np.shape(batchData_f1)[0],:]                                 
                # ---- prediction for feat3 
                print('batchData_f4 shape:', np.shape(batchData_f4))
                pred_f4  = Train_Params_f4['model'].predict(x=batchData_f4)
                

                # adding prediction obtained from feature 1, feature 2 and feature 3
                if PARAMS['alpha_flag']:
                    pred = PARAMS['alpha2']*pred + (1-PARAMS['alpha2'])*pred_f4
                else:
                    pred = pred + pred_f4
            ##################################################################
                
            # # if train seg dur and test seg dur are not equal
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg'] and not PARAMS['majority_vote_flag']:
            #     pred, batchLabel = postProcessing_predCls(PARAMS, pred, batchLabel_f1)  # using mean predicted scores of a test segment

            # --- condition for num of output node
            if PARAMS['numNodeOtpt'] == 1:
                pred_lab = np.zeros(np.shape(pred))
                index = pred > 0.5
                pred_lab[index] = 1
            else:
                pred_lab = np.argmax(pred, axis=1)
                
            # if not PARAMS['trn_frmNumseg'] == PARAMS['test_frmNumSeg']  and PARAMS['majority_vote_flag']:
            #     print('majority vote')
            #     pred_lab, batchLabel_f1 = postProcessing_majorityVote_modified(PARAMS, pred_lab, batchLabel_f1) 
                
                
            
                
            
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel_f1.tolist())            

            # if not np.shape(batchLabel_f1)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of batch labels are not equal')

            # if not np.shape(pred)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of predictions are not equal')            

            # if not np.shape(pred_lab)[0] == np.shape(batchData_f1)[0]:
            #     print('batch size and num of pred_lab are not equal')
                                        
            # if not np.size(PtdLabels) == np.size(GroundTruth):
            #     print('PtdLabels and GroundTruth have different size')
            #     # print('Filenmae:',fName_f1)
            #     break
                
                
            # print('pred_lab:', pred_lab)
            print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2))
            print('ground_truth: ', np.sum(batchLabel_f1==0), np.sum(batchLabel_f1==1), np.sum(batchLabel_f1==2))
            
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            
            print(PARAMS['classes'][clNum], fl, np.shape(batchData_f1), ' acc=', np.round(np.sum(pred_lab==batchLabel_f1)*100/len(batchLabel_f1), 2))

    ConfMat, fscore, precision, recall, accuracy = misc.getPerformance(PtdLabels, GroundTruth)
    print('fscore shape: ', np.shape(fscore))
    
    print('item_not_found_cntr:',item_not_found_cntr)
    return ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth
    
    
    

def test_cnn_v2(PARAMS, Train_Params_f1, Train_Params_f2):
    
    ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth = test_model_v2(PARAMS, Train_Params_f1, Train_Params_f2)

    Test_Params = {
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels,
        'Predictions_test': Predictions,
        'GroundTruth_test': GroundTruth,
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall
        }

    return Test_Params



def test_cnn_v3(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3):
    
    ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth = test_model_v3(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3)

    Test_Params = {
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels,
        'Predictions_test': Predictions,
        'GroundTruth_test': GroundTruth,
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall
        }

    return Test_Params

def test_cnn_v4(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3, Train_Params_f4):
    
    ConfMat, fscore, precision, recall, accuracy, PtdLabels, Predictions, GroundTruth = test_model_v4(PARAMS, Train_Params_f1, Train_Params_f2, Train_Params_f3, Train_Params_f4)

    Test_Params = {
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels,
        'Predictions_test': Predictions,
        'GroundTruth_test': GroundTruth,
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall
        }

    return Test_Params