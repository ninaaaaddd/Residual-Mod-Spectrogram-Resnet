#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:57:05 2022

@author: eee
"""


import numpy as np
# from getData import getDataDetails
# import misc 
# import cnn_classifier as CNN
import datetime
import os
import matplotlib.pyplot as plt
import tsne_Plot as tsne

from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d




# ---- parameters initializations
fs = 8000                   # sampling rate
frmsSzN = int((25*fs)/1000)  # frame size in ms   need to change
frmShN = int((10*fs)/1000)   # frame shift in ms
NumFilter_f1 = 161              # featutre dimension 
NumFilter_f2 = 64              # featutre dimension 
Num_Feat_forComb = 1      # num of features used for classification

# ------ Model: parametrs
batchsize = int(33/3)        # batch size
epochNum = 50                # epoch numbers
numNodeOtpt = 3              # num of output nodes in classifier architecture
whichModel = 'cnn_gru_attn'  # name of classifier 


classes={0:'real', 1:'fake'}

# ---- train parameters
trn_segDurLevel = '1000ms'                                      # segment duration label for training
trn_segDur = 1000 # in ms                                       # segment duration in ms for training
trn_segDurN = int((trn_segDur*fs)/1000)                         # segment duration in samples for training
trn_frmNumSeg = int(np.floor((trn_segDurN-frmsSzN)/frmShN)+1)   # number of frames in a train segment duration
trn_segShft = int(np.floor(trn_frmNumSeg))                 # num of frames by which a train segment is shifting 

# ---- test parametrs
test_segDur = 1000  # in ms                                      # test segment duration in ms
test_segDurN = int((test_segDur*fs)/1000)                        # test segment duration in samples
test_frmNumSeg = int(np.floor((test_segDurN-frmsSzN)/frmShN)+1)  # number of frames in a test segment duration
test_segShft = 25 #int(np.floor(test_frmNumSeg/2))                 # num of frames by which a test segment is shifting ##### changed from 1 to 25


input_shape_f1 = (NumFilter_f1, trn_frmNumSeg, 1)                      # shape of input to the classifier
input_shape_f2 = (NumFilter_f2, trn_frmNumSeg, 1)                      # shape of input to the classifier
OHE_lab_shape = (trn_frmNumSeg, len(classes))  # ????????????


# -- different flags
alpha = 0.5
alpha_flag = 1
majority_vote_flag = 0
feat_nfeatXnFrame_flag = 0
manual_Normalize_flag = 1
# norm_method = 'mean_max'
norm_method = 'zscore'
use_GPU = False
OHE_flag = 1

# --- path to saved model
OtPtDir = '/home/eee/PhD/..../'


# ---- different parameters
PARAMS = {'today': datetime.datetime.now().strftime("%Y-%m-%d"),
          'classes': classes,

          'opDir': OtPtDir,
          'Num_Feat_forComb': Num_Feat_forComb,
          'alpha': alpha,
          'First_layer_kernel': "First_layer_kernel",
        #   'First_layer_kernel': First_layer_kernel,
          'Dense_layer': "Dense_layer",
        #   'Dense_layer': Dense_layer,

          'frmsSzN': frmsSzN,
          'frmShN': frmShN,

          'NumFilter_f1': NumFilter_f1,
          'input_shape_f1': input_shape_f1,

          'NumFilter_f2': NumFilter_f2,
          'input_shape_f2': input_shape_f2,

          'OHE_lab_shape': OHE_lab_shape,
          'batch_size': batchsize,
          'epochs': epochNum,
          'numNodeOtpt': numNodeOtpt,
          'which_model': whichModel,

          'Segment_dur': trn_frmNumSeg,
          'CNN_patch_size': trn_frmNumSeg,
          'CNN_patch_shift': trn_segShft,
          'trn_frmNumseg': trn_frmNumSeg,
          'trn_segDurLevel': trn_segDurLevel,

          'test_frmNumSeg': test_frmNumSeg,
          'CNN_patch_shift_test': test_segShft,
          'test_segShft': test_segShft,

          'train_steps_per_epoch': 0,
          'val_steps': 0,
          'test_steps': 0,

          'save_flag': 1,
          'alpha_flag': alpha_flag,
          'majority_vote_flag': majority_vote_flag,
          'feat_nfeatXnFrame_flag': feat_nfeatXnFrame_flag,
          'data_generator': 0,
          'OHE_flag': OHE_flag,
          'use_GPU': use_GPU,

          'manual_Normalize_flag': manual_Normalize_flag,
          'norm_method': norm_method
          }
    
print("here")

# Insert the code to read the wavfiles, calculate features, load the model
from torchvision.models import resnet34
import torch
import torch.nn as nn
import librosa
import scipy
from scipy.signal import lfilter, hamming,resample
from scipy.signal.windows import hann
import numpy as np
from glob import glob
class SpeechDataGenerator:

    def __init__(self, manifest):
        self.audio_links = glob(manifest+"/**.wav")
        # self.audio_links = [line.rstrip('\n').split(',')[0] for line in open(manifest)]
        if manifest.split("/")[-1] == "fake_com":
           self.labels = [0]*len(self.audio_links)
        else:
           self.labels = [1]*len(self.audio_links)

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        spec = load_data(audio_link)[np.newaxis, ...]
        feats = np.asarray(spec)
        label_arr = np.asarray(class_id)

        return feats, label_arr

    def get_features(self, idx):
        audio_link = self.audio_links[idx]
        spec = load_data(audio_link, res=True)[np.newaxis, ...]
        feats = np.asarray(spec)
        return feats 
    
def get_model(device, num_classes, pretrained=False):
    model = resnet34(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2,2), padding=(3,3), bias=False)
    model.to(device, dtype=torch.float)
    return model
# model = get_model("cpu", 2, pretrained=False)
# checkpoints = torch.load("Models/Checkpoints_LJ"+"/best_model/best_checkpoint.pt",map_location=torch.device('cpu'))
# model.load_state_dict(checkpoints['state_dict'])
import scipy
import math
import sys
sys.path.insert(0, '/home/devesh/Desktop/Rnd/amplitude-modulation-analysis-module')
from am_analysis import am_analysis as ama
def excitation(sample,fs):
  # if fs!=8000:
  #   sample=resample(sample,8000)+0.00001
  #   fs=8000

  lporder=10
  residual=LPres(sample,fs,20,10,lporder,0)
  henv = np.abs(scipy.signal.hilbert(residual))
  resPhase=np.divide(residual, henv)
  return residual, henv, resPhase



def ResFilter_v2(PrevSpFrm,SpFrm,FrmLPC,LPorder,FrmSize,plotflag):
  # print('This is getting executed')
  ResFrm=np.asarray(np.zeros((1,FrmSize)))
  ResFrm=ResFrm[0,:]
  # print('b: ', (ResFrm))
  tempfrm=np.zeros((1,FrmSize+LPorder))
  # print(np.shape(tempfrm))
  # tempfrm[1:LPorder]=PrevSpFrm
  # #tempfrm(1:FrmSize)=PrevSpFrm(1:FrmSize);
  # tempfrm[LPorder+1:LPorder+FrmSize]=SpFrm[1:FrmSize]


  temp_PrevSpFrm=np.asmatrix(PrevSpFrm)
  temp_SpFrm=np.asmatrix(SpFrm[:FrmSize])
  if (np.shape(temp_PrevSpFrm)[0]==1):
    temp_PrevSpFrm=temp_PrevSpFrm.T
  if (np.shape(temp_SpFrm)[0]==1):
    temp_SpFrm=temp_SpFrm.T

  # print(np.shape(temp_PrevSpFrm))
  # print(np.shape(temp_SpFrm))
  tempfrm=np.concatenate((temp_PrevSpFrm, temp_SpFrm))
  tempfrm=np.asarray(tempfrm)[:,0]
  # print((tempfrm))
  # print(np.shape(tempfrm))


  for i in range(FrmSize):
    t=0
    for j in range(LPorder):
      # print(FrmLPC[j+1], tempfrm[-j+i+LPorder-1])
      # print(FrmLPC[j+1])
      t=t+FrmLPC[j+1]*tempfrm[-j+i+LPorder-1]

    ResFrm[i]=SpFrm[i]-(-t)

  return ResFrm

def LPres(speech ,fs, framesize, frameshift,lporder, preemp):
  if (framesize>50):
    print("Warning!")
  else:
   # Converting unit of variuos lengths from 'time' to 'sample number'
   Nframesize	= round(framesize * fs / 1000)
   Nframeshift	= round(frameshift * fs / 1000)
   Nspeech 	= len(speech)

   #Transpose the 'speech' signal if it is not of the form 'N x 1'
   speech=speech.reshape(Nspeech,1)
	#speech = speech(:); % Make it a column vector
  #PREEMPHASIZING SPEECH SIGNAL
#   if (preemp != 0):
#     speech = preemphasize(speech)
  #COMPUTING RESIDUAL
  res = np.asarray(np.zeros((Nspeech,1)))[:,0]

  #NUMBER OF FRAMES
  lporder=int(lporder)
  nframes=math.floor((Nspeech-Nframesize)/Nframeshift)+1
  j	= 1
  for i in range(0,Nspeech-Nframesize,Nframeshift):
    SpFrm	= speech[i:i+Nframesize]

    winHann =  np.asmatrix(hann(Nframesize))
    y_frame = np.asarray(np.multiply(winHann,SpFrm.T))

    lpcoef	= librosa.lpc(y=y_frame[0,:],order=lporder)

    if(i <= lporder):
      PrevFrm=np.zeros((1,lporder))
    else:
      # print('i: ', i)
      PrevFrm=speech[(i-lporder):(i)]
    ResFrm	= ResFilter_v2(np.real(PrevFrm),np.real(SpFrm),np.real(lpcoef),lporder,Nframesize,0)

    res[i:i+Nframeshift]	= ResFrm[:Nframeshift]
    j	= j+1

  res[i+Nframeshift:i+Nframesize]	= ResFrm[Nframeshift:Nframesize]
    #PROCESSING LASTFRAME SAMPLES,
  if(i < Nspeech):
    SpFrm	= speech[i:Nspeech]


    winHann =  np.asmatrix(hamming(len(SpFrm)))
    y_frame = np.asarray(np.multiply(winHann,SpFrm.T))
    lpcoef	= librosa.lpc(y=y_frame[0,:],order=lporder)
    # print(lpcoef)
    PrevFrm	= speech[(i-lporder):(i)]
    ResFrm	= ResFilter_v2(np.real(PrevFrm),np.real(SpFrm),np.real(lpcoef),lporder,Nframesize,1)
    # print(ResFrm)
    res[i:i+len(ResFrm)]	= ResFrm[:len(ResFrm)]
    j	= j+1
  hm	= hamming(2*lporder)
  for i in range(1,round(len(hm)/2)):
    res[i]	= res[i] * hm[i]      #attenuating first lporder samples
  return res


def modulation_spectogram_from_wav(audio_data,fs):
    x=audio_data
    x = x / np.max(x)
    # x=preemphasize(x)
    # residual, henv, resPhase = excitation(x, fs)
    residual=x
    win_size_sec = 0.04
    win_shft_sec = 0.01
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(residual, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = ['Modulation Spectrogram'])
    # print(stft_modulation_spectrogram)
    X_plot=ama.plot_modulation_spectrogram_data(stft_modulation_spectrogram, 0 , modf_range = np.array([0,20]), c_range =  np.array([-90, -50]))
    # print(type(X_plot))
    return X_plot

def res_modulation_spectogram_from_wav(audio_data,fs):
    x=audio_data
    x = x / np.max(x)
    # x=preemphasize(x)
    residual, henv, resPhase = excitation(x, fs)
    win_size_sec = 0.04
    win_shft_sec = 0.01
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(residual, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = ['Modulation Spectrogram'])
    # print(stft_modulation_spectrogram)
    X_plot=ama.plot_modulation_spectrogram_data(stft_modulation_spectrogram, 0 , modf_range = np.array([0,20]), c_range =  np.array([-90, -50]))
    # print(type(X_plot))
    return X_plot

def load_wav(audio_filepath, sr, min_dur_sec=5):
     audio_data, fs = librosa.load(audio_filepath, sr=8000)
     len_file = len(audio_data)

    #  if len_file <= int(min_dur_sec * sr):
    #     temp = np.zeros((1, int(min_dur_sec * sr) - len_file))
    #     joined_wav = np.concatenate((audio_data, temp[0]))
    #  else:
    #     joined_wav = audio_data


     return audio_data,fs#joined_wav,fs

import tqdm

def load_data(filepath, sr=8000, min_dur_sec=5, win_length=160, hop_length=80, n_mels=40, spec_len=504, res=False):
    import wave
    audio_data,fs = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    # raw = wave.open(filepath)
    if res == False:
      linear_spect = modulation_spectogram_from_wav(audio_data,fs)
    elif res == True:
       linear_spect = res_modulation_spectogram_from_wav(audio_data,fs)
    # mag, _ = librosa.magphase(linear_spect)
    # mag = np.log1p(mag)
    mag_T = linear_spect
    if mag_T.shape[1]>505:
      mag_T=mag_T[:,:505]
    shape = np.shape(mag_T)
    padded_array = np.zeros((161, 505))
    padded_array[:shape[0],:shape[1]] = mag_T
    mag_T=padded_array
    randtime = np.random.randint(0, mag_T.shape[1] - spec_len)
    spec_mag = mag_T[:, randtime:randtime + spec_len]

    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

weight_mod, weight_res = 0, 1
real_path = "/home/devesh/Desktop/Rnd/Datasets/all_real"
fake_path = "/home/devesh/Desktop/Rnd/Datasets/all_fake"
print("fake loading")
fake_obj = SpeechDataGenerator(manifest=fake_path)
print("real loading")
real_obj = SpeechDataGenerator(manifest=real_path)
# fake_feat_all=[]
# for i in range(len(fake_obj.audio_links)):
#     feats = fake_obj.get_features(i)
#     fake_feat_all.append(feats)
train_obj = SpeechDataGenerator(manifest="/home/devesh/Desktop/Rnd/Datasets/Combined_LJ/all")
# train_feat=[]
# for i in tqdm.tqdm(range(len(train_obj.audio_links))):
#     feats = train_obj.get_features(i)
#     train_feat.append(feats)
    
    # ........tsne on test files   

tot_real_files = np.size(real_obj.audio_links)    #insert the test file features
tot_fake_files = np.size(fake_obj.audio_links)
# np.save("/home/devesh/Desktop/Rnd/res_TTS_train.npy", np.array(train_feat))
train_feat1 = np.load("/home/devesh/Desktop/Rnd/res_LJ_train.npy")
# train_feat2 = np.load("/home/devesh/Desktop/Rnd/score_TTS_train.npy")
# print(train_feat1.shape)
# train_feat = weight_res*train_feat1 + weight_mod*train_feat2
# train_feat = 
train_feat = train_feat1

if(train_feat.shape != train_feat1.shape):
   raise ValueError
numDim_numFrames_flag = 1
num_tsne_plots = 1

real_files = int(np.round(tot_real_files/num_tsne_plots))
fake_files = int(np.round(tot_fake_files/num_tsne_plots))


featName = 'LJ Score level Combination (Res + Mod)'
# trainfeatall -> mix all for training both real and fake

#--------------------------------    
#Example for stats calculation mean, std, max, min of thr train features
feat_mean = np.expand_dims(np.mean(train_feat, axis = 0), axis = 0) 
feat_std = np.expand_dims(np.std(train_feat, axis = 0), axis = 0)
feat_max = np.expand_dims(np.max(train_feat, axis = 0), axis = 0)
feat_min = np.expand_dims(np.min(train_feat, axis = 0), axis = 0)

mean_f1 = feat_mean.astype(np.float32)
std_f1 = feat_std.astype(np.float32)
max_f1 = feat_max.astype(np.float32)
min_f1 = feat_min.astype(np.float32)
embed_mod = None
embed_res = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device ="cpu"

num_classes = 2
model_res = get_model(device, num_classes, pretrained=False)
checkpoints_res = torch.load("/home/devesh/Desktop/Rnd/Models/Checkpoints_res_LJ/best_model/best_checkpoint.pt",map_location=torch.device('cpu'))
model_res.load_state_dict(checkpoints_res['state_dict'])

model_mod = get_model(device, num_classes, pretrained=False)
checkpoints_mod = torch.load("/home/devesh/Desktop/Rnd/Models/Checkpoints_LJ/best_model/best_checkpoint.pt",map_location=torch.device('cpu'))
model_mod.load_state_dict(checkpoints_mod['state_dict'])

print("started")
def hook_fn_mod(module, input, output):
    # Save the output (embedding) to a variable
    global embed_mod
    # all_data = np.append(all_data, output, axis = 0)
    # print(output.shape)
    embed_mod = output[:,:,0,0]

# Get the second-to-last layer
second_last_layer_mod = None
for name, module in model_mod.named_children():
    if isinstance(module, nn.Sequential):
        for layer_name, layer in module.named_children():
            if not isinstance(layer, nn.Sequential):
                second_last_layer_mod = layer

print("started")
def hook_fn_res(module, input, output):
    # Save the output (embedding) to a variable
    global embed_res
    # all_data = np.append(all_data, output, axis = 0)
    # print(output.shape)
    embed_res= output[:,:,0,0]

# Get the second-to-last layer
second_last_layer_res= None
for name, module in model_res.named_children():
    if isinstance(module, nn.Sequential):
        for layer_name, layer in module.named_children():
            if not isinstance(layer, nn.Sequential):
                second_last_layer_res = layer

# If the second-to-last layer is not found, raise an error
if second_last_layer_res is None or second_last_layer_mod is None:
    raise ValueError("Second-to-last layer not found!")

# Register the forward hook to the second-to-last layer
hook_handle_mod = second_last_layer_mod.register_forward_hook(hook_fn_mod)
hook_handle_res = second_last_layer_res.register_forward_hook(hook_fn_res)
def gen(paths):
    all_data=[]
    i=0
    global model_res, model_mod
    for path in tqdm.tqdm(paths):
        spec=load_data(path, res = True)[np.newaxis, ...]
        feats = np.asarray(spec)
        feats = torch.from_numpy(feats)
        feats = feats.unsqueeze(0)

        feats = feats.to(device)
        label = model_res(feats.float())

        spec=load_data(path, res = False)[np.newaxis, ...]
        feats = np.asarray(spec)
        feats = torch.from_numpy(feats)
        feats = feats.unsqueeze(0)

        feats = feats.to(device)
        label = model_mod(feats.float())
        if i == 0:
            all_data = weight_mod*embed_mod.detach().numpy() + weight_res*embed_res.detach().numpy()
            i=1
        else:
            all_data = np.append(all_data, weight_mod*embed_mod.detach().numpy() + weight_res*embed_res.detach().numpy(), axis = 0)
    return all_data
      

# print("\n\n full model arch\n",model)
#----------------------------------
for i in range(num_tsne_plots):
    endi_real = 0
    endi_fake = 0
    if i == 0:
        init_real = i
        init_fake = i
    else:
        init_real = endi_real
        init_fake = endi_fake
        
    endi_real = init_real + real_files
    endi_fake = init_fake + fake_files
    
# ---- for individual feature
        # here Train_Params_f1 is the pretrained model of the first feature (1. mod spec 2. residual)
    # inx = list(range(init_real,endi_real))

    # real_data = tsne.load_data(["/home/devesh/Desktop/Rnd/npy files/validation/251_137823_000047_000000_gen.wav.npy"], [0], numDim_numFrames_flag, mean_f1, std_f1, max_f1, min_f1, PARAMS, model)
    real_data=gen(real_obj.audio_links)
    print("real_done")
    # inx = list(range(init_fake, endi_fake))
    # fake_data = tsne.load_data(["/home/devesh/Desktop/Rnd/npy files/test/84_121550_000256_000000_gen.wav.npy"], inx, numDim_numFrames_flag, mean_f1, std_f1, max_f1, min_f1, PARAMS, model)
    fake_data=gen(fake_obj.audio_links)
    print("fake done")
    # real_data = tsne.load_data("xzc", inx, numDim_numFrames_flag, mean_f1, std_f1, max_f1, min_f1, PARAMS, model)
    
    # fake_data = tsne.load_data(fake_feat_all, inx, numDim_numFrames_flag, mean_f1, std_f1, max_f1, min_f1, PARAMS, model)

#------------------------   

    # # --- for feature combination
    # here Train_Params_f2 is the pretrained model of the second feature

    # inx = list(range(init_real,endi_real))
    # real_data = tsne.load_data_bothFeat(testfile_real, inx, numDim_numFrames_flag, PARAMS, Train_Params_f1, Train_Params_f2)
    
    # inx = list(range(init_fake, endi_fake))
    # fake_data = tsne.load_data_bothFeat(testfile_fake, inx, numDim_numFrames_flag, PARAMS, Train_Params_f1, Train_Params_f2)

#--------------------------


    real_data = np.array(real_data)
    fake_data = np.array(fake_data)
    real_data1 = real_data
    fake_data1 = fake_data

    
    data = np.append(real_data1, fake_data1, axis = 0)
    print(real_data1.shape)
    print(fake_data1.shape)
    print(data.shape)
    # Norm_data = preprocessing.scale(data, axis=0)
    label = np.append(np.zeros((np.shape(real_data1)[0],1)), np.ones((np.shape(fake_data1)[0],1)), axis = 0)


    # DIM = np.array(list(range(0,np.shape(data)[1])))

    # data2 = data[:, DIM]

    tsnes = TSNE(
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

    data_embedded = tsnes.fit_transform(data)
    print(data_embedded.shape)
    
    
    inx = list(range(0,np.shape(data)[0]))
    inx1 = inx[:np.shape(real_data1)[0]]
    inx2 = inx[np.shape(real_data1)[0]: np.shape(real_data1)[0] + np.shape(fake_data1)[0]]

    
    real_emb = data_embedded[inx1,:]
    fake_emb = data_embedded[inx2,:]

            
    from scipy.io import savemat
    tsne_emb_com = {}
    tsne_emb_com['real_emb_com'] = real_emb
    tsne_emb_com['fake_emb_com'] = fake_emb


    savemat('tsne_emb_com.mat', tsne_emb_com)
    
    plt.figure()
    plt.scatter(x=data_embedded[inx1,0], y=data_embedded[inx1,1], c='c', label='Real')
    plt.scatter(x=data_embedded[inx2,0], y=data_embedded[inx2,1], c='b', label='Fake')

    plt.legend()
    #plt.xlabel('Frequency bins')
    #plt.ylabel('Likelihood')
    # plt.title('TSNE plot of proposed '+featName+' feature (random '+str(2*numPts)+' points)')
    plt.title(featName)
    plt.savefig("/home/devesh/Pictures/res_LJ_base_res.png")
    # plt.show()

## Score_tts  = base mod
## score_tts_base_res = base res
## score_tts_base_combo = base combo
    
    # # Creating figure
    # fig = plt.figure()
    # ax = plt.axes(projection ="3d")
    
    # # plt.scatter(data_embedded[inx1,0], data_embedded[inx1,1], data_embedded[inx1,2], c='c', label='Real')
    # # plt.scatter(data_embedded[inx2,0], data_embedded[inx2,1], data_embedded[inx2,2], c='b', label='Fake')
    # plt.scatter(data_embedded[inx1,0], data_embedded[inx1,1], c='c', label='Real')
    # plt.scatter(data_embedded[inx2,0], data_embedded[inx2,1], c='b', label='Fake')

    # # ax.legend()
    # #plt.xlabel('Frequency bins')
    # #plt.ylabel('Likelihood')
    # plt.title('TSNE plot of proposed '+featName+' feature (random '+str(2*numPts)+' points)')
    # # plt.title(featName)
    # plt.show()    



