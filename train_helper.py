from amplitude_modulation.am_analysis import am_analysis as ama
import librosa
import numpy as np
import os
import shutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from tqdm import tqdm
from residual import excitation
from logger import logger
## Adding openyxl for checking number of columns in Excel file
import openpyxl as op
from pathlib import Path
import multiprocessing as mp
main_dict = {}


def load_wav(audio_filepath, sr=8000):
    """Loads wav file and returns it as a numpy array

    :param audio_filepath: path to wav file
    :type audio_filepath: str
    :param sr: sampling rate. Defaults to 8000
    :type sr: int
    :return: audio data and sampling rate
    :rtype: tuple
    """
    audio_data, fs = librosa.load(audio_filepath, sr=sr)
    return audio_data, fs


def modulation_spectogram_from_wav(audio_data, fs, res, preemp):
    """Returns modulation spectogram from wav file

    :param audio_data: audio data
    :type audio_data: np.array
    :param fs: sampling rate
    :type fs: int
    :param res: whether to use residual phase or not
    :type res: bool
    :param premp: whether to use preemphasis or not
    :type premp: bool
    :return: modulation spectogram
    :rtype: np.array
    """

    x = audio_data
    x = x / np.max(x)
    if res:
        x, henv, resPhase = excitation(x, fs, preemp)
    win_size_sec = 0.04
    win_shft_sec = 0.01
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(x, fs, win_size=round(
        win_size_sec*fs), win_shift=round(win_shft_sec*fs), channel_names=['Modulation Spectrogram'])
    X_plot = ama.plot_modulation_spectrogram_data(
        stft_modulation_spectrogram, 0, modf_range=np.array([0, 20]), c_range=np.array([-90, -50]))
    return X_plot

def load_data(filepath, sr=8000, win_length=160, hop_length=80, n_mels=40, spec_len=504, res=True, preemp=False):
    """Loads data from filepath

    :param filepath: path to wav file
    :type filepath: str
    :param sr: sampling rate. Defaults to 8000
    :type sr: int
    :param win_length: window length for spectrogram. Defaults to 160
    :type win_length: int
    :param hop_length: hop length for spectrogram. Defaults to 80
    :type hop_length: int
    :param n_mels: number of mel filters. Defaults to 40
    :type n_mels: int
    :param spec_len: length of spectrogram. Defaults to 504
    :type spec_len: int
    :return: spectrogram
    :rtype: np.array
    """
    audio_data, fs = load_wav(filepath, sr=sr)
    linear_spect = modulation_spectogram_from_wav(audio_data, fs, res, preemp)
    mag_T = linear_spect
    if mag_T.shape[1] > 505:
        mag_T = mag_T[:, :505]

    shape = np.shape(mag_T)
    padded_array = np.zeros((161, 505))
    padded_array[:shape[0], :shape[1]] = mag_T
    mag_T = padded_array

    randtime = np.random.randint(0, mag_T.shape[1] - spec_len)
    spec_mag = mag_T[:, randtime:randtime + spec_len]

    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

def cus(audio_link):
    file_name = audio_link.replace(".wav","")
    file_name = file_name.replace("/","-")
    file_name = file_name.replace("\\","-")
    if not os.path.exists("temp"):
        os.makedirs("temp")
    save_path = os.path.join("temp",file_name+".npy")
    if os.path.exists(save_path):
        return 1
    else:
        spec = load_data(audio_link, sr=8000)
        # print("Spec", spec)
        try:
            spec = spec[np.newaxis, ...]
        except TypeError:
            logger.error("Error in loading file: {}".format(audio_link))
            return -1
        feat = np.asarray(spec)
        np.save(save_path, feat, allow_pickle=True)


class SpeechDataGenerator:
    """Data generator for speech data"""

    def __init__(self, manifest):
        """Initializes data generator

        :param manifest: path to manifest file
        :type manifest: str
        :return: None
        """
        self.audio_links = [line.rstrip('\n').split(
            ',')[0] for line in open(manifest)]
        self.feats = []
        global main_dict
        num_cores = 12
        # print(len(self.audio_links))    

        # Create a multiprocessing pool with the specified number of cores
        pool = mp.Pool(processes=num_cores)
        # print(self.audio_links[:5])
        # Use pool.map() to apply the load_data function to each element of self.audio_links
        pool.map(cus, np.array(self.audio_links))

        # Close the pool to free up resources
        pool.close()

        # Wait for all processes to finish
        pool.join()

        print("processing done")
        
        for audio_link in tqdm(self.audio_links):            
            file_name = audio_link.replace(".wav","")
            file_name = file_name.replace("/","-")
            file_name = file_name.replace("\\","-")
            save_path = os.path.join("temp",file_name+".npy")
            if os.path.exists(save_path):
                feat = np.load(save_path, allow_pickle=True)
                self.feats.append(feat)
            else:
                spec = load_data(audio_link, sr=8000)
                spec = spec[np.newaxis, ...]
                feat = np.asarray(spec)
                self.feats.append(feat)
                np.save(save_path, feat, allow_pickle=True)
        self.labels = [int(line.rstrip('\n').split(',')[1])
                       for line in open(manifest)]

    def __len__(self):
        """Returns length of data generator

        :return: length of data generator
        :rtype: int
        """
        return len(self.audio_links)

    def __getitem__(self, idx):
        """Returns item at index idx

        :param idx: index
        :type idx: int
        :return: features and labels
        :rtype: tuple
        """
        feats = self.feats[idx]
        class_id = self.labels[idx]
        label_arr = np.asarray(class_id)

        return feats, label_arr


def get_model(device, num_classes, pretrained=False):
    """Returns model

    :param device: device to run model on
    :type device: torch.device
    :param num_classes: number of classes
    :type num_classes: int
    :param pretrained: whether to use pretrained model. Defaults to False
    :type pretrained: bool
    :return: model
    :rtype: torch.nn.Module
    """
    model = resnet34(weights=pretrained)
    model.fc = nn.Linear(512, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device, dtype=torch.float)
    return model


def create_output_dirs(checkpoint_path):
    os.makedirs(os.path.join(checkpoint_path,
                "current_checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, "final_model"), exist_ok=True)


# Load Data
def load_data_loaders(train_manifest, valid_manifest, batch_size):
    """Returns data loaders

    :param train_manifest: path to train manifest file
    :type train_manifest: str
    :param valid_manifest: path to valid manifest file
    :type valid_manifest: str
    :param batch_size: batch size
    :type batch_size: int
    :return: data loaders
    :rtype: dict
    """
    train_data = SpeechDataGenerator(manifest=train_manifest)

    if valid_manifest != None:
        test_data = SpeechDataGenerator(manifest=valid_manifest)

    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    
    if valid_manifest != None:
        test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True)
    else:
        test_loader={}

    loaders = {
        'train': train_loader,
        'test': test_loader,
    }
    return loaders


def save_ckp(state, model, valid_loss, valid_loss_min, checkpoint_path, best_model_path, final_model_path,
             save_for_each_epoch):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    if save_for_each_epoch:
        checkpoint_name = final_model_path.split("\\")[-1]
        new_checkpoint_name = str(state['epoch']-1) + "_" + checkpoint_name
        f_path = os.path.join(final_model_path.replace(
            checkpoint_name, new_checkpoint_name))
        torch.save(model, f_path)

    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if valid_loss <= valid_loss_min:
        best_fpath = best_model_path
        torch.save(model, final_model_path)
        shutil.copyfile(f_path, best_fpath)


def train(start_epochs, n_epochs, device, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda,
          checkpoint_path, save_for_each_epoch=True):
    """Trains model

    :param start_epochs: starting epoch
    :type start_epochs: int
    :param n_epochs: number of epochs
    :type n_epochs: int
    :param device: device to run model on
    :type device: torch.device
    :param valid_loss_min_input: minimum validation loss
    :type valid_loss_min_input: float
    :param loaders: data loaders
    :type loaders: dict
    :param model: model
    :type model: torch.nn.Module
    :param optimizer: optimizer
    :type optimizer: torch.optim
    :param criterion: loss function
    :type criterion: torch.nn
    :param use_cuda: whether to use cuda
    :type use_cuda: bool
    :param checkpoint_path: path to save checkpoints
    :type checkpoint_path: str
    :param save_for_each_epoch: whether to save model for each epoch. Defaults to True
    :type save_for_each_epoch: bool
    :return: trained model
    :rtype: torch.nn.Module
    """
    # initialize tracker for minimum validation loss
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=1e-1, patience=1, verbose=True)
    valid_loss_min = valid_loss_min_input

    # create checkpoints
    path = checkpoint_path
    checkpoint_path = os.path.join(
        path, "current_checkpoint", "current_checkpoint.pt")
    best_model_path = os.path.join(path, "best_model", "best_checkpoint.pt")
    final_model_path = os.path.join(path, "final_model", "final_model.pt")

    # load checkpoint from last run if available
    if os.path.isfile(checkpoint_path):
        print("loaded model from ", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cuda') if use_cuda else torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epochs = 0  # checkpoint['epoch']
        valid_loss_min = checkpoint['valid_loss_min']

    create_output_dirs(path)

    for epoch in range(start_epochs, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_predict = []
        valid_predict = []
        train_target = []
        valid_target = []
        temp_predict = []
        temp_target = []
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(loaders['train']), total=len(loaders['train']), leave=False):
            # print('\n', 'This train iteration: ')
            # move to GPU

            target=target.type(torch.LongTensor)
            data, target = data.to(
                device, dtype=torch.float), target.to(device)
            
            
            # print(data.is_cuda)
            # find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record the average training loss, using something like
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_predict = train_predict + temp_predict
            train_target = train_target + temp_target

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in tqdm(enumerate(loaders['test']), total=len(loaders['test']), leave=False):
            # move to GPU
            # print('\n', 'This valid iteration: ')
            target=target.type(torch.LongTensor)
            data, target = data.to(
                device, dtype=torch.float), target.to(device)
            # update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss = valid_loss + \
                ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            valid_predict = valid_predict + temp_predict
            valid_target = valid_target + temp_target

        # calculate average losses
        train_loss = train_loss / len(loaders['train'])
        valid_loss = valid_loss / len(loaders['test'])
        train_acc = accuracy_score(train_target, train_predict)
        valid_acc = accuracy_score(valid_target, valid_predict)
        valid_f1_score = f1_score(valid_target, valid_predict, average='macro')
        print("Classification Report: {}".format(classification_report(valid_target, valid_predict, output_dict=False,
                                                                       labels=np.unique(valid_predict))))
        # print training/validation statistics
        print(
            'Epoch: {} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.10f} \tValidation  Accuracy: {:.6f} '.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))
        logger.debug(
            'Epoch: {} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.10f} \tValidation  Accuracy: {:.6f} '.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step(valid_loss)
        # save checkpoint
        save_ckp(checkpoint, model, valid_loss, valid_loss_min, checkpoint_path, best_model_path, final_model_path,
                 save_for_each_epoch)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).    Model Saved......'.format(
                valid_loss_min, valid_loss))
            # save_ckp(checkpoint, model, True, checkpoint_path, best_model_path, final_model_path)
            valid_loss_min = valid_loss
    return model


# Set Device
def start_training(train_csv, valid_csv, model_path, save_for_each_epoch=True,
                cuda=True, batch_size=10, num_epochs=1, learning_rate=0.01,
                num_classes=2):
    """For training the model

    :param train_csv: path to csv file containing training data
    :type train_csv: str
    :param valid_csv: path to csv file containing validation data
    :type valid_csv: str
    :param model_path: path to save model
    :type model_path: str
    :param save_for_each_epoch: save the model after each epoch
    :type save_for_each_epoch: bool
    :param cuda: use cuda or not
    :type cuda: bool
    :param batch_size: batch size for training
    :type batch_size: int
    :param num_epochs: number of epochs for training
    :type num_epochs: int
    :param learning_rate: learning rate for training
    :type learning_rate: float
    :param num_classes: number of classes
    :type num_classes: int
    :return: trained model
    :rtype: torch.nn.Module
    """
    use_cuda = cuda
    # device = "cpu"
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        use_cuda = torch.cuda.is_available()

    train_manifest = train_csv
    valid_manifest = valid_csv
    checkpoint_path = model_path

    # Load_Data
    loaders = load_data_loaders(train_manifest, valid_manifest, batch_size)

    # Load Model
    model = get_model(device, num_classes, pretrained=False)
    # Set model hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(device)
    train(1, num_epochs, device, np.Inf, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
          save_for_each_epoch)
