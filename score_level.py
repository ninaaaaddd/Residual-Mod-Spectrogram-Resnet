import math
import numpy as np
import torch
from train_helper import get_model, load_data
labels = {
    0: "Fake",
    1: "Real"
}


def load_model(model_path, device, num_classes=2):
    """Load model from checkpoint

    :param model_path: path to checkpoint
    :type model_path: str
    :param device: device to load model on
    :type device: torch.device
    :param num_classes: number of classes, defaults to 2
    :type num_classes: int, optional
    :return: model
    :rtype: torch.nn.Module
    """
    model = get_model(device, num_classes, pretrained=False)
    checkpoints = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoints['state_dict'])
    model.eval()
    return model


def get_features(wav_path, device, res=True):
    """Get features from wav file

    :param wav_path: path to wav file
    :type wav_path: str
    :param device: device to load model on
    :type device: torch.device
    :return: features
    :rtype: torch.Tensor
    """
    spec = load_data(wav_path, res=res)[np.newaxis, ...]
    feats = np.asarray(spec)
    feats = torch.from_numpy(feats)
    feats = feats.unsqueeze(0)
    feats = feats.to(device)
    return feats


def inf(modulation_model_path, residual_model_path, weights, wav_path):
    """Inference on single file

    :param modulation_model_path: path to modulation model checkpoint
    :type modulation_model_path: str
    :param residual_model_path: path to residual model checkpoint
    :type residual_model_path: str
    :param weights: weights for score level fusion
    :type weights: list
    :param wav_path: path to wav file
    :type wav_path: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = load_model(modulation_model_path, device)
    feats = get_features(wav_path, device, res=False)
    label = model(feats.float())
    _, pred = label.max(1)
    likelihood = label.detach().numpy()
    probs_mod = get_probs(likelihood)

    model = load_model(residual_model_path, device)
    feats = get_features(wav_path, device)
    label = model(feats.float())
    _, pred = label.max(1)
    likelihood = label.detach().numpy()
    probs_res = get_probs(likelihood)
    total_probs = weights[0]*np.array(probs_mod) + weights[1]*np.array(probs_res)
    print(labels[total_probs.argmax()] + " probability->" + str(total_probs.max()))
    print(labels[total_probs.argmin()] + " probability->" + str(total_probs.min()))


def softmax_func(value1,value2):
    deno = math.e**value1 + math.e**value2
    return math.e**value1/deno, math.e**value2/deno


def get_probs(likelihood):
    """ Get the probabilities from the likelihood
    
    :param likelihood: likelihood
    :type likelihood: np.array
    :return: probability of fake, probability of real
    :rtype: float, float
    """
    prob_0, prob_1 = softmax_func(likelihood[0][0], likelihood[0][1])
    return prob_0, prob_1


if __name__ == "__main__":
    ## Testing Example
    inf(modulation_model_path="Models/Checkpoints_cmu/best_model/best_checkpoint.pt",
        residual_model_path="Models/Checkpoints_res_cmu/best_model/best_checkpoint.pt",
        wav_path="/home/devesh/Desktop/Rnd/Residual-Modulation-Spectrogram/Datasets/Combined_cmu/Train/orig_cmu/arctic_a0019.wav",
        weights=[0.8, 0.2])