from train_helper import get_model, load_data, load_data_loaders
import torch
import numpy as np
from score_level import get_probs
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


def get_features(wav_path, device):
    """Get features from wav file

    :param wav_path: path to wav file
    :type wav_path: str
    :param device: device to load model on
    :type device: torch.device
    :return: features
    :rtype: torch.Tensor
    """
    spec = load_data(wav_path)[np.newaxis, ...]
    feats = np.asarray(spec)
    feats = torch.from_numpy(feats)
    feats = feats.unsqueeze(0)
    feats = feats.to(device)
    return feats


def inf(model_path, wav_path):
    """Inference on single file

    :param model_path: path to model checkpoint
    :type model_path: str
    :param wav_path: path to wav file
    :type wav_path: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = load_model(model_path, device)
    feats = get_features(wav_path, device)
    label = model(feats.float())
    _, pred = label.max(1)
    likelihood = label.detach().numpy()
    probs = get_probs(likelihood)
    print(f"Predicted: {labels[0]}, Probability: {probs[0]:.4f}")
    print(f"Predicted: {labels[1]}, Probability: {probs[1]:.4f}")
    print("Output: ", labels[pred.item()])

import math
from eer import compute_eer
def softmax_func(value1,value2):
        deno=math.e**value1+math.e**value2
        return [math.e**value1/deno, math.e**value2/deno]

def test_inf(model_path, test_path):
    """Inference on multiple files
    
    :param model_path: path to model checkpoint
    :type model_path: str
    :param test_path: path to excel test file
    :type test_path: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = load_model(model_path, device)
    loaders = load_data_loaders(test_path, None, batch_size=1)
    actual = []
    predicted = []
    import sklearn
    from tqdm import tqdm
    likelihood = []
    for batch_idx, (data, target) in tqdm(enumerate(loaders["train"])):
        data, target = data.to(device), target.to(device)
        output = model(data.float())
        actual.append(target.item())
        probs = softmax_func(output[0][0],output[0][1])
        likelihood.append([probs[0].item(),probs[1].item()])
        _, pred = output.max(1)
        predicted.append(pred.item())
        del data, target, output
        # print("Output: ", labels[pred.item()])
    # print(likelihood.shape)
    actual = np.array(actual)
    zero_pos = actual==0
    one_pos = actual==1
    likelihood = np.array(likelihood)
    value_1 = likelihood[:,1]
    bona_cm= value_1[one_pos]
    spoof_cm = value_1[zero_pos]
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    logger.error(eer_cm)
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    logger.error(sklearn.metrics.accuracy_score(actual, predicted))
    del model

from logger import logger
if __name__ == "__main__":
    ## Testing Example
    models = ["C:\\Users\\Ninad\\OneDrive\\Desktop\\Modulation_Spectrogram_resnet\\Residual-Modulation-Spectrogram\\Models\\Lambani_Train_Soliga_Test\\best_model\\best_checkpoint.pt"]
    datasets= ["C:\\Users\\Ninad\\OneDrive\\Desktop\\Modulation_Spectrogram_resnet\\Residual-Modulation-Spectrogram\\Datasets\\Combined_cmu\\Train\\data.csv"]
    for model in models:
         for dataset in datasets:
            logger.error(model +"  "+ dataset)
            test_inf(model_path=model,
            test_path=dataset)
