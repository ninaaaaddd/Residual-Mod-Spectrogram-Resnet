from train_helper import get_model, load_data
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


if __name__ == "__main__":
    ## Testing Example
    inf(model_path="Models/Checkpoints_cmu/best_model/best_checkpoint.pt",
        wav_path="Datasets/Combined_cmu/Test/fake_cmu/ksp_arctic_b0340_gen.wav")
