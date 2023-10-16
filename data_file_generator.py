import pandas as pd
from glob import glob
from joblib import Parallel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def generate_csv(path, valid = False):
    """ Generates the csv files for the dataset
    
    :param path: path of the dataset 
    :type path: path or str
    :param valid: whether to build validation from this or not. Defaults to False
    :type valid: boolean
    :return: None
    """
    audio_files = glob(path + '\\**\\*.wav', recursive=True)

    audio_data = pd.DataFrame({'path': audio_files})

    label_encoder = LabelEncoder()
    audio_data['speaker'] = audio_data['path'].apply(lambda x: x.split('\\')[-2]) ## Here 7 is the string number after spliting Train & Test path
    audio_data['label'] = label_encoder.fit_transform(audio_data['speaker'])

    audio_data.to_csv(path + '\\audio_data.csv', index=False, header=False)

    if valid:
        X_train, X_test, y_train, y_test = train_test_split(audio_data['path'], audio_data['label'], test_size=0.25, shuffle=True, random_state=42, stratify=audio_data['label'])
    else:
        X_train, y_train = audio_data['path'], audio_data['label']

    train_data = pd.DataFrame({'path': X_train, 'label': y_train})
    if valid:
        valid_data = pd.DataFrame({'path': X_test, 'label': y_test})

    train_data.to_csv(path+'\\data.csv', index=False, header=False)
    if valid:
        valid_data.to_csv(path+'\\valid_data.csv', index=False, header=False)

if __name__ == "__main__":
    generate_csv(path="Datasets\\Combined_cmu\\Train", valid=True)
    generate_csv(path="Datasets\\Combined_cmu\\Test", valid=False)
    print("CSV files generated successfully!")


