# Residual-Modulation-Spectrogram
This repository contains code for the following research paper <link for research paper>. 
Here we used residual modulation spectrogram on three datasets LJ Speech, Libritts, and CMU-arctic to build a model capable of segregating fake and bonafide speech.
In the spectrogram, since the length of the x-axis can vary depending on the length of the audio file we are taking only the first 505 columns. In case the number is less than 505, we will populate the rest of the array with 0.

# Pre-requisites
- python - 3.11
- packages mentioned in requirements.txt (in case of any issues you can try installing from base_req.txt)
  <br> To install run ''' pip install -r requirements.txt '''
- Preferably a system with cuda enabled, RAM>=4gb, CPU >= Intel core i3
- The dataset directory should be in the format
  - fake folder -> all files containing the fake speech (.wav format)
  - orig folder -> all files containing the bonafide speech (.wav format)
# Links for the resources used in the research paper
- Models -> <a href= "https://drive.google.com/drive/folders/1-6wf9VIW17at2WqDXgRIlMnNhdM0nkKR?usp=drive_link"> click here </a>
- Datasets -> <a href="https://drive.google.com/drive/folders/1AzbVFs3tPC9svvL8FAcoWfTZfPSYvsj5?usp=drive_link"> click here </a>

# Running the code
## To train
- run data_file_generator.py. This generates the csv file containing the wav file paths and the labels required to train the model.
  <br> Arguments
  - path -> The path for the train/test folder.
  - valid -> Whether to generate the validation dataset from the same wav files. (25% validation). Defaults to False
- Once this is done run the train.py file. It requires three arguments
  <br> Arguments
  - train_csv -> Path to training csv file
  - valid_csv -> Path to validation csv file
  - model_path -> path for saving the model
### Default Parameters
  - save_for_each_epoch=True,
  - cuda=True
  - batch_size=10
  - num_epochs=20
  - learning_rate=0.01
  - num_classes=2 (0 -> fake, 1->real)

  This can be modified in the train.py files by adding the arguments

## To test
- Run single file inference.py
   <br> Arguments
  - model_path -> The path for the model to be used
  - wav_path -> The path for the wav file

  A sample input has been added in both the training and testing file.

