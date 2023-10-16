from train_helper import start_training


if __name__=="__main__":
    start_training(train_csv="Datasets\\Combined_cmu\\Train\\data.csv", 
                valid_csv="Datasets\\Combined_cmu\\Train\\valid_data.csv", 
                model_path="Models\\Lambani_Train_Soliga_Test",
                num_epochs=5)