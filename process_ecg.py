from utils.data_util import *
import pickle

if __name__ == '__main__':

    
    AF_DATASET_DIR = 'C:\\Users\\srivi\\OneDrive\\Desktop\\ECG-GAN\\training2017'   # AF Classification from a Short Single Lead ECG Recording - The PhysioNet Computing in Cardiology Challenge 2017
    LABEL_PATH = 'C:\\Users\\srivi\\OneDrive\\Desktop\\ECG-GAN\\training2017\\REFERENCE-original.csv'

    dataloader = DataLoader()
    print("Loading Data ...")
    ECG_AF = dataloader.load_af_challenge_db(AF_DATASET_DIR, LABEL_PATH, save=True)
    print("Processing ECG signal...")
    print("Processing AF dataset...")
    AF_hrbt = dataloader.process_signals(signals=ECG_AF, sampling_rate=300, save=True, save_name='AF_heartbeat.pkl')
    X_af, y_af = dataloader.prepare_input_challenge(AF_hrbt, save=True)
    
    # Print the sizes of the processed data
    print(f"Processed AF signals: {len(X_af)}")
    print(f"Processed AF labels: {len(y_af)}")

