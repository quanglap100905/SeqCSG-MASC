import os

class Config:
    # Paths
    RAW_FILE = '/kaggle/input/datasets/tisdang/hotel-review-splitted/text_image_dataset.json'
    TRIPLET_CSV = '/kaggle/input/datasets/quanglapnguyen/top5-triples/top5_triples.csv'
    IMG_DIR = '/kaggle/working/images'
    
    PROCESSED_DIR = '/kaggle/working/hotel_data'
    TRAIN_JSON = os.path.join(PROCESSED_DIR, 'train_masc.json')
    TEST_JSON = os.path.join(PROCESSED_DIR, 'test_masc.json')
    
    SAVE_DIR = './log_masc_graph'
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'best_model_masc.pth')

    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 2
