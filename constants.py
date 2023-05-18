import torch 
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
DEVICE = (DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)

CLEANED_DATASET_DIR = "cleaned_datasets"
SHAKESPEARE_PATH="tests/tiny_shakespeare.txt"
