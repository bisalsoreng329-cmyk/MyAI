import torch
import os

PROJECT_NAME = "MyAI"
VERSION = "1.0.0"

VOCAB_SIZE = 50000
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FFN_DIM = 2048
MAX_SEQ_LEN = 512
DROPOUT = 0.1

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

VOCAB_FILE = os.path.join(BASE_DIR, "vocab.json")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "model.pt")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
