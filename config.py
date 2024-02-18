from pathlib import Path


EMBED_DIM = 512

TEXT_MODEL = "distilbert-base-multilingual-cased"
VISION_MODEL = "google/vit-base-patch16-224"

EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
PRECISION = '16-mixed'


# Get the current directory of this script
ROOT_DIR = Path(__file__).parent

# Navigate the directory structure
DATA_DIR = ROOT_DIR / 'data'
LOGS_DIR = ROOT_DIR / 'lightning_logs'
CHECK_DIR = ROOT_DIR / 'checkpoints'
