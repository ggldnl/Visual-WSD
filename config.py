
PROJECTION_LAYERS = 3
EMBED_DIM = 512
TRANSFORMER_EMBED_DIM = 768
MAX_LEN = 64  # Maximum length of text

TEXT_MODEL = "distilbert-base-multilingual-cased"  # 541 MB
VISION_MODEL = "google/vit-base-patch16-224"

EPOCHS = 5
BATCH_SIZE = 96
LEARNING_RATE = 1e-4
TEMPERATURE = 0.5
VISION_FREEZE_LAYERS = 2
CAPTION_FREEZE_LAYERS = 6
UNFREEZE_LAYERS = True
LOSS_FN = "asymmetric_loss"

DATA_PATH = "/kaggle/input/coco-2017-dataset/coco2017/"
ANNOTATION_PATH = DATA_PATH + "annotations/captions_train2017.json"
LOG_PATH = "/kaggle/working/logs/"