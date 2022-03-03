cfg = {}
cfg["ARTICLES_PATH"] = "datasets/articles.csv"
cfg["TRANSACTIONS_PATH"] = "datasets/transactions_train.csv"
cfg["CUSTOMERS_PATH"] = "datasets/customers.csv"
cfg['DATA_DIR'] = "datasets"

cfg["MIN_SEQUENCE_LENGTH"] = 2
cfg["MAX_SEQUENCE_LENGTH"] = 50
cfg["PERIODS"] = 7
cfg["AGE_FILL_VALUE"] = 36
cfg["BATCH_SIZE"] = 16
cfg["EPOCHS"] = 100

# MODEL CONFIGS
cfg["TRANSACTION_ENCODER_INPUT_SIZE"] = 632
cfg["CUSTOMER_ENCODER_INPUT_SIZE"] = 11
cfg["HIDDEN_SIZE"] = 1024
cfg["N_CLASSES"] = 2
cfg["N_LAYERS"] = 3
cfg["N_HEADS"] = 8
cfg["LR"] = 0.01
cfg['MODELS_PATH'] = "saved_models"
cfg['CONTINUE_TRAINING'] = False
MAX_SEQUENCE_LENGTH = 10
ARTICLE_FEATURES = ['article_id', 'product_type_no', 'graphical_appearance_no',
                    'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                    'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
