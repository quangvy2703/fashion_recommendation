ARTICLES_PATH = "datasets/articles.csv"
TRANSACTIONS_PATH = "datasets/transactions_train.csv"
CUSTOMERS_PATH = "datasets/customers.csv"
DATA_DIR = "datasets"

MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 10
PERIODS = 7
AGE_FILL_VALUE = 36
BATCH_SIZE = 16
EPOCHS = 100

# MODEL CONFIGS
TRANSACTION_ENCODER_INPUT_SIZE = 632
CUSTOMER_ENCODER_INPUT_SIZE = 11
HIDDEN_SIZE = 1024
N_CLASSES = 2

LR = 0.01
ARTICLE_FEATURES = ['article_id', 'product_type_no', 'graphical_appearance_no',
                    'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                    'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']