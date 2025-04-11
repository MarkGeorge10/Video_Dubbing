import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import logging
import os
from datetime import datetime
import pickle  # Added import for pickle
import time

# Set up argument parsing
parser = argparse.ArgumentParser(description="English to Turkish Translation Model")
parser.add_argument('--use-gpu', action='store_true', help="Use GPU if available")
parser.add_argument('--num-workers', type=int, default=1, help="Number of workers for data loading")
args = parser.parse_args()

# Configure logging
output_dir = '/storage/work/mqf5675/Masters/NLP/project/model_output/seq2seq'
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
os.makedirs(f'{output_dir}/logs', exist_ok=True)  # Directory for logs

log_file = f'{output_dir}/logs/translation_en_tur_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Save logs to a file
        logging.StreamHandler()         # Also print to console
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting English to Turkish translation script.")

# Set up GPU
if args.use_gpu and tf.config.list_physical_devices('GPU'):
    logger.info("GPU detected and will be used.")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logger.warning("No GPU detected or --use-gpu not specified; using CPU.")

# Step 1: Load and Preprocess the Dataset
logger.info("Loading dataset...")
try:
    data = pd.read_csv('/storage/work/mqf5675/Masters/NLP/project/data/TR2EN.txt', sep='\t', names=['english', 'turkish'])
    data = data[:350000]

    logger.info(f"Dataset loaded successfully with {len(data)} samples.")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    raise

# Step 2: Prepare input (English) and target (Turkish) texts
input_texts = data['english'].tolist()
target_texts = ['<start> ' + text + ' <end>' for text in data['turkish']]
logger.info("Input and target texts prepared with <start> and <end> tokens.")

# Set hyperparameters
latent_dim = 256
epochs = 10
batch_size = 64
logger.info(f"Hyperparameters set: latent_dim={latent_dim}, epochs={epochs}, batch_size={batch_size}")

# Step 3: Tokenize the input (English) texts
logger.info("Tokenizing input (English) texts...")
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_maxlen = max(len(seq) for seq in input_sequences)
input_vocab_size = len(input_tokenizer.word_index) + 1
logger.info(f"Input tokenizer created: vocab_size={input_vocab_size}, maxlen={input_maxlen}")

# Step 4: Tokenize the target (Turkish) texts
logger.info("Tokenizing target (Turkish) texts...")
target_tokenizer = Tokenizer(filters='', split=' ', lower=False)
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_maxlen = max(len(seq) for seq in target_sequences)
target_vocab_size = len(target_tokenizer.word_index) + 1
logger.info(f"Target tokenizer created: vocab_size={target_vocab_size}, maxlen={target_maxlen}")

# Debug tokenizer
try:
    start_idx = target_tokenizer.word_index['<start>']
    end_idx = target_tokenizer.word_index['<end>']
    logger.info(f"Target tokenizer verified: <start> index={start_idx}, <end> index={end_idx}")
except KeyError as e:
    logger.error(f"Token not found in target tokenizer: {str(e)}")
    raise

# Save tokenizers
os.makedirs(f'{output_dir}/tokenizer', exist_ok=True)
with open(f'{output_dir}/tokenizer/input_tokenizer.pkl', 'wb') as f:
    pickle.dump(input_tokenizer, f)
with open(f'{output_dir}/tokenizer/target_tokenizer.pkl', 'wb') as f:
    pickle.dump(target_tokenizer, f)
logger.info("Tokenizers saved successfully.")

# Save input_maxlen and target_maxlen
maxlen_dict = {'input_maxlen': input_maxlen, 'target_maxlen': target_maxlen}
with open(f'{output_dir}/tokenizer/maxlen_values.pkl', 'wb') as f:
    pickle.dump(maxlen_dict, f)
logger.info(f"Maxlen values saved: input_maxlen={input_maxlen}, target_maxlen={target_maxlen}")

# Step 5: Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=input_maxlen, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=target_maxlen, padding='post')
logger.info("Sequences padded.")

# Step 6: Split data into training and validation sets
train_input_seq, val_input_seq, train_target_seq, val_target_seq = train_test_split(
    input_sequences, target_sequences, test_size=0.2, random_state=42
)
train_input_texts, val_input_texts = train_test_split(input_texts, test_size=0.2, random_state=42)
train_target_texts, val_target_texts = train_test_split(target_texts, test_size=0.2, random_state=42)
logger.info(f"Data split: {len(train_input_seq)} training samples, {len(val_input_seq)} validation samples.")

# Step 7: Prepare decoder input/output
train_target_input_seq = train_target_seq[:, :-1]
train_target_output_seq = train_target_seq[:, 1:]
val_target_input_seq = val_target_seq[:, :-1]
val_target_output_seq = val_target_seq[:, 1:]
logger.info("Decoder input/output sequences prepared.")

# Step 8: Define the Encoder
encoder_inputs = Input(shape=(input_maxlen,))
encoder_embedding = Embedding(input_vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Step 9: Define the Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Step 10: Define and Compile the Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['sparse_categorical_accuracy'])
logger.info("Model compiled with sparse_categorical_crossentropy loss and accuracy metric.")

# Step 11: Define Callbacks
modelcheckpoint = ModelCheckpoint(filepath=output_dir + "/weights.{epoch:02d}.hdf5", save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"Starting Epoch {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_duration = time.time() - self.epoch_start_time
        logger.info(f"Epoch {epoch + 1}/{self.params['epochs']} completed: "
                    f"loss={logs.get('loss'):.4f}, "
                    f"accuracy={logs.get('sparse_categorical_accuracy'):.4f}, "
                    f"val_loss={logs.get('val_loss'):.4f}, "
                    f"val_accuracy={logs.get('val_sparse_categorical_accuracy'):.4f}, "
                    f"duration={epoch_duration:.2f} seconds")
# Step 12: Train the Model
logger.info("Starting model training...")
history = model.fit(
    [train_input_seq, train_target_input_seq], train_target_output_seq,
    batch_size=batch_size, epochs=epochs,
    validation_data=([val_input_seq, val_target_input_seq], val_target_output_seq),
    callbacks=[modelcheckpoint, early_stopping, LoggingCallback()],
    verbose=0  # Set to 0 to avoid duplicate logging with custom callback
)
logger.info("Training completed.")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
os.makedirs(f'{output_dir}/plots', exist_ok=True)
plt.savefig(f'{output_dir}/plots/training_history.png')
plt.close()
logger.info("Training history plot saved.")

# Step 13: Define Inference Models
encoder_model = Model(encoder_inputs, encoder_states, name='encoder_model')
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name='decoder_model')

# Step 14: Save the Models
try:
    model.save(output_dir + '/training_model.h5')
    encoder_model.save(output_dir + '/encoder_model.h5')
    decoder_model.save(output_dir + '/decoder_model.h5')
    logger.info("Models saved successfully.")
except Exception as e:
    logger.error(f"Failed to save models: {str(e)}")
    raise

logger.info("Script execution completed.")