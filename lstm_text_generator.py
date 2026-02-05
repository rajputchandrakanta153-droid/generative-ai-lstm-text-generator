import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # optional

import numpy as np
import tensorflow as tf
import re

# ------------------------------
# Imports (fixed for Pylance)
# ------------------------------
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

# ------------------------------
# 1. Load & preprocess dataset
# ------------------------------
def load_text(path="shakespeare.txt", max_chars=200_000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found!")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    text = re.sub(r'[^a-z0-9\s.,!?;:\'\n]', '', text)
    return text[:max_chars]

text = load_text()
print(f"Total characters used: {len(text)}")

# ------------------------------
# 2. Character-level tokenization
# ------------------------------
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
encoded_text = np.array([char_to_idx[c] for c in text], dtype=np.int32)

# ------------------------------
# 3. Build tf.data dataset
# ------------------------------
SEQ_LENGTH = 40
BATCH_SIZE = 128
STEP = 3

def build_dataset(encoded_text, seq_length=SEQ_LENGTH, step=STEP):
    ds = tf.data.Dataset.from_tensor_slices(encoded_text)
    ds = ds.window(seq_length + 1, shift=step, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_length + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.shuffle(10000)
    return ds

dataset = build_dataset(encoded_text)

total_sequences = len(encoded_text) - SEQ_LENGTH
train_size = int(0.9 * total_sequences)

# ------------------------------
# 3a. Repeat datasets for full epochs
# ------------------------------
train_ds = dataset.take(train_size).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.skip(train_size).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = (total_sequences - train_size) // BATCH_SIZE

# ------------------------------
# 4. Build model
# ------------------------------
model = Sequential([
    Embedding(vocab_size, 64, input_length=SEQ_LENGTH),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

model.summary()

# ------------------------------
# 5. Train model
# ------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ------------------------------
# 6. Text generation
# ------------------------------
def generate_text(seed="to be", length=100, temperature=0.8):
    seed = seed.lower()
    generated = seed

    for _ in range(length):
        input_seq = [char_to_idx.get(c, 0) for c in generated[-SEQ_LENGTH:]]
        input_seq = [0]*(SEQ_LENGTH-len(input_seq)) + input_seq
        preds = model.predict(np.array([input_seq]), verbose=0)[0]
        preds = np.log(preds + 1e-8)/temperature
        probs = np.exp(preds)/np.sum(np.exp(preds))
        next_idx = np.random.choice(vocab_size, p=probs)
        generated += idx_to_char[next_idx]

    return generated

# ------------------------------
# 7. Generate sample text
# ------------------------------
seed = "to be, or not to be"
print("\n--- Generated Text ---\n")
print(generate_text(seed))

