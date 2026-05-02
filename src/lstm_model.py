"""
lstm_model.py
LSTM classifier with BatchNorm + Dense projection to handle raw
high-dimensional (1280-d) pretrained features without overfitting.
"""

import logging

logger = logging.getLogger(__name__)


def build_lstm_model(feature_size):
    """
    Architecture
    ------------
    Input            : (1, feature_size)          e.g. (1, 1298)
    BatchNorm        : normalises each feature
    Dense(128, relu) : learned projection to lower-d space
    LSTM(64)         : sequence classifier (single timestep)
    Dropout(0.3)
    Dense(32, relu)
    Dropout(0.3)
    Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    tf.get_logger().setLevel("ERROR")

    inp = layers.Input(shape=(1, feature_size), name="lstm_input")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x, name="LSTM_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


def train_lstm(model, X_train, y_train, X_val, y_val,
               epochs=40, batch_size=32):
    """
    Train the LSTM. Features are reshaped to (N, 1, D).
    """
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X_train_seq = np.expand_dims(X_train, axis=1)
    X_val_seq = np.expand_dims(X_val, axis=1)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    logger.info(f"Training LSTM for up to {epochs} epochs ...")
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    final_val = max(history.history["val_accuracy"])
    logger.info(f"LSTM training complete — best val_accuracy: {final_val:.4f}")
    return history
