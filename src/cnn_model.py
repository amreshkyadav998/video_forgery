"""
cnn_model.py
MobileNetV2 with data augmentation and two-phase fine-tuning.

Phase 1  - Head only (backbone frozen), high LR, 15 epochs
Phase 2  - Unfreeze top-20 backbone layers, low LR, up to 30 epochs
           with EarlyStopping and ReduceLROnPlateau

After training the classifier, build_feature_extractor() creates a
model that outputs the 128-d "features" layer for downstream SVM / LSTM.
"""

import logging

logger = logging.getLogger(__name__)


class CombinedHistory:
    """Merge Keras History objects from two training phases."""

    def __init__(self, h1, h2):
        self.history = {}
        for key in h1.history:
            self.history[key] = h1.history[key] + h2.history.get(key, [])


def build_cnn_classifier(input_shape=(128, 128, 3)):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    tf.get_logger().setLevel("ERROR")

    base = MobileNetV2(
        input_shape=input_shape, include_top=False,
        weights="imagenet", pooling="avg",
    )
    base.trainable = False

    inp = layers.Input(shape=input_shape, name="cnn_input")

    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomContrast(0.10)(x)

    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = layers.Dense(128, activation="relu", name="features")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("CNN classifier built (MobileNetV2 backbone, frozen).")
    model.summary(print_fn=logger.info)
    return model, base


def train_cnn(model, base_model,
              X_train, y_train, X_val, y_val,
              phase1_epochs=15, phase2_epochs=30, batch_size=32):
    """Two-phase training: head-only then fine-tune top-20 backbone layers."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    logger.info(f"Phase 1: Training head for {phase1_epochs} epochs "
                f"(backbone frozen) ...")
    h1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=phase1_epochs,
        batch_size=batch_size,
        verbose=1,
    )
    best_p1 = max(h1.history["val_accuracy"])
    logger.info(f"Phase 1 complete -- best val_accuracy: {best_p1:.4f}")

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1,
        ),
    ]

    logger.info(f"Phase 2: Fine-tuning top 20 backbone layers "
                f"for up to {phase2_epochs} epochs ...")
    h2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=phase2_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    best_p2 = max(h2.history["val_accuracy"])
    logger.info(f"Phase 2 complete -- best val_accuracy: {best_p2:.4f}")

    return CombinedHistory(h1, h2)


def build_feature_extractor(trained_model):
    """
    Create a model that outputs the 128-d 'features' layer
    from a trained CNN_Classifier.  Augmentation layers become
    no-ops during model.predict().
    """
    from tensorflow.keras import Model

    feat_layer = trained_model.get_layer("features")
    extractor = Model(
        inputs=trained_model.input,
        outputs=feat_layer.output,
        name="Feature_Extractor",
    )
    logger.info(f"Feature extractor built -- output dim: "
                f"{extractor.output_shape[-1]}")
    return extractor


def extract_features(model, images, batch_size=32):
    logger.info(f"Extracting CNN features from {len(images)} images ...")
    features = model.predict(images, batch_size=batch_size, verbose=0)
    logger.info(f"Feature matrix shape: {features.shape}")
    return features
