import tensorflow as tf

def build_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="mlp_baseline")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
