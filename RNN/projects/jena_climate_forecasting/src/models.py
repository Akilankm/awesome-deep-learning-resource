import tensorflow as tf

def build_model(kind: str, input_shape):
    layer_map = {
        "SimpleRNN": tf.keras.layers.SimpleRNN,
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU,
    }
    if kind not in layer_map:
        raise ValueError(f"Unknown model kind: {kind}")
    inputs = tf.keras.Input(shape=input_shape)
    x = layer_map[kind](16)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name=kind.lower())
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
