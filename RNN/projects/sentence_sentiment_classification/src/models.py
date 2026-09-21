from __future__ import annotations
import tensorflow as tf

def make_vectorizer(vocabulary, sequence_length: int):
    layer=tf.keras.layers.TextVectorization(
        max_tokens=len(vocabulary),
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
        name="text_vectorization",
    )
    layer.set_vocabulary(vocabulary)
    return layer

def build_sentence_model(kind: str, vocabulary, sequence_length: int=50,
                         embedding_dim: int=32, recurrent_units: int=32):
    recurrent={
        "SimpleRNN":tf.keras.layers.SimpleRNN,
        "LSTM":tf.keras.layers.LSTM,
        "GRU":tf.keras.layers.GRU,
    }
    if kind not in recurrent:
        raise ValueError(f"Unsupported recurrent layer: {kind}")

    vectorizer=make_vectorizer(vocabulary,sequence_length)
    inputs=tf.keras.Input(shape=(),dtype=tf.string,name="sentence")
    x=vectorizer(inputs)
    x=tf.keras.layers.Embedding(
        input_dim=len(vocabulary),
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(x)
    x=recurrent[kind](recurrent_units,name=kind.lower())(x)
    x=tf.keras.layers.Dropout(.20,name="dropout")(x)
    outputs=tf.keras.layers.Dense(1,activation="sigmoid",name="positive_probability")(x)
    model=tf.keras.Model(inputs,outputs,name=f"{kind.lower()}_sentence_classifier")
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return model
