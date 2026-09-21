# Dataset card — Fashion-MNIST

The flagship CNN project uses Fashion-MNIST: 28×28 grayscale images of 10 apparel categories. TensorFlow exposes the canonical dataset through `tf.keras.datasets.fashion_mnist.load_data()`.

The original training partition is deterministically split into training and validation subsets. The canonical test partition remains untouched until final evaluation.

Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
