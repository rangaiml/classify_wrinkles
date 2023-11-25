import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class WrinkleDetectionLayer(Layer):
    def __init__(self, alpha=1, beta=1, gamma=1, **kwargs):
        super(WrinkleDetectionLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def build(self, input_shape):
        # Define the learnable parameters (weights) for the wrinkle detection kernel
        self.alpha_weight = self.add_weight("alpha_weight", shape=(1,), initializer="ones", trainable=True)
        self.beta_weight = self.add_weight("beta_weight", shape=(1,), initializer="ones", trainable=True)
        self.gamma_weight = self.add_weight("gamma_weight", shape=(1,), initializer="ones", trainable=True)

    def call(self, inputs):
        region1, region2 = inputs

        # Explicitly cast tensors to float32
        region1 = tf.cast(region1, dtype=tf.float32)
        region2 = tf.cast(region2, dtype=tf.float32)

        color_difference = tf.reduce_sum(tf.abs(region1 - region2))
        texture_diff = tf.reduce_sum(tf.square(region1 - region2))
        # Explicitly cast tensors to int32
        spatial_diff = tf.reduce_sum(tf.abs(tf.range(tf.shape(region1)[1], dtype=tf.float32) - tf.range(tf.shape(region2)[1], dtype=tf.float32)) * tf.abs(region1 - region2))

        wrinkle_score = (
            self.alpha_weight * color_difference +
            self.beta_weight * texture_diff +
            self.gamma_weight * spatial_diff
        )

        return wrinkle_score


def main():
    # Load wrinkled and pressed images
    wrinkled_image_path = "/Users/ranga/PycharmProjects/wrinkle_detective/wrinkled.png"
    pressed_image_path = "/Users/ranga/PycharmProjects/wrinkle_detective/pressed.png"

    wrinkled_image = tf.keras.preprocessing.image.load_img(wrinkled_image_path, target_size=(224, 224))
    pressed_image = tf.keras.preprocessing.image.load_img(pressed_image_path, target_size=(224, 224))

    wrinkled_image_array = tf.keras.preprocessing.image.img_to_array(wrinkled_image)
    pressed_image_array = tf.keras.preprocessing.image.img_to_array(pressed_image)

    wrinkled_image_array = np.expand_dims(wrinkled_image_array, axis=0)
    pressed_image_array = np.expand_dims(pressed_image_array, axis=0)

    # Create patches for testing
    patch_size = 64
    wrinkled_patch = wrinkled_image_array[:, :patch_size, :patch_size, :]
    pressed_patch = pressed_image_array[:, :patch_size, :patch_size, :]

    # Create the wrinkle detection model
    input_wrinkled = tf.keras.layers.Input(shape=(patch_size, patch_size, 3), name='wrinkled_input')
    input_pressed = tf.keras.layers.Input(shape=(patch_size, patch_size, 3), name='pressed_input')

    wrinkled_score = WrinkleDetectionLayer(name='wrinkled_score')(inputs=[input_wrinkled, input_pressed])
    pressed_score = WrinkleDetectionLayer(name='pressed_score')(inputs=[input_pressed, input_wrinkled])

    # Model output (you can define a custom loss based on the wrinkle scores)
    model = tf.keras.models.Model(inputs=[input_wrinkled, input_pressed], outputs=[wrinkled_score, pressed_score])

    # Compile and train your model as needed
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Test the model with the example patches
    wrinkled_score, pressed_score = model.predict([wrinkled_patch, pressed_patch])
    print("Wrinkled Detection Score:", wrinkled_score)
    print("Pressed Detection Score:", pressed_score)


if __name__ == "__main__":
    main()
