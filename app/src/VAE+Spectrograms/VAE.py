import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

tf.compat.v1.disable_eager_execution()

# uses Relu (Prelu/LeakyRelu better?)
class VAE:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape 
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    # encoder model
    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        encoder_layers = encoder_input
        for layer_index in range(self._num_conv_layers):
            layer_number = layer_index + 1
            conv_layer = Conv2D(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"encoder_conv_layer_{layer_number}"
            )
            encoder_layers = conv_layer(encoder_layers)
            encoder_layers = ReLU(name=f"encoder_relu_{layer_number}")(encoder_layers)
            encoder_layers = BatchNormalization(name=f"encoder_bn_{layer_number}")(encoder_layers)
        
        self._shape_before_bottleneck = backend.int_shape(encoder_layers)[1:]
        encoder_layers = Flatten()(encoder_layers)
        self.mu = Dense(self.latent_space_dim, name="mu")(encoder_layers)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(encoder_layers)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = backend.random_normal(shape=backend.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + backend.exp(log_variance / 2) * epsilon
            return sampled_point

        encoder_output = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, encoder_output, name="encoder")

    # decoder model
    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        decoder_layers = Dense(np.prod(self._shape_before_bottleneck), name="decoder_dense")(decoder_input)
        decoder_layers = Reshape(self._shape_before_bottleneck)(decoder_layers)
        for layer_index in reversed(range(1, self._num_conv_layers)):
            layer_num = self._num_conv_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            decoder_layers = conv_transpose_layer(decoder_layers)
            decoder_layers = ReLU(name=f"decoder_relu_{layer_num}")(decoder_layers)
            decoder_layers = BatchNormalization(name=f"decoder_bn_{layer_num}")(decoder_layers)

        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        decoder_layers = conv_transpose_layer(decoder_layers)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(decoder_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    # VAE model
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    # building VAE
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    # compile VAE
    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer = Adam(learning_rate=learning_rate), loss=self._calculate_combined_loss,
                            metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss])

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = (self.reconstruction_loss_weight * reconstruction_loss) + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = backend.mean(backend.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * backend.sum(1 + self.log_variance - backend.square(self.mu) - backend.exp(self.log_variance), axis=1)
        return kl_loss
    
    # train VAE
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    # get reconstructions
    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    # Save the VAE to the current directory
    def save(self):
        parameters = [self.input_shape, self.conv_filters, self.conv_kernels, self.conv_strides, self.latent_space_dim]
        with open(os.path.join(".", "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)

        self.model.save_weights(os.path.join(".", "weights.h5"))

    # load VAE from save (/.)
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder

#train
LEARNING_RATE = 0
BATCH_SIZE = 0
EPOCHS = 0
SPECTROGRAMS_PATH = ""

if __name__ == "__main__":
    x_train = []
    for root, _, file_names in os.walk(SPECTROGRAMS_PATH): ###########################
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) 
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]

    autoencoder = VAE(
        # input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)

    autoencoder.save("VAE_model")

# test layers
# if __name__ == "__main__":
#     autoencoder = VAE(
#         input_shape=(256, 64, 1),
#         conv_filters=(512, 256, 128, 64, 32),
#         conv_kernels=(3, 3, 3, 3, 3),
#         conv_strides=(2, 2, 2, 2, (2, 1)),
#         latent_space_dim=128
#     )
#     autoencoder.encoder.summary()
#     autoencoder.decoder.summary()
#     autoencoder.model.summary()