import math
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from distributions import ReparameterizedGaussian, GaussScaleMixturePrior


class VariationalDense(Layer):
    def __init__(self, units, pi, sigma_1, sigma_2, activation=None):
        super().__init__()
        self.units = units
        self.pi = pi
        self.sigma_1, self.sigma_2 = sigma_1, sigma_2
        self.activation = activations.get(activation)

        # placeholders for variables created in build function
        self.W_mu, self.W_rho, self.W_prior, self.W_var_posterior = [None] * 4
        self.b_mu, self.b_rho, self.b_prior, self.b_var_posterior = [None] * 4

        self.W_log_prior, self.b_log_prior = 0, 0
        self.W_log_variational_posterior = 0
        self.b_log_variational_posterior = 0

    def build(self, input_shape: tf.TensorShape) -> None:
        # weight variables
        self.W_mu = self.add_weight(
            shape=(input_shape[1], self.units),
            initializer=tf.random_normal_initializer(),
            trainable=True)
        self.W_rho = self.add_weight(
            shape=(input_shape[1], self.units),
            initializer=tf.random_normal_initializer(),
            trainable=True)

        self.W_prior = GaussScaleMixturePrior(self.pi, self.sigma_1, self.sigma_2)

        # bias variables
        self.b_mu = self.add_weight(
            shape=(self.units,),
            initializer=tf.zeros_initializer(),
            trainable=True)
        self.b_rho = self.add_weight(
            shape=(self.units,),
            initializer=tf.zeros_initializer(),
            trainable=True)

        self.b_prior = GaussScaleMixturePrior(self.pi, self.sigma_1, self.sigma_2)

    @property
    def log_prior(self):
        return self.W_log_prior + self.b_log_prior

    @property
    def log_variational_posterior(self):
        return self.W_log_variational_posterior + self.b_log_variational_posterior

    def call(self, inputs: tf.Tensor, training=None, **kwargs) -> tf.Tensor:
        # sample from the variational posterior distribution
        self.W_var_posterior = ReparameterizedGaussian(self.W_mu, self.W_rho)
        self.b_var_posterior = ReparameterizedGaussian(self.b_mu, self.b_rho)

        w = self.W_var_posterior.sample()
        b = self.b_var_posterior.sample()

        if training:
            self.W_log_prior = self.W_prior.log_prob(w)
            self.b_log_prior = self.b_prior.log_prob(b)
            self.W_log_variational_posterior = self.W_var_posterior.log_prob(w)
            self.b_log_variational_posterior = self.b_var_posterior.log_prob(b)
        else:
            self.W_log_prior, self.b_log_prior = 0, 0
            self.W_log_variational_posterior, self.b_log_variational_posterior = 0, 0

        return self.activation(inputs @ w + b)
