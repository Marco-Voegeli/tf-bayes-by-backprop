import tensorflow as tf
from tensorflow.keras.models import Model

from layers import VariationalDense


class BayesianNetwork(Model):
    """
    Exemplary implementation of a Bayesian neural network with
    three VariationalDense layers with 64, 32 and 1 units.
    """
    def __init__(self, num_batches, pi, sigma_1, sigma_2):
        """Creates a new instance of the specified network architecture

        :param num_batches: total number of batches contained
                            in the dataset used for averaging over data
                            independent loss term
        """
        super(BayesianNetwork, self).__init__()
        self.num_batches = num_batches
        self.dense_1 = VariationalDense(32, pi, sigma_1, sigma_2, activation='relu')
        self.dense_2 = VariationalDense(32, pi, sigma_1, sigma_2, activation='relu')
        self.dense_3 = VariationalDense(1, pi, sigma_1, sigma_2)

    @property
    def log_prior_loss(self) -> tf.Tensor:
        """Total prior log loss log p(w) summed up
        over all initialized VariationalDense layers

        :return: total log prior log loss (log p(w))
        """
        return (self.dense_1.log_prior +
                self.dense_2.log_prior +
                self.dense_3.log_prior)

    @property
    def log_variational_posterior_loss(self) -> tf.Tensor:
        """Total variational posterior loss q(w|θ) summed
        up over all initialized VariationalDense layers

        :return: total variational posterior log loss (log q(w|θ))
        """
        return (self.dense_1.log_variational_posterior +
                self.dense_2.log_variational_posterior +
                self.dense_3.log_variational_posterior)

    @property
    def kullback_leibler_loss(self) -> tf.Tensor:
        """Total approximated Kullback-Leibler loss calculated
        from the total log prior loss p(w) and total variational
        posterior loss q(w|θ) normalized with the total number of
        batches of the dataset

        :return: approximate Kullback-Leibler loss
        """
        return 1 / self.num_batches * \
               (self.log_variational_posterior_loss -
                self.log_prior_loss)

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        x = self.dense_1(inputs, training)
        x = self.dense_2(x, training)
        x = self.dense_3(x, training)
        if training: self.add_loss(self.kullback_leibler_loss)
        return x

    def get_config(self):
        return {'pi': self.pi, 'sigma_1': self.sigma_1, 'sigma_2': self.sigma2}
