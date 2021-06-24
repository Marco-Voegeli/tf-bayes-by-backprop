import math
import tensorflow as tf
from abc import abstractmethod
from overrides import overrides
import tensorflow_probability as tfp


class DistributionBase:
    """Basic interface that should be implemented for all
    different kinds of distributions
    """
    def __init__(self):
        self.log_2_pi = math.log(math.sqrt(2 * math.pi))

    @abstractmethod
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def prob(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def sample(self) -> tf.Tensor:
        pass


class Gaussian(DistributionBase):
    """Default Gaussian (normal) distribution with centered
    around mu and with standard deviation of sigma
    """
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu, self.sigma = mu, sigma

    @overrides
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        raw_log_probabilities = tfp.distributions \
            .Normal(self.mu, self.sigma) \
            .log_prob(x)
        return tf.reduce_sum(raw_log_probabilities)

    @overrides
    def prob(self, x: tf.Tensor) -> tf.Tensor:
        raw_prob = tfp.distributions \
            .Normal(self.mu, self.sigma) \
            .prob(x)
        return tf.reduce_sum(raw_prob)

    @overrides
    def sample(self) -> tf.Tensor:
        return tf.random.normal(self.mu.shape, self.mu, self.sigma)


class ReparameterizedGaussian(DistributionBase):
    """Reparameterized Gaussian distribution as described
    in Blundell et al. (Ch: 3.2 Gaussian variational posterior)
    """
    def __init__(self, mu: tf.Tensor, rho: tf.Tensor):
        super().__init__()
        self.mu, self.rho = mu, rho

    @overrides
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        raw_log_probabilities = tfp.distributions \
            .Normal(self.mu, self.sigma(self.rho)) \
            .log_prob(x)
        return tf.reduce_sum(raw_log_probabilities)

    @overrides
    def prob(self, x: tf.Tensor) -> tf.Tensor:
        raw_probabilities = tfp.distributions \
            .Normal(self.mu, self.sigma(self.rho)) \
            .log_prob(x)
        return tf.reduce_sum(raw_probabilities)

    @overrides
    def sample(self) -> tf.Tensor:
        e = tf.random.normal(self.mu.shape, 0, 1)
        return self.mu + self.sigma(self.rho) * e

    @staticmethod
    def sigma(rho: tf.Tensor) -> tf.Tensor:
        return tf.math.softplus(rho)


class GaussScaleMixturePrior(DistributionBase):
    """Gaussian scale mixture prior distribution
    (spike-and-slab) as described in Blundell et al.
    (Ch: 3.3 Scale mixture prior)
    """
    def __init__(self, pi, sigma_1, sigma_2):
        super().__init__()
        self.pi = pi
        self.sigma_1, self.sigma_2 = sigma_1, sigma_2
        self.gaussian1 = Gaussian(0, sigma_1)
        self.gaussian2 = Gaussian(0, sigma_2)

    @overrides
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        prob1 = self.gaussian1.prob(x)
        prob2 = self.gaussian2.prob(x)
        mixture = (self.pi * prob1) + ((1 - self.pi) * prob2)
        return tf.reduce_sum(tf.math.log(mixture))

    @overrides
    def prob(self, x: tf.Tensor) -> tf.Tensor:
        prob1 = self.gaussian1.prob(x)
        prob2 = self.gaussian2.prob(x)
        mixture = (self.pi * prob1) + ((1 - self.pi) * prob2)
        return tf.reduce_sum(mixture)

    @overrides
    def sample(self) -> tf.Tensor:
        raise NotImplementedError()
