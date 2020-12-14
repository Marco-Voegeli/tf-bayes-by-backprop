import tensorflow as tf
import tensorflow_probability as tfp


def negative_log_likelihood(y_true: tf.Tensor, y_pred: tf.Tensor, sigma=1.0) -> tf.Tensor:
    dist = tfp.distributions.Normal(loc=y_true, scale=sigma)
    return tf.reduce_sum(-dist.log_prob(y_pred))
