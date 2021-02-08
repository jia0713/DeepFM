import numpy as np
import pandas as pd
import tensorflow as tf
import config
from DataReader import dataParser


class DeepFM(object):
    """

    [DeepFM model, original paper: 
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He]

    Args:
    feature_size ([int]): [total feature sizes:N]
    filed_size ([int]): [total field sizes:F]
    embedding_size ([int]): [feature embedding sizes:K]
    cfg: [other configurations]

    """

    def __init__(self, feature_size, field_size, embedding_size, cfg=config):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.dfm_params = cfg.dfm_params
        self._init_graph()

    def _init_weights(self):
        weights = {}

        # embeddings
        weights["feature_embeddings"] = tf.Variable(tf.random.normal([self.feature_size, self.embedding_size], 0.0, 0.1,
                                                                     name="feature_embeddings"))    # N * K
        weights["feature_bias"] = tf.Variable(tf.random.normal([self.feature_size, 1], 0.0, 0.1,
                                                               name="feature_bias"))       # N * 1, first order bias, \omega in original paper

        # deep layers
        num_layers = len(self.dfm_params["deep_layers"])
        input_size = self.field_size * self.embedding_size
        # glorot initialization
        glorot = np.sqrt(
            2.0 / (input_size + self.dfm_params['deep_layers'][0]))
        weights["layer_0"] = tf.Variable(np.random.normal(loc=0.0, scale=glorot,
                                                          size=(input_size, self.dfm_params["deep_layers"][0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0.0, scale=glorot,
                                                         size=(1, self.dfm_params["deep_layers"][0])), dtype=np.float32)
        for i in range(1, num_layers):
            glorot = np.sqrt(
                2.0 / (self.dfm_params["deep_layers"][i-1] + self.dfm_params['deep_layers'][i]))
            weights["layer_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(
                self.dfm_params["deep_layers"][i-1], self.dfm_params["deep_layers"][i])), dtype=np.float32)
            weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot,
                                                                  size=(1, self.dfm_params["deep_layers"][i])), dtype=np.float32)

        # final concat layers
        input_size = self.field_size + self.embedding_size + \
            self.dfm_params["deep_layers"][-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_layer"] = tf.Variable(np.random.normal(
            loc=0.0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
        weights["concat_bias"] = tf.Variable(
            tf.constant(0.01), dtype=np.float32)

        return weights

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feat_index = tf.placeholder(
                tf.int32, (None, self.field_size), name="feat_index")    # None * F
            self.feat_value = tf.placeholder(
                tf.float32, (None, self.field_size), name="feat_value")    # None * F
            self.label = tf.placeholder(
                tf.float32, (None, 1), name="label")    # None
            self.weights = self._init_weights()

            # first order items
            self.y_first_order_embedding = tf.nn.embedding_lookup(
                self.weights["feature_bias"], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(
                self.y_first_order_embedding, self.feat_value), 2)    # None * F * 1
            # self.y_first_order = tf.nn.dropout()

            # second order items
            self.embeddings = tf.nn.embedding_lookup(
                self.weights["feature_embeddings"], self.feat_index)    # None * F * K, latent vector V
            self.reshaped_feat_value = tf.reshape(
                self.feat_value, [-1, self.field_size, 1])
            self.embeddings = tf.multiply(
                self.embeddings, self.reshaped_feat_value)
            self.embeddings_sum = tf.reduce_sum(self.embeddings, 1)
            self.embeddings_sum_square = tf.square(
                self.embeddings_sum)  # None * K
            self.embeddings_square_sum = tf.reduce_sum(
                tf.square(self.embeddings), 1)    # None * K
            self.y_second_order = 0.5 * \
                tf.subtract(self.embeddings_sum_square,
                            self.embeddings_square_sum)
            # self.y_second_order = tf.nn.dropout()

            # deep component
            self.y_deep = tf.reshape(
                self.embeddings, shape=[-1, self.field_size * self.embedding_size])    # None * (F * K)
            for i in range(0, len(self.dfm_params["deep_layers"])):
                self.y_deep = tf.add(tf.matmul(
                    self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.dfm_params["deep_layer_activation"](
                    self.y_deep)

            # DeepFM
            self.concat_input = tf.concat(
                [self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            self.final_output = tf.add(tf.matmul(
                self.concat_input, self.weights["concat_layer"]), self.weights["concat_bias"])
            self.final_output = tf.nn.sigmoid(self.final_output)

            # loss and optimizer
            self.loss = tf.losses.log_loss(tf.reshape(
                self.label, (-1, 1)), self.final_output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.dfm_params['learning_rate'], beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

# if __name__ == "__main__":
#     Xi, Xv = dataParser()
#     print(Xi.shape)
#     dfm = DeepFM(feature_size=100, field_size=37, embedding_size=8)
