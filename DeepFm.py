import random
import numpy as np
import pandas as pd
import tensorflow as tf
import config
from time import time
from sklearn.model_selection import train_test_split
from DataReader import dataParser

from tensorflow.python import debug as tf_debug


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
        self.batch_size = cfg.dfm_params["batch_size"]
        self.epoch = cfg.dfm_params["epoch"]
        self._init_graph()

    def _init_weights(self):
        weights = {}

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random.normal(
                [self.feature_size, self.embedding_size],
                0.0,
                0.1,
                name="feature_embeddings",
            )
        )  # N * K
        weights["feature_bias"] = tf.Variable(
            tf.random.normal([self.feature_size, 1], 0.0, 0.1, name="feature_bias")
        )  # N * 1, first order bias, \omega in original paper

        # deep layers
        num_layers = len(self.dfm_params["deep_layers"])
        input_size = self.field_size * self.embedding_size
        # glorot initialization
        glorot = np.sqrt(2.0 / (input_size + self.dfm_params["deep_layers"][0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(
                loc=0.0,
                scale=glorot,
                size=(input_size, self.dfm_params["deep_layers"][0]),
            ),
            dtype=np.float32,
        )
        weights["bias_0"] = tf.Variable(
            np.random.normal(
                loc=0.0, scale=glorot, size=(1, self.dfm_params["deep_layers"][0])
            ),
            dtype=np.float32,
        )
        for i in range(1, num_layers):
            glorot = np.sqrt(
                2.0
                / (
                    self.dfm_params["deep_layers"][i - 1]
                    + self.dfm_params["deep_layers"][i]
                )
            )
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(
                    loc=0.0,
                    scale=glorot,
                    size=(
                        self.dfm_params["deep_layers"][i - 1],
                        self.dfm_params["deep_layers"][i],
                    ),
                ),
                dtype=np.float32,
            )
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(
                    loc=0.0, scale=glorot, size=(1, self.dfm_params["deep_layers"][i])
                ),
                dtype=np.float32,
            )

        # final concat layers
        input_size = (
            self.field_size + self.embedding_size + self.dfm_params["deep_layers"][-1]
        )
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_layer"] = tf.Variable(
            np.random.normal(loc=0.0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32,
        )
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def _init_graph(self):
        """
        [intialize graph]
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feat_index = tf.placeholder(
                tf.int32, (None, self.field_size), name="feat_index"
            )  # None * F
            self.feat_value = tf.placeholder(
                tf.float32, (None, self.field_size), name="feat_value"
            )  # None * F
            self.label = tf.placeholder(tf.float32, (None, 1), name="label")  # None
            self.weights = self._init_weights()

            self.reshaped_feat_value = tf.reshape(
                self.feat_value, [-1, self.field_size, 1]
            )

            # first order items
            self.y_first_order_embedding = tf.nn.embedding_lookup(
                self.weights["feature_bias"], self.feat_index
            )
            self.y_first_order = tf.reduce_sum(
                tf.multiply(self.y_first_order_embedding, self.reshaped_feat_value),
                2,
                name="y_first_order",
            )  # None * F * 1
            # self.y_first_order = tf.nn.dropout()

            # second order items
            self.embeddings = tf.nn.embedding_lookup(
                self.weights["feature_embeddings"], self.feat_index
            )  # None * F * K, latent vector V
            # self.reshaped_feat_value = tf.reshape(
            #     self.feat_value, [-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, self.reshaped_feat_value)
            self.embeddings_sum = tf.reduce_sum(self.embeddings, 1)
            self.embeddings_sum_square = tf.square(self.embeddings_sum)  # None * K
            self.embeddings_square_sum = tf.reduce_sum(
                tf.square(self.embeddings), 1
            )  # None * K
            self.y_second_order = tf.multiply(
                0.5,
                tf.subtract(self.embeddings_sum_square, self.embeddings_square_sum),
                name="y_second_order",
            )

            # self.y_second_order = tf.nn.dropout()

            # deep component
            self.y_deep = tf.reshape(
                self.embeddings, shape=[-1, self.field_size * self.embedding_size]
            )  # None * (F * K)
            for i in range(0, len(self.dfm_params["deep_layers"])):
                self.y_deep = tf.add(
                    tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                    self.weights["bias_%d" % i],
                )
                self.y_deep = self.dfm_params["deep_layer_activation"](self.y_deep)

            # DeepFM
            self.concat_input = tf.concat(
                [self.y_first_order, self.y_second_order, self.y_deep], axis=1
            )
            self.final_output = tf.add(
                tf.matmul(self.concat_input, self.weights["concat_layer"]),
                self.weights["concat_bias"],
            )
            self.final_output = tf.nn.sigmoid(self.final_output)

            # loss and optimizer
            self.loss = tf.losses.log_loss(
                tf.reshape(self.label, (-1, 1)), self.final_output
            )
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.dfm_params["learning_rate"],
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
            ).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            # self.sess = tf_debug.TensorBoardDebugWrapperSession(
            #     self.sess, "127.0.0.1:6000")
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def shuffle_datasets(self, Xi, Xv, y, valid_ratio):
        random_seed = self.dfm_params["random_seed"]
        if valid_ratio == 0.0:
            random.seed(random_seed)
            random.shuffle(Xi)
            random.seed(random_seed)
            random.shuffle(Xv)
            random.seed(random_seed)
            random.shuffle(y)
            Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid = (
                Xi,
                Xv,
                y,
                None,
                None,
                None,
            )
        else:
            Xi_train, Xi_valid, Xv_train, Xv_valid, y_train, y_valid = train_test_split(
                Xi, Xv, y, test_size=valid_ratio, random_state=random_seed
            )
            y_valid = [[y_] for y_ in y_valid]
        return Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid

    # get batch data
    def get_batch(self, Xi, Xv, y, index):
        batch_size = self.batch_size
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y
            #  self.dropout_keep_fm: self.dropout_fm,
            #  self.dropout_keep_deep: self.dropout_deep,
            #  self.train_phase: True
        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        # test_value = self.sess.run(self.loss, feed_dict=feed_dict)
        return loss, opt

    def eval(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi, self.feat_value: Xv, self.label: y}
        loss, final_output = self.sess.run(
            (self.loss, self.final_output), feed_dict=feed_dict
        )
        return loss, final_output

    def fit(
        self,
        Xi,
        Xv,
        y,
        valid_ratio=0.0,
        early_stopping=False,
        refit=False,
        eval_mode=True,
    ):
        """
        Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         is the feature index of feature field j of sample i in the training set
        Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         is the feature value of feature field j of sample i in the training set
                         can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        y_train: label of each sample in the training set
        Xi_valid: list of list of feature indices of each sample in the validation set
        Xv_valid: list of list of feature values of each sample in the validation set
        y_valid: label of each sample in the validation set
        early_stopping: perform early stopping or not
        refit: refit the model on the train+valid dataset or not
        :return: None
        """
        (
            Xi_train,
            Xv_train,
            y_train,
            Xi_valid,
            Xv_valid,
            y_valid,
        ) = self.shuffle_datasets(Xi, Xv, y, valid_ratio)
        if y_valid == None:
            eval_mode = False
        for epoch in range(self.epoch):
            total_batch = len(y_train) // self.batch_size
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(
                    Xi_train, Xv_train, y_train, i
                )
                loss, _ = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # test_value, _ = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            print("Epoch %s, train loss is %.4f" % (str(epoch), loss))

            if eval_mode:
                correct = 0
                val_loss, final_output = self.eval(Xi_valid, Xv_valid, y_valid)
            final_output = [[1] if output >= 0.5 else [0] for output in final_output]
            for i in range(len(y_valid)):
                if final_output[i] == y_valid[i]:
                    correct += 1
            ratio = correct / len(y_valid)
            print("val loss is %.4f, accuracy is %.4f" % (val_loss, ratio))


if __name__ == "__main__":
    Xi, Xv, y = dataParser()
    feature_size, field_size = Xi.shape[0], Xi.shape[1]
    dfm = DeepFM(feature_size, field_size, 8)
    dfm.fit(Xi, Xv, y, valid_ratio=0.3)
