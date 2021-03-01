import config
import numpy as np
import pandas as pd


class FeatDict(object):
    """
    [Geneaation of Feature Dictionary]

    """

    def __init__(self, cfg=config):
        self.train = pd.read_csv(cfg.Train_file)
        self.test = pd.read_csv(cfg.Test_file)
        self.NUMERIC_COLS = cfg.NUMERIC_COLS
        self.IGNORE_COLS = cfg.IGNORE_COLS
        self.label = self.train["target"]
        self._drop_ignore_cols()
        self._gen_feat_dict()
        self._gen_feat_matrix()
        self.field_size = 0

    def _drop_ignore_cols(self):
        for col in self.train.columns:
            if col in self.IGNORE_COLS:
                self.train = self.train.drop(columns=col)
        for col in self.test.columns:
            if col in self.IGNORE_COLS:
                self.test = self.test.drop(columns=col)

    def _gen_feat_dict(self):
        self.feat_dict = {}
        self.total_feat = 0
        self.df = pd.concat([self.train, self.test], axis=0)
        self.total_field = len(self.df.columns)
        for col in self.df.columns:
            if col in self.IGNORE_COLS:
                continue
            if col in self.NUMERIC_COLS:
                self.feat_dict[col] = self.total_feat
                self.total_feat += 1
            else:
                cat_list = list(self.df[col].unique())
                self.feat_dict[col] = dict(
                    zip(
                        cat_list,
                        range(self.total_feat, self.total_feat + len(cat_list)),
                    )
                )
                self.total_feat += len(cat_list)

    def _gen_feat_matrix(self):
        self.train_feat_index = self.train.copy()
        self.train_feat_value = self.train.copy()
        self.test_feat_index = self.test.copy()
        self.test_feat_value = self.test.copy()
        for col in self.df.columns:
            if col in self.IGNORE_COLS or col in self.NUMERIC_COLS:
                self.train_feat_index[col] = self.feat_dict[col]
                self.test_feat_index[col] = self.feat_dict[col]
                continue
            else:
                self.train_feat_index[col] = self.train_feat_index[col].map(
                    self.feat_dict[col]
                )
                self.test_feat_index[col] = self.test_feat_index[col].map(
                    self.feat_dict[col]
                )
                self.train_feat_value[col] = 1
                self.test_feat_value[col] = 1


def dataParser():
    fd = FeatDict()
    Xi, Xv, y, feature_size, field_size = (
        np.array(fd.train_feat_index),
        np.array(fd.train_feat_value),
        np.array(fd.label),
        fd.total_feat,
        fd.total_field
    )
    return Xi, Xv, y, feature_size, field_size


# if __name__ == "__main__":
#     matrix = dataParser()
#     print(matrix.head())
