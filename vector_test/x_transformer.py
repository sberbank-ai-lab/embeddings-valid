import datetime

import pandas as pd

from vector_test import cls_loader
from vector_test.file_reader import FeatureFile
from vector_test.preprocessing.category_encoder import CategoryEncoder


class XTransformer:
    def __init__(self, conf, feature_name, preprocessing):
        self.conf = conf
        self.feature_name = feature_name

        self.preprocessing = None
        self.load_preprocessors(preprocessing)

        self.feature_list = None
        self.load_time = None
        self.load_features()

    def get_df_features(self, df_target):
        features = [df.df.set_index(df.cols_id) for df in self.feature_list]
        index = df_target.df.set_index(df_target.cols_id).index
        features = [df.reindex(index=index) for df in features]
        features = [df.rename(columns={col: f'f_{i}__{col}' for col in df.columns}) for i, df in enumerate(features)]
        return pd.concat(features, axis=1)

    def fit_transform(self, df_target):
        features = self.get_df_features(df_target)
        X = features
        for p in self.preprocessing:
            X = p.fit_transform(X)
        return X

    def transform(self, df_target):
        features = self.get_df_features(df_target)
        X = features
        for p in self.preprocessing:
            X = p.transform(X)
        return X

    def load_preprocessors(self, preprocessing):
        self.preprocessing = []
        for t_name, params in preprocessing:
            p = cls_loader.create(t_name, params)
            self.preprocessing.append(p)

    def load_features(self):
        _start = datetime.datetime.now()
        current_features = self.conf.features[self.feature_name]
        read_params = current_features['read_params']
        if type(read_params) is not list:
            read_params = [read_params]

        self.feature_list = [FeatureFile.read_table(self.conf, **f) for f in read_params]
        self.load_time = datetime.datetime.now() - _start

    def get_feature_fit_info(self):
        info = []
        for p in self.preprocessing:
            if type(p) is CategoryEncoder:
                info.append(f'Encoded cols: {p.cols_for_encoding}')
                info.append(f'Dropped cols: {p.cols_for_drop}')
        return ' '.join(info)

