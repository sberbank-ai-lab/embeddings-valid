import os

import luigi
import json

from sklearn.model_selection import StratifiedKFold, KFold

from vector_test.config import Config
from vector_test.file_reader import TargetFile, IdFile


class FoldSplitter(luigi.Task):
    conf = luigi.Parameter()

    def output(self):
        conf = Config.read_file(self.conf)
        path = os.path.join(conf.work_dir, 'folds', 'folds.json')
        return luigi.LocalTarget(path)

    def run(self):
        conf = Config.read_file(self.conf)
        validation_schema = conf.validation_schema

        self.output().makedirs()

        folds = None
        if validation_schema == Config.VALID_TRAIN_TEST:
            folds = self.train_test_split()
        if validation_schema == Config.VALID_CROSS_VAL:
            folds = self.cross_val_split()
        assert folds is not None

        with self.output().open('w') as f:
            json.dump(folds, f, indent=2)

    def _read_id_split_save(self, df_target, read_file_params, save_path):
        conf = Config.read_file(self.conf)
        ids = IdFile.read_table(conf, **read_file_params)
        df_target_fold = df_target.select_ids(ids)
        path = os.path.join(conf.work_dir, 'folds', save_path)
        df_target_fold.dump(path)
        return {'path': path, 'shape': df_target_fold.df.shape}

    def _select_pos_save(self, df_target, pos, save_path):
        conf = Config.read_file(self.conf)
        df_target_fold = df_target.select_pos(pos)
        path = os.path.join(conf.work_dir, 'folds', save_path)
        df_target_fold.dump(path)
        return {'path': path, 'shape': df_target_fold.df.shape}

    def train_test_split(self):
        conf = Config.read_file(self.conf)

        df_target = TargetFile.read_table(conf, **conf['target'])
        train_info = self._read_id_split_save(df_target, conf['split.train_id'], 'target_train.pickle')
        valid_info = self._read_id_split_save(df_target, conf['split.valid_id'], 'target_valid.pickle')
        if 'test_id' in conf['split']:
            test_info = self._read_id_split_save(df_target, conf['split.test_id'], 'target_test.pickle')
        else:
            test_info = None

        folds = {
            i: {
                'train': train_info,
                'valid': valid_info,
                'test': test_info,
            } for i in range(conf['split.n_iteration'])
        }
        return folds

    def get_folds(self, df_target):
        conf = Config.read_file(self.conf)
        cv_split_count = conf['split.cv_split_count']
        random_state = conf['split.random_state']
        if conf['split.is_stratify']:
            skf = StratifiedKFold(cv_split_count, shuffle=True, random_state=random_state)
            return skf.split(df_target.df, df_target.target_values)
        else:
            sf = KFold(cv_split_count, shuffle=True, random_state=random_state)
            return sf.split(df_target.df)

    def cross_val_split(self):
        conf = Config.read_file(self.conf)

        df_target = TargetFile.read_table(conf, **conf['target'])

        if 'test_id' in conf['split']:
            test_info = self._read_id_split_save(df_target, conf['split.test_id'], 'target_test.pickle')
        else:
            test_info = None

        ids = IdFile.read_table(conf, **conf['split.train_id'])
        df_target_train = df_target.select_ids(ids)

        folds = {}
        for i, (i_train, i_valid) in enumerate(self.get_folds(df_target_train)):
            train_info = self._select_pos_save(df_target_train, i_train, f'target_train_{i}.pickle')
            valid_info = self._select_pos_save(df_target_train, i_valid, f'target_valid_{i}.pickle')

            folds[i] = {
                'train': train_info,
                'valid': valid_info,
                'test': test_info,
            }

        return folds



