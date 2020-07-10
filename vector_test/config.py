import argparse
import os

import logging
from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)


class Config:
    VALID_TRAIN_TEST = 'train-test'
    VALID_CROSS_VAL = 'crossval'

    def __init__(self, conf=None, root_path=None):
        self.conf = conf
        self.root_path = root_path

    @classmethod
    def get_conf(cls, args=None):
        p = argparse.ArgumentParser()
        p.add_argument('-c', '--conf', required=True)
        args, overrides = p.parse_known_args(args)

        logger.info(f'args: {args}, overrides: {overrides}')
        name = args.conf
        logger.info(f'Load config from "{name}"')
        file_conf = ConfigFactory.parse_file(name, resolve=False)

        root_path = os.path.dirname(os.path.abspath(name))

        overrides = ','.join(overrides)
        over_conf = ConfigFactory.parse_string(overrides)
        conf = over_conf.with_fallback(file_conf)
        return cls(conf=conf, root_path=root_path)

    @classmethod
    def read_file(cls, file_name):
        logger.info(f'Load config from "{file_name}"')
        file_conf = ConfigFactory.parse_file(file_name, resolve=False)

        root_path = os.path.dirname(os.path.abspath(file_name))

        return cls(conf=file_conf, root_path=root_path)

    def __getitem__(self, item):
        return self.conf[item]

    def _read_enabled(self, key):
        return {name: {k: v for k, v in params.items() if k != 'enabled'}
                for name, params in self.conf[key].items() if params['enabled']}

    @property
    def work_dir(self):
        return os.path.join(self.root_path, self.conf['environment.work_dir'])

    @property
    def data_files(self):
        return self._read_enabled('data_files')

    @property
    def features(self):
        data_files = self.data_files
        features = self._read_enabled('features')
        features = {name: {'data_files': [data_files[k] for k in params['data_files']]}
                    for name, params in features.items()}
        features.update({name: {'data_files': [x]} for name, x in data_files.items()})
        return features

    @property
    def external_scores(self):
        return {k: v for k, v in self.conf['external_scores'].items()}

    @property
    def models(self):
        return self._read_enabled('models')

    @property
    def validation_schema(self):
        split_params = self.conf['split']

        if 'train_id' not in split_params:
            raise AttributeError(f'There is no "train" key in "split" config')

        if 'valid_id' in split_params:
            if 'n_iteration' not in split_params:
                raise AttributeError(f'"n_iteration" should be defined '
                                     f'when both "train" and "valid" keys are presented in "split" config')
            return self.VALID_TRAIN_TEST
        else:
            for attr in ['cv_split_count', 'is_stratify', 'random_state']:
                if attr not in split_params:
                    raise AttributeError(f'"{attr}" should be defined when only "train" key is presented '
                                         f'and "valid" key is absent in "split" config')
            return self.VALID_CROSS_VAL

    @property
    def folds(self):
        split_params = self.conf['split']
        validation_schema = self.validation_schema

        if validation_schema == self.VALID_TRAIN_TEST:
            return [i for i in range(split_params['n_iteration'])]
        if validation_schema == self.VALID_CROSS_VAL:
            return [i for i in range(split_params['cv_split_count'])]

        raise AssertionError(f'Unknown validation_schema: {validation_schema}')

    @property
    def metrics(self):
        return self._read_enabled('metrics')
