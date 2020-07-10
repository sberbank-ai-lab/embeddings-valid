import json
import logging
import os

import luigi

from vector_test.config import Config
from vector_test.file_reader import TargetFile
from vector_test.metrics import Metrics
from vector_test import cls_loader
from vector_test.tasks.fold_splitter import FoldSplitter
from vector_test.x_transformer import XTransformer


class FoldEstimator(luigi.Task):
    conf = luigi.Parameter()
    model_name = luigi.Parameter()
    feature_name = luigi.Parameter()
    fold_id = luigi.IntParameter()

    total_cpu_count = luigi.IntParameter()

    def requires(self):
        return FoldSplitter(conf=self.conf)

    def output(self):
        conf = Config.read_file(self.conf)
        fold_name = '__'.join([
            f'm_{self.model_name}',
            f'f_{self.feature_name}',
            f'i_{self.fold_id}'
        ])
        path = os.path.join(conf.work_dir, fold_name, 'results.json')
        return luigi.LocalTarget(path)

    @property
    def resources(self):
        conf = Config.read_file(self.conf)

        cpu_count = conf.models[self.model_name]['cpu_count']
        return {'cpu': round(cpu_count / self.total_cpu_count, 2)}

    def run(self):
        conf = Config.read_file(self.conf)

        x_transf = XTransformer(conf, self.feature_name)
        conf_model = conf.models[self.model_name]
        model = cls_loader.create(conf_model['cls_name'], conf_model['params'])
        scorer = Metrics(conf)
        on_error = conf.error_handling

        results = {
            'fold_id': self.fold_id,
            'model_name': self.model_name,
            'feature_name': self.feature_name,
        }

        # folds with train-valid-test ids and targets
        with self.input().open('r') as f:
            folds = json.load(f)
        current_fold = folds[str(self.fold_id)]

        try:
            target_train = TargetFile.load(current_fold['train']['path'])
            X_train = x_transf.fit_transform(target_train)
            model.fit(X_train, target_train.target_values)

            if scorer.is_check_train:
                results['scores_train'] = scorer.score(model, X_train, target_train.target_values)
            results['scores_valid'] = self.score_data(current_fold['valid']['path'], x_transf, model, scorer)
            if current_fold['test'] is not None:
                results['scores_test'] = self.score_data(current_fold['test']['path'], x_transf, model, scorer)

        except BaseException:
            if on_error == conf.ON_ERROR_SKIP:
                results = None
                logging.getLogger('luigi-interface').exception('Fail', stack_info=True)
            elif on_error == conf.ON_ERROR_FAIL:
                raise
            else:
                raise AssertionError(f'Unknown error_handling: "{on_error}"')

        with self.output().open('w') as f:
            results = [] if results is None else [results]
            json.dump(results, f, indent=2)

    def score_data(self, target_path, x_transf, model, scorer):
        target_data = TargetFile.load(target_path)
        X_data = x_transf.transform(target_data)
        return scorer.score(model, X_data, target_data.target_values)
