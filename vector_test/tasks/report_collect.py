import json
import os
import datetime
from functools import partial

import pandas as pd
from pandas import json_normalize
import scipy.stats

import luigi

from vector_test.config import Config
from vector_test.tasks.external_score import ExternalScore
from vector_test.tasks.fold_estimator import FoldEstimator


def values(x):
    return x.tolist()


def print_list(x, float_format):
    return '[' + ' '.join([float_format.format(i) for i in x]) + ']'


def t_interval(x, p=0.95):
    eps = 1e-9
    n = len(x)
    s = x.std(ddof=1)

    return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=(s + eps) / (n ** 0.5))


def t_int_l(x, p=0.95):
    return t_interval(x, p)[0]


def t_int_h(x, p=0.95):
    return t_interval(x, p)[1]


def t_pm(x, p=0.95):
    l, h = t_interval(x, p)
    return (h - l) / 2


class ReportCollect(luigi.Task):
    conf = luigi.Parameter()
    total_cpu_count = luigi.IntParameter()

    def requires(self):
        conf = Config.read_file(self.conf)

        for model_name in conf.models:
            for feature_name in conf.features:
                for fold_id in conf.folds:
                    yield FoldEstimator(
                        conf=self.conf,
                        model_name=model_name,
                        feature_name=feature_name,
                        fold_id=fold_id,
                        total_cpu_count=self.total_cpu_count,
                    )
        for name, external_path in conf.external_scores.items():
            yield ExternalScore(
                conf=self.conf,
                name=name,
                external_path=external_path,
            )

    def output(self):
        conf = Config.read_file(self.conf)

        path = os.path.join(conf.root_path, conf['report_file'])
        return luigi.LocalTarget(path)

    def load_results(self):
        parts = []
        total_count = 0
        error_count = 0
        for i in self.input():
            total_count += 1
            with open(i.path, 'r') as f:
                scores = json.load(f)

            if len(scores) == 0:
                error_count += 1
                os.remove(i.path)
            parts.extend(scores)

        pd_report = json_normalize(parts)
        return pd_report, total_count, error_count

    @staticmethod
    def format_report(pd_report):
        pd_report = pd_report.melt(id_vars=['model_name', 'feature_name', 'fold_id'], var_name='_metric')
        pd_report = pd.concat([
            pd_report,
            pd_report['_metric'].str.extract(r'(?P<split_name>\w+)\.(?P<metric_name>\w+)'),
        ], axis=1)
        other_metrics = sorted(pd_report['metric_name'].unique().tolist())
        pd_report = pd_report.sort_values(['metric_name', 'model_name', 'feature_name', 'split_name', 'fold_id'])
        pd_report = pd_report.groupby(['metric_name', 'model_name', 'feature_name', 'split_name'])['value'].agg(
            ['mean', t_pm, t_int_l, t_int_h, 'std', values])
        return pd_report, other_metrics

    def run(self):
        conf = Config.read_file(self.conf)

        pd_report, total_count, error_count = self.load_results()
        pd_report, other_metrics = self.format_report(pd_report)

        splits = []
        if conf['report.is_check_train']:
            splits.append('scores_train')
        splits.extend(['scores_valid', 'scores_test'])

        with self.output().open('w') as f:
            self.print_header(f)
            self.print_errors(f, total_count, error_count)

            for k in conf['report'].keys():
                if k in ('is_check_train', 'error_handling'):
                    continue

                self.print_row_pandas(f, k, pd_report, splits, **conf['report'].get(k, {}))
                other_metrics = [m for m in other_metrics if m != k]

            if len(other_metrics) > 0:
                print('Other metrics:', file=f)
                for k in other_metrics:
                    self.print_row_pandas(f, k, pd_report, splits)

            self.print_footer(f)

    def print_header(self, f):
        self.print_line(f)
        _text = f"""Vector testing report
Params:
    conf: "{self.conf}"
"""
        print(_text, file=f)

    def print_errors(self, f, total_count, error_count):
        print(f"Collected {total_count} files with {error_count} errors", file=f)
        if error_count > 0:
            print(f"Check logs for detail information", file=f)
        print('', file=f)

    def print_row_pandas(self, f, metric_name, pd_report, splits,
                         keep_columns=None, float_format='{:.4f}'):
        self.print_line(f)
        with pd.option_context(
                'display.float_format', float_format.format,
                'display.max_columns', None,
                'display.max_rows', None,
                'display.expand_frame_repr', False,
                'display.max_colwidth', 200,
        ):
            print(f'Metric: "{metric_name}"', file=f)
            if keep_columns is None:
                keep_columns = pd_report.columns

            df = pd_report.loc[metric_name].unstack().swaplevel(axis=1).reindex(
                columns=[(l0, l1) for l0 in splits for l1 in keep_columns])
            if 'values' in keep_columns:
                for l0 in splits:
                    df[(l0, 'values')] = df[(l0, 'values')].apply(partial(print_list, float_format=float_format))

            print(df, file=f)
            print('', file=f)


    def print_footer(self, f):
        self.print_line(f)
        _text = f"End of report.     Current time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        print(_text, file=f)
        self.print_line(f)

    def print_line(self, f):
        print('-' * 120, file=f)
