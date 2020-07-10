import json
import os

import luigi

from vector_test.config import Config


class ExternalScore(luigi.Task):
    conf = luigi.Parameter()
    name = luigi.Parameter()
    external_path = luigi.Parameter()

    def output(self):
        conf = Config.read_file(self.conf)
        path = os.path.join(conf.work_dir, 'external', self.name, 'scores.json')
        return luigi.LocalTarget(path)

    def run(self):
        conf = Config.read_file(self.conf)

        path = os.path.join(conf.root_path, self.external_path)
        with open(path, 'r') as f:
            external_scores = json.load(f)

        with self.output().open('w') as f:
            json.dump(external_scores, f, indent=2)
