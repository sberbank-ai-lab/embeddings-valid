"""
export LUIGI_CONFIG_PATH=/mnt/data/kireev/pycharm_1_vec_test/luigi.cfg
# use `--local-schedule` for debug purpose


cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH='.' luigi \
    --workers 5 \
    --module vector_test ReportCollect \
    --conf test_conf/train-test.hocon --total-cpu-count 18
less test_conf/train-test.txt

cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/crossval.work/; rm test_conf/crossval.txt
PYTHONPATH='.' luigi \
    --workers 5 \
    --module vector_test ReportCollect \
    --conf test_conf/crossval.hocon --total-cpu-count 18
cat test_conf/crossval.txt
"""

import luigi

if __name__ == '__main__':
    luigi.run()
