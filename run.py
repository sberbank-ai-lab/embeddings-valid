"""
cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH='.' luigi --local-schedule \
    --module vector_test ReportCollect \
    --conf test_conf/train-test.hocon
cat test_conf/train-test.txt

cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/crossval.work/; rm test_conf/crossval.txt
PYTHONPATH='.' luigi --local-schedule \
    --module vector_test ReportCollect \
    --conf test_conf/crossval.hocon
cat test_conf/crossval.txt
"""

import luigi

if __name__ == '__main__':
    luigi.run()
