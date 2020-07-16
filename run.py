"""
export LUIGI_CONFIG_PATH=/mnt/data/kireev/pycharm_1_vec_test/luigi.cfg

# debug run
# use `--local-schedule` for debug purpose
cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH='.' luigi \
    --workers 5 \
    --module vector_test ReportCollect \
    --conf test_conf/train-test.hocon --total-cpu-count 18
less test_conf/train-test.txt


# production run
cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m vector_test --workers 5 --conf test_conf/train-test.hocon --total_cpu_count 18
less test_conf/train-test.txt

cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/crossval.work/; rm test_conf/crossval.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m vector_test --workers 5 --conf test_conf/crossval.hocon --total_cpu_count 18
less test_conf/crossval.txt


cd /mnt/data/kireev/pycharm_1_vec_test/
rm -r test_conf/single-file.work/; rm test_conf/single-file.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m vector_test --workers 5 --conf test_conf/single-file.hocon --total_cpu_count 18
less test_conf/single-file.txt


cd /mnt/data/kireev/pycharm_1/dltranz/experiments/scenario_gender/
rm -r conf/vector_test.work/; rm results/vector_test.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test"   \
    python -m vector_test --workers 10 --conf conf/vector_test.hocon --total_cpu_count 9
less results/vector_test.txt

"""
