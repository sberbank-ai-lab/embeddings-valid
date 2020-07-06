# Vector test project
Estimate your feature vector quality on downstream task

# Concepts
## Target file
Tabular file with id field(s) and target columns.
This is a base for quality estimation.

Id can be one ore many columns. dtypes can be string, int, date or datetime.
Id columns are used for join with other filed. Names of columns should be identical across the files,
and you can rename some columns when file loads.

Targets for all data should be in one file. Many target files will be in the next releases.

## Id file
Contains only ids columns. Used for split data on folds for train, valid and test.
You can use `Target file` as `Id file`. Target columns will bew removed.

## Feature file
Tabular file with ids and feature columns.
You can estimate this specific feature quality for target prediction,
compare it with features from an other file or mix features from many files and compare ensemble.

Id used for join feature and target file. Join is always left: `target` left join `features` on `id`.
Missing values filled according preprocessing strategy.
 
## Validation schema
### Cross validation setup
Only one target file is provided. This tool split it on N folds and use each one for validation.
We have N scores provided by N models trained on N-1 folds each.
Result is a mean of N scores with p% confidence interval.

### Train-test setup
Train and valid target files are provided. Train on 1st and mesure score on 2nd file.
Repeat it N times.
We have N scores provided by N models trained on full train file.
Result is a mean of N scores with p% confidence interval.

### Isolated test (optional)
Isolated test target file can be provided.
All models for both cross-validation and train-test setup estimated on isolated test.
N scores from N models provided.
Result is a mean of N scores with p% confidence interval.

## Estimator
Many estimators can be used for learn feature and target relations.
We expect sklearn-like fit-predict interface.

## Metrics
Choose the metrics for feature estimate.

There are predefined metrics like predict time or feature count.

## Feature preprocessing
Some features requires preprocessing, you can configure it.

## Missing values
Filled by ...

## Feature report
Next releases. We can estimate features and provide some recommendations about preprocessing.

## Final report
For each metrics we provide on part of report.
Each part contains 3 sections for train, validation and optional for test files.
Each section contains results for different feature files and it combinations.
Each result contains mean with confidence intervals of N scores from N folds (or iterations).

## Config
All settings should be described in single configuration file.
Report preparation splits on task and execution plan prepared based on config.


# Execution plan
1. Read target files, split it based on validation schema.
2. (Next release) Run feature check task for each feature file
3. Based on feature file combinations prepare feature estimate tasks.
2. Run train-estimate task for each split. Save results.
3. Collect report


# Environment
`vector_test` module should be available for current python.

Directory with config file is `root_path`. All paths described in config starts from `root_path`.
`config.work_dir` is a directory where intermediate files and reports are saved.
`config.report_file` is a file with final report.

# How to run
## Test example `train-test.hocon`
```
# delete old files
rm -r test_conf/train-test.work/
rm test_conf/train-test.txt

# run report collection
PYTHONPATH='.' luigi --local-schedule \
    --module vector_test ReportCollect \
    --conf test_conf/train-test.hocon

# check final report
cat test_conf/train-test.txt
```
## Test example `crossval.hocon`
```
# delete old files
rm -r test_conf/crossval.work/
rm test_conf/crossval.txt

# run report collection
PYTHONPATH='.' luigi --local-schedule \
    --module vector_test ReportCollect \
    --conf test_conf/crossval.hocon

# check final report
cat test_conf/crossval.txt
```
