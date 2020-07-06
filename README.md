# Vector test project
Estimate your feature vector

# Concepts
## Target file
Tabular file with id and target columns.
This is a base for quality estimation.

## Feature file
Tabular file with id and feature columns.
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

## Feature preprocessing
Some features requires preprocessing, you can configure it.

## Missing values
Filled by ...

## Feature report
We can estimate features and provide some recommendations about preprocessing.

## Final report
Contains 3 sections for train, validation and optional test files.
Each section contains results for different feature files and it combinations.
Each result contains mean with confidence intervals of N scores from N folds (or iterations).
Measure can be some metric or time or some parameter like feature count.

## Config
All settings should be described in configuration file.
Report preparation splits on task and execution plan prepared based on config.


# Execution plan
1. Read target files, split it based on validation schema.
2. Run feature check task for each feature file
3. Based on feature file combinations prepare feature estimate tasks.
2. Run train-estimate task for each split. Save results.
3. Collect report

