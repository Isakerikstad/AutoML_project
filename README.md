[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/_jadz4xj)
# AutoML Exam - SS24 (Tabular Data)
This repo serves as a template for the exam assignment of the AutoML SS24 course
at the university of Freiburg.

The aim of this repo is to provide a minimal installable template to help you get up and running.

To obtain test results on the _final dataset_, refer [here](#Autoevaluation-on-the-test-dataset).

## Installation

To install the repository, first create an environment of your choice and activate it. 

You can change the Python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-tabular-env python=3.11
conda activate automl-tabular-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:
```bash
python -c "import automl"
```

We make no restrictions on the Python library or version you use, but we recommend using Python 3.8 or higher.

## Code
We provide the following:

* `download-openml.py`: A script to download the dataset from openml given a `--task`, corresponding to an OpenML Task ID. We wil provide those suggested as training datasets, prior to us releasing the test dataset.

* `run.py`: A script that loads in a downloaded dataset, trains an _AutoML-System_ and then generates predictions for
`X_test`, saving those predictions to a file. For the training datasets, you will also have access to `y_test` which
is present in the `./data` folder, however you **will not** have access to `y_test` for the test dataset we provide later.
Instead you will generate the predictions for `X_test` and submit those to us through github classrooms.

* `./src/automl`: This is a python package that will be installed above and contain your source code for whatever
system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

**You are completely free to modify, install new libraries, make changes and in general do whatever you want with the
code.** The only requirement for the exam will be that you can generate predictions for `X_test` in a `.npy` file
that we can then use to give you a test score through github classrooms.

## Data

### Practice datasets:

* [y_prop_4_1 (361092)](https://www.openml.org/search?type=task&id=361092&collections.id=299&sort=runs)
* [Bike_Sharing_Demand (361099)](https://www.openml.org/search?type=task&id=361099&collections.id=299&sort=runs)
* [Brazilian Houses (361098)](https://www.openml.org/search?type=task&id=361098&collections.id=299&sort=runs)

You can download the required OpenML practice data using:
```bash
python download-openml.py --task <task_id>
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers, `1, ..., n` are **outer folds**, meaning you can treat each one of them as
a seperate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── 361092
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── 361098
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```
### Final Test dataset:

**UPDATE**
The final dataset can now be downloaded by running:
```bash
python download-openml.py --task exam_dataset
```

In case you have trouble with the automatic download, you can also download it [here](https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-24-tabular/exam_dataset.zip). Unzip the file and place it in the `/data` folder with the following structure.
There will be only one fold, `1`, for the test dataset.
```bash
./data
├── exam-dataset
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   └── y_train.parquet
```

**Note:** You will not have access to `y_test` for the test dataset, only `X_test`.

## Autoevaluation on the test dataset

Only activates on the `test` branch:
```bash
git checkout -b test  # to create the branch
# or
# if branch exists
git checkout test
git merge <name-of-branch-where-current-code-to-test-exists>
# ensure that your latest `predictions.npy` exists
git push origin test  # depending on MERGE-CONFLICTS might need to resolve and add files
# wait for some time (few minutes) or monitor the web UI of Github to see Actions passing
git pull
# test scores will be downloaded under `./exam_dataset/test_out/`
```

* To initialize auto-evaluation for the test data, create a `test` branch. 
* After publishing it, the evaluation script will automatically trigger.
* After creating the `test` branch, you may also run the evaluation script on any other branch manually.
  * To do that, navigate to the `Actions` tab at the GitHub remote repository and proceed by pressing the `Run workflow` button.
  * Triggering the workflow by pushing to the `test` branch is highly recommended for your own logging purposes (use the commit message).
* For the evaluation to run correctly, make sure the `predictions.npy` is at the right location.
* The results are also pushed to your repo (don't forget to `git pull`)
* If no test predictions generated, check the errors in the Github action (red cross inline with your last commit on the test branch)

```bash
./data
├── exam_dataset
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_train.parquet
│   │   └── predictions.npy
│   └── test_out
│   │   ├── test_evaluation_output_2024-MM-DD_HH-mm
.   .   .
```

Note that any edits to the test evaluation script are prohibitted and monitored! In case you want to change anything, contact the TA team!

## Running an initial test
This will train a dummy AutoML system and generate predictions for `X_test`:
```bash
python run.py --task brazilian_houses --seed 42 --output-path preds-42-brazil.npy
```

You are free to modify these files and command line arguments as you see fit.

## Final submission

The following must be submitted by `August 6, 2024, 23:59 CET` for a successful project submission and poster participation:

#### **1) Poster submission**
Upload your poster as a PDF file named as `final_poster_tabular_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1lyE-iLGXIKi31CLFwueGhjfcsR_8r7_L/edit?usp=sharing&ouid=107220015291298974152&rtpof=true&sd=true).

#### **2) Test predictions**
The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

#### **3) Reproducibility instructions**
TL;DR: Code and instructions to _reproduce_ the above test predictions.

A `run_instructions.md` file that guides through the command to run the designed AutoML solution on the training set of the *final-test-dataset*.
This command should return either a: (i) hyperparameter configuration, (ii) a partially trained model on a hyperparameter configuration, or (iii) a fully trained model in `24 hours` at most.
A second command that given (i), (ii), or (iii) would do the needful that yields predictions for `test_X` for the *final-test-dataset*. This is the `final_test_preds.npy`.

#### **4) Team information**
Upload a file `team_info.txt` with the list of matriculation IDs of team members (*NO NAMES*). (E.g.: 1234567, 7654321)

### Submission checklist:
- [ ] Poster
- [ ] Test predictions
- [ ] Reproducibility instructions
- [ ] Team info
- [x] *Example to denote task being done*
<!-- This is a comment. -->

## Reference performance

| Dataset | Final Test performance |
| -- | -- |
| y_prop | 0.87 |
| bike_sharing | 0.935 |
| brazilian_houses | 0.969 |
| final test dataset | 0.873 |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.

## Tips
* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
`pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
created by your IDE.
