# Run Instructions

This document provides instructions to reproduce the test predictions using the AutoML solution described in the provided notebooks.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or above
- Jupyter Notebook or Jupyter Lab
- Necessary Python libraries:
pandas
numpy
scikit-learn
autogluon
auto-sklearn
flaml
tpot
mljar-supervised

You can install the required libraries using:
```bash
pip install pandas numpy scikit-learn autogluon auto-sklearn flaml tpot mljar-supervised
```

## Data Preparation

Ensure you have the exam dataset files in the following structure:
../../data/exam_dataset/
├── X_train.parquet
├── y_train.parquet
└── X_test.parquet


## Train the models

1. Open and run the 'Final_submission.ipynb' notebook in Jupyter.

2. For AutoSKLearn:
   - Use a Conda environment with AutoSKLearn installed.
   - Run the AutoSKLearn cell to train the model and save predictions.

3. For other models (FLAML, AutoGluon, TPOT, MLJAR):
   - Switch to a Python 3 kernel.
   - Run the cells for each model to train and save predictions.

Note: We recommend using Amazon SageMaker Studio Lab or a similar platform that allows easy kernel switching.

## Generating Final Predictions

After training all models, run the following cells:

1. Load all saved predictions.

2. Create the final ensemble prediction using the following weights (determined from previous experiments):
   - AutoSklearn: 0.067
   - Gluon: 0.584
   - MLJAR: 0.349

3. Save the final predictions:

```python
final_exam_pred = (0.067 * autosklearn_pred + 0.584 * gluon_pred + 0.349 * mljar_pred)
predictions_array = final_exam_pred.to_numpy()
np.save('final_test_preds.npy', predictions_array)
```	

This process should complete within 24 hours and yield the final_test_preds.npy file containing predictions for the test set of the final-test-dataset.

## PS:

The 'All_Tools_On_Deck.ipynb' notebook also contains the code for training all models and saving predictions. However, it also contains prevoius experiments (some faulty ones as well) if that may be of interest.
