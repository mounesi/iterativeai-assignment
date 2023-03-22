## Computer Vision Project: Sign Language MNIST
This is a test project for building a production-ready pipeline for machine learning experiments to classify Sign Language images using the Sign Language MNIST dataset. The project requires building a pipeline with three stages: data loading, training, and evaluation. The project uses DVC to manage data versioning and pipeline execution.

### Dataset
The Sign Language MNIST dataset consists of 27,455 grayscale images of hand gestures representing letters in American Sign Language. The dataset is split into 24,855 training images and 2,600 test images.

### Project Structure
The project has the following structure:

```
.
├── data/
│   ├── sign_mnist_test.csv
│   └── sign_mnist_train.csv
├── models/
│   └── my_model.h5
├── src/
│   ├── data_load.py
│   ├── evaluate.py
│   └── train.py
├── .dvcignore
├── .dvc/
├── .gitignore
├── dvc.yaml
├── params.yaml
└── README.md
```

`data/` contains the input data for the pipeline: the training and test sets in CSV format.
`models/` contains the trained model in Keras format.
`src/` contains the source code for the pipeline, with separate scripts for data loading, training, and evaluation.
`.dvcignore` specifies the files and directories to ignore when running DVC commands.
`.dvc/` contains the DVC cache and metadata.
`.gitignore` specifies the files and directories to ignore when committing to Git.
`dvc.yaml` specifies the pipeline stages and dependencies.
`params.yaml` specifies the parameters and hyperparameters for the pipeline.
`README.md` (this file) contains information about the project and how to use it.

### Pipeline
The pipeline consists of three stages:

1. data_load: loads the training and test data from CSV files, preprocesses it, and saves it to disk in a format suitable for training.
2. train_model: loads the preprocessed data, trains a convolutional neural network (CNN) model on the training set, and saves the trained model to disk.
3. evaluate_model: loads the trained model and the preprocessed test data, evaluates the model on the test set, and prints the evaluation metrics.


The pipeline uses DVC to manage data versioning and pipeline execution. The input data is downloaded from Kaggle and cached locally by DVC. The preprocessed data and the trained model are also cached by DVC.

### Usage
To run the pipeline, follow these steps:

Clone the repository:

```
git clone https://github.com/mounesi/iterativeai-assignment.git
cd repo
```
Install the required packages:

```
pip install -r requirements.txt
```

Set up your Kaggle credentials by creating a kaggle.json file in the .kaggle/ directory with your API key. Refer to the Kaggle documentation for instructions.

Download the input data by running the following command:

```
dvc pull
```
Run the pipeline by running the following command:

```
dvc repro
```

### Conclusion

This test project demonstrates how to build a production-ready pipeline for machine learning experiments using DVC. By using DVC to manage data versioning and pipeline execution, we can easily rerun the pipeline with different hyperparameters and input data, and track the results of each experiment.

In this project, we used the Sign Language MNIST dataset to build a CNN model to classify hand gestures representing letters in American Sign Language. The pipeline consists of three stages: data loading, training, and evaluation. The input data is downloaded from Kaggle, preprocessed, and cached by DVC. The trained model and the preprocessed data are also cached by DVC, so that the pipeline can be rerun with different hyperparameters without having to recompute everything from scratch.

We hope that this project will serve as a useful reference for building your own machine learning pipelines using DVC.

