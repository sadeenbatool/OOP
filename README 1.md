# Project OPP

-[Libraries](#libraries)

-[Load the Dataset](#load-the-dataset)

-[Data Analyzer](#data-anlyzer)

-[Graph Plotter](#graph-plotter)

-[Univariate Plot](#univariate-plot)

-[Bivaraite Plot](#bivaraite-plot)

-[SVM Trainer Class](#svm-trainer-class)

-[Model Persistence Class](#model-persistence-class)

## Libraries

import pandas as pd

    Purpose: Used for reading and manipulating structured data (CSV, Excel, etc.)

    Common Use: pd.read_csv(), df.head(), df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

    Seaborn: High-level plotting (histograms, heatmaps, violin plots)

    Matplotlib: Low-level control for customizing charts

%matplotlib inline

    Purpose: Jupyter-specific magic command to render plots inline (right below code cells)

from sklearn.preprocessing import StandardScaler

    Purpose: Standardizes features (mean=0, std=1)

    Why: SVMs are sensitive to feature scaling

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

    train_test_split: Separates data into training/testing sets

    GridSearchCV: Tunes hyperparameters using cross-validation

    cross_val_score: Measures model performance over multiple splits

from sklearn.svm import SVR, SVC

    SVR: Support Vector Regressor (for continuous output)

    SVC: Support Vector Classifier (for categorical output)

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

    Regression Metrics:

      mean_squared_error, mean_absolute_error, r2_score

    Classification Metrics:

     accuracy_score, precision_score, recall_score, f1_score

from datetime import datetime

    Purpose: For logging, versioning, or adding timestamps to saved files.

import pickle

    Purpose: Save Python objects like models or scalers to disk.

   Alternatives: joblib (preferred for large NumPy arrays)

## Load the Dataset

df = pd.read_csv("StudentsPerformance.csv")

    Reads data from a .csv file into a DataFrame, which is a table-like structure with rows and columns.

df.head()

    Returns the top 5 rows (by default) of the dataset to help preview and inspect the structure and sample values.

## Data Anlyzer

class DataAnalyzer:

    Load a dataset from a CSV file.

    Perform summary statistics.

    Inspect the structure and content of the dataset.

    Check for missing values and duplicates.

    Understand data types and relationships.

def __init__(self, file_path):

    Takes a CSV file path as input.

    Stores it for future loading via pandas.read_csv().

def data_load(self):

     Reads the CSV into a DataFrame and stores it as self.df.

def head(self, n=5):

     Shows the top n rows of the DataFrame.

def sample(self, n=5):

    Returns a random sample of n rows from the dataset.

def tail(self, n=5):

    Displays the last n rows of the dataset.

def datatypes(self):

    Shows the data types of each column.

def describe(self):

    Summarizes statistics of numeric columns (mean, std, min, max, etc.).

def check_nulls(self):

      Returns a count of missing values in each column.

def column_names(self):

      Returns a list of all column headers.

def shape(self):

     Returns a tuple representing the number of rows and columns.

def info(self):

     Displays concise details about the DataFrame, including:

         Column names

         Data types

         Null counts

         Memory usage

def size(self):

    Returns the total number of data points (cells) in the dataset.

def correlation_matrix(self, numeric_only=True):

    Shows pairwise correlation between numeric columns.

    Helps identify relationships and multicollinearity.

def duplicated(self):

    Counts how many rows in the dataset are exact duplicates.

## Graph Plotter

class GraphPlotter:

    A Python class designed to:

    Display distributions of numeric data.

    Visualize category frequencies.

    Explore relationships between features.

    Understand inter-variable correlations.

def __init__(self, data):

    Accepts a pandas DataFrame (data) and stores it as an instance variable for all plotting methods.

def histogram(self, column, bins=10, title=None):

    Visualizes the distribution of a numerical column.

    Adds a KDE (Kernel Density Estimate) line to understand the probability distribution.

    Useful for detecting skewness, outliers, and central tendencies.

def countplot(self, column, title=None):

    Displays the frequency (count) of each category in a categorical column.

    Helps identify imbalances or dominant classes.

def scatter_plot(self, x_col, y_col, title=None):

    Plots individual points for two numeric variables on a 2D plane.

    Reveals trends, clusters, correlations, and potential outliers.

def heatmap(self, title="Correlation Heatmap"):

    Displays the correlation matrix of numerical variables as a color-coded grid.

    Annotated with correlation coefficients.

    Extremely useful for feature selection and detecting multicollinearity in machine learning.

## Univariate Plot

class UnivariatePlots:

    Definition: A univariate plot deals with a single variable.

     A visualization utility that supports:

     Distribution analysis

     Outlier detection

     Category frequency inspection

def __init__(self, data):

    Stores the input DataFrame (data) to use in all plotting functions.

def histogram(self, column, bins=10, title=None):

    Purpose: Plot the frequency distribution of a numeric column.

    Extras: Overlays a KDE (Kernel Density Estimate) to show data distribution smoothness.

    Use Case: Ideal for detecting skewness, modality, and outliers.

def boxplot(self, column, title=None):

    Purpose: Show the distribution, central tendency, and presence of outliers.

    Use Case: Great for detecting outliers and comparing medians across groups.

def violinplot(self, column, title=None):

    Purpose: Combines boxplot and KDE for richer distribution insights.

    Use Case: Useful for seeing multimodal distributions along with quartile data.

def kde_plot(self, column, title=None):

    Purpose: Plots a smooth estimate of the variable’s probability density function.

    Use Case: Ideal for understanding the overall shape and spread of the data.

def countplot(self, column, title=None):

    Purpose: Shows the count of unique values in a categorical column.

    Use Case: Helpful to spot dominant or underrepresented classes.

def pie_chart(self, column, title=None):

    Purpose: Displays category proportions in a pie chart.

    Use Case: Great for presentations where proportion-based visualizations are preferred.

## Bivaraite Plot

class BivariatePlots:
     
    Definition: A bivariate plot visualizes the relationship between two variables.
    
    A visualization toolset built on Matplotlib and Seaborn.

    Supports both categorical vs. numerical and numerical vs. numerical comparisons.

def __init__(self, data):

    data (pd.DataFrame): Dataset containing the variables to be visualized.

    Description: Stores the input DataFrame to be used across all plotting methods.

scatter_plot(x_col, y_col, title=None)

    Type: Numeric vs. Numeric

    Purpose: Shows relationship and spread between two continuous variables.

    Use Case: Detect linear/non-linear correlation or clustering patterns.

 line_plot(x_col, y_col, title=None)

    Type: Numeric vs. Numeric (often over time or sequence)

    Purpose: Visualizes trends, especially useful for time-series or progression data.

    Use Case: Plotting performance over time or growth metrics.

bar_plot(x_col, y_col, title=None)

     Type: Categorical vs. Numeric

    Purpose: Compares mean/aggregate values of a numeric variable across categories.

    Use Case: Compare average scores between groups (e.g., gender, ethnicity).

 box_plot(x_col, y_col, title=None)

     Type: Categorical vs. Numeric

     Purpose: Displays distribution, median, quartiles, and outliers per group.

     Use Case: Spot variability or skewed data between groups.

violin_plot(x_col, y_col, title=None)

    Type: Categorical vs. Numeric

    Purpose: Combines boxplot and KDE to show richer distribution by category.

    Use Case: Analyze multimodal distributions across groups.

 strip_plot(x_col, y_col, title=None)

    Type: Categorical vs. Numeric

    Purpose: Plots individual observations (with jitter) for density insight.

    Use Case: Useful for visualizing raw values and their grouping.

swarm_plot(x_col, y_col, title=None)

    Type: Categorical vs. Numeric

    Purpose: Improved version of strip plot; avoids overlapping dots.

    Use Case: Dense datasets where exact values and groupings matter.

## SVM Trainer Class

class SVMTrainer:

    Built for Support Vector Regression (SVR) or Support Vector Classification (SVC).

    Handles data preparation, scaling, model training, and evaluation.

    Provides an intuitive interface for users working with structured datasets.

def __init__(self, data, target_column, task='regression'):

    data: pd.DataFrame — Input dataset.

    target_column: str — Column to be predicted.

    task: str, default 'regression' — Either 'regression' or 'classification'.

def preprocess(self, drop_columns=None, fillna_value=None):

    Removes or fills missing values.

    Drops unnecessary columns if specified.

    Encodes categorical variables via One-Hot Encoding.

    Splits into features X and target y.

def split_data(self, test_size=0.2):

    Splits X and y into training and testing subsets.

    Uses train_test_split with a default test_size=0.2.

def scale_data(self):

    Applies StandardScaler to scale numerical features.

    Ensures all variables contribute equally to the SVM model.

def train_model(self, kernel='rbf', C=1.0, gamma='scale'):

    Trains an SVR model (by default).

    Hyperparameters:

    kernel: e.g. 'linear', 'rbf', 'poly'

    C: regularization strength

    gamma: kernel coefficient

def predict(self):

    Predicts target values using the trained model on X_test.

def evaluate(self):

     Evaluates model using:

        Mean Squared Error (MSE)

        Mean Absolute Error (MAE)

        R² Score

## Model Persistence Class

class ModelPersistence

    Scope: Serialization and deserialization of ML objects.

    Use case: Save models after training for reuse without retraining, especially in production.

 save_model(model, filename="svm_model.pkl")

    Saves the trained model to disk using binary format.

    Default filename: "svm_model.pkl".

save_scaler(scaler, filename="scaler.pkl")

    Stores the scaler used (e.g., StandardScaler) for future use on new data.

    Essential for ensuring consistent data transformation during inference.

 save_metadata(metadata_dict, filename="metadata.pkl")

    Saves a dictionary containing extra model info like:

      hyperparameters

      training date

      author info

      dataset summary

load_model(filename="svm_model.pkl")

     Loads and returns the previously saved model.

 load_scaler(filename="scaler.pkl")

    Restores the scaler for use on new incoming data.

 load_metadata(filename="metadata.pkl")

    Loads and returns the metadata dictionary for reference or audits.
