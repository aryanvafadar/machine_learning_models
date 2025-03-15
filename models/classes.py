# import numpy and pandas
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib
import os
import glob
from pathlib import Path
from config import output_files_folder, get_prediction_csv, get_ml_file

# import sklearn metrics, preprocessing and training/testing data splits
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedKFold

# import linear and non linear models for classification and regression tasks
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier


""" DataFrame Creator Class"""
class DatasetCreator:
    """
    Class that creates a dataframe from a csv file, and prepares the frame for the model testing and fitting. 

    To instantiate the DatasetCreator object, a csv file must be passed in.
    
    The class provides also provides functions to:
    1) created_frame(): Create a DataFrame from a csv file. Returns a dataframe
    2) clean_frame(): Cleanes the dataframe. Returns a cleaned dataframe
    3) encode_frame(): Encodes all non-numeric columns to numeric, and returns an encoded frame. The encoded frame is necessary for ML models.
    """
    
    def __init__(self, csv_file):
        """To instantiate a DatasetCreator object, we need to pass in a csv_file"""
        self.csv_file = csv_file
        self.initial_frame = None
        self.cleaned_frame = None
        self.encoded_frame = None
        
        self.removed_columns = None

    def create_frame(self) -> pd.DataFrame:
        try:
            """Create the initial dataframe with the csv_file, and return an initial frame"""
            logging.info(f"Opening file {self.csv_file} to create into a pandas dataframe.")
            
            frame = pd.read_csv(self.csv_file)
            frame.columns = frame.columns.map(lambda x: x.lower() if isinstance(x, str) else x)
            self.initial_frame = frame
            
            logging.info("Initial DataFrame successfully created and set to self.initial_frame")
            logging.info(f"Shape of Initial Frame: {self.initial_frame.shape}")
            logging.info(f"Initial Frame Columns: {self.initial_frame.columns}")
            logging.info(f"Initial Frame DataTypes: {dict(self.initial_frame.dtypes)}")
            logging.info(f"Intial Frame Contains Null Values?: {self.initial_frame.isna().any()}")
            logging.info(f"Total Number of Null Values in Each Column: {self.initial_frame.isna().sum()}")
            
            return frame
        
        except Exception as e:
            logging.error(f"Unable to create initial dataframe. Received error {e}")

    def remove_columns(self, columns: list) -> bool:
        """
        Removes a specific list of columns from the initial DataFrame. Updates self.initial_frame with the remaining columns.

        Args:
            columns (list): List of columns to remove from the DataFrame.

        Returns:
            bool: True if columns are removed successfully, False otherwise.
        """
        
        # ensure self.initial_frame has been initialized
        if self.initial_frame is None or self.initial_frame.empty:
            logging.error("Unable to remove columns. No initial_frame created. Please create the initial_frame first.") 
            return None
        
        try:
            logging.info(f'remove_columns function called. List of columns to remove: {columns}')
            
            # current columns in the dataframe. columns have already been set to lowercase
            current_cols = list(self.initial_frame.columns)
            logging.info(f"Current columns in the dataframe: {current_cols}")
            
            # columns the user would like to remove, set to lowercase
            cols_to_remove = [col.lower() for col in columns]
            logging.info("Requested to remove columns have been set to lowercase.")
        
            # Check if all requested to remove columns exist in the dataframe by using set difference.
            missing_cols = set(cols_to_remove) - set(current_cols)
            if missing_cols:
                logging.error(f"Columns not found in the dataframe: {missing_cols}. Please remove and retry.")
                raise Exception
            
            # if all columns exist in the DataFrame, log and remove the columns
            logging.info("All columns exist within the DataFrame, and will be removed.")
            self.initial_frame = self.initial_frame.drop(columns=cols_to_remove)
            
            # set self.removed_columns 
            self.removed_columns = cols_to_remove
            
            # log results
            logging.info(f"Columns {cols_to_remove} have been removed from the dataframe.")
            logging.info(f"Columns that remain in the frame: {list(self.initial_frame.columns)}.")
            
            return True
            
            
        except Exception as e:
            logging.error(f"Unable to remove columns from the initial frame. Received error {e}")
            return False
    
    def handle_nulls(self, handle_method: list[str], column: list[str], fill_value=None, replace_value=None) -> pd.DataFrame:
        """
        Handle null values within the DataFrame.

        Args:
            - handle_method (list[str]): List of methods to handle nulls. Options: 'drop_all', 'fill_all', 'back_fill', 'forward_fill', 'replace_all', 'interpolate'.
            - column (list[str]): List of columns where null handling should be applied.
            - fill_value (optional): Value to use for 'fill_all' method.
            - replace_value (optional): Value to use for 'replace_all' method.

        Returns:
            - Updated DataFrame or None if an error occurs.
        """
        frame = self.initial_frame.copy()
        available_methods = ['drop_all', 'fill_all', 'back_fill', 'forward_fill', 'replace_all', 'interpolate']

        # Normalize column names and method inputs
        frame_columns = frame.columns.str.lower()
        handle_method = [method.lower() for method in handle_method]
        column = [col.lower() for col in column]

        # Validate input lengths
        if len(handle_method) != len(column):
            logging.error(f"Mismatch between methods and columns. Methods: {len(handle_method)}, Columns: {len(column)}.")
            return None

        # Validate method and column names
        invalid_methods = set(handle_method) - set(available_methods)
        invalid_columns = set(column) - set(frame_columns)
        if invalid_methods or invalid_columns:
            logging.error(f"Invalid input detected. Methods: {invalid_methods}, Columns: {invalid_columns}.")
            return None

        try:
            # Apply null handling methods to each column
            for method, col in zip(handle_method, column):
                if method == 'fill_all':
                    if fill_value is None:
                        logging.error("fill_value is required for 'fill_all' method.")
                        return None
                    frame[col] = frame[col].fillna(value=fill_value)
                    logging.info(f"Filled nulls in column '{col}' with '{fill_value}'.")
                elif method == 'back_fill':
                    frame[col] = frame[col].fillna(method='bfill')
                    logging.info(f"Backfilled nulls in column '{col}'.")
                elif method == 'forward_fill':
                    frame[col] = frame[col].fillna(method='ffill')
                    logging.info(f"Forward-filled nulls in column '{col}'.")
                elif method == 'replace_all':
                    if replace_value is None:
                        logging.error("replace_value is required for 'replace_all' method.")
                        return None
                    frame[col] = frame[col].replace(to_replace=np.nan, value=replace_value)
                    logging.info(f"Replaced nulls in column '{col}' with '{replace_value}'.")
                elif method == 'interpolate':
                    frame[col] = frame[col].interpolate()
                    logging.info(f"Interpolated nulls in column '{col}'.")

            # Apply drop_all last if specified
            if 'drop_all' in handle_method:
                frame.dropna(inplace=True)
                logging.info("Dropped all rows with null values.")
                
            # Log null values to the user
            logging.info(f"Null Check: {frame.isnull().sum()}")

            # Update and return the modified DataFrame
            self.initial_frame = frame
            logging.info("Successfully handled null values. Updated DataFrame set to self.initial_frame.")
            return self.initial_frame

        except Exception as e:
            logging.error(f"Error handling null values: {e}")
            return None
            
    def move_label_end(self, label: str) -> bool:
        """
        Moves the specified label/target column in self.initial_frame to the end.

        Args:
            label (str): The column name to move to the end.

        Returns:
            bool: True if the label was successfully moved, False otherwise.
        """
        
        # check if label exists within the dataframe
        if label not in list(self.initial_frame.columns):
            logging.error("Label does not exist in the current list of columns. Please review and retry.")
            return False
        
        try:
            logging.info(f"Label {label} exists in the dataframe. Attempting to remove...")
            
            # remove the label column from the dataframe. .pop() returns a series
            label_col = self.initial_frame.pop(label)
            logging.info('Label column has been removed from the dataframe, and will be reinserted at the end.')
            logging.info(f"Label Column: {label_col}")
            
            # add label back to the dataframe
            self.initial_frame[label] = label_col
            logging.info("Label column added back to the dataframe.")
            logging.info(f"Sample of new self.initial_frame: {self.initial_frame.sample(10)}")
            
            return True
        
        except Exception as e:
            logging.error(f"Unable to remove and reinsert the label column to the end of the dataframe. Received error {e}")
            return False
    
    def clean_frame(self) -> pd.DataFrame:
        """
        Takes self.initial_frame and cleans the frame by removing whitespaces and symbols, commas and special characters.
        
        Sets the cleaned frame to self.cleaned_frame and returns a new frame if successful. Else returns None if not successful.
        """
        
        try:
            logging.info("Frame cleaning function called. Making a copy of self.initial_frame before beginning cleaning.")
            frame = self.initial_frame.copy()
            
            frame.columns = frame.columns.str.strip() # remove whitespaces from column headers
            logging.info("Whitespaces from column headers have been removed.")
            
            # remove whitespaces from rows
            frame = frame.apply(lambda col: col.str.strip() if col.dtype == 'object' else col) 
            logging.info("Whitespaces removed from rows/samples in the dataset.")
            
            # remove special characters, symbols from rows
            frame = frame.replace(to_replace=r"[^a-zA-Z0-9\s]", value="", regex=True) 
            logging.info("Special characters, symbols and commas removed from the frame.")
            
            logging.info("Self.initial_frame has been cleaned. New cleaned frame has been set to self.cleaned_frame")
            self.cleaned_frame = frame
            
            return self.cleaned_frame
        
        except Exception as e:
            logging.error(f"Unable to clean dataframe. Received error {e}")
            return None

    def encode_frame(self, label: str) -> pd.DataFrame:
        """
        Iterates through self.cleaned_frame and encodes all string valus within the frame.
        
        If the column has 2 unique string values then the column will be binary encoded through simple 0/1 mapping.
        
        If the column has more than 2 unique string values, it will be OneHotEncoded with SkLearns OneHotEncoder.
        
        Returns an encoded dataframe and sets it to self.encoded_frame. Otherwise returns None if not successful.
        """
        
        logging.info("Frame encoding function has been called. All string data will be converted to numeric (int).")
        
        try:
            # Make a coyp of self.cleaned_frame
            frame = self.cleaned_frame.copy()
            logging.info("Self.cleaned_frame has been copied.")
            
            # Instantiate OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, drop=None)
            logging.info("SkLearn OneHotEncoder has been instantiated.")

            # Iterate through each column and look for object datatypes. These have strings in them
            logging.info("Iterating through the dataframe to search for columns whose data is of type object.")
            
            for column in frame.select_dtypes(include=['object']):
                if not frame[column].dtype == object:
                    logging.info(f"DataFrame column {column} is of type int or float, and no encoding needed.")
                    continue

                # get number of unique values in the column
                num_uniques = frame[column].nunique()
                logging.info(f"Number of uniques found in column {column}: {num_uniques}")
        
                # if num_uniques equals to 2, then apply simple binary mapping
                if num_uniques == 2:
                    logging.info(f"Because column {column} has 2 unique values, it will be binary encoded.")
                    uniques_list = list(frame[column].unique())
                    mapping = {
                        uniques_list[0]: 0,
                        uniques_list[1]: 1,
                    }
                    frame[column] = frame[column].map(mapping)
                    logging.info(f"Encoding completed. Mapping: {mapping}")
        
                # if num_uniques is greater than 2, then apply OneHotEncoding
                if num_uniques > 2:
                    logging.info(f"Because column {column} has more than 2 unique values, it will be OneHotEncoded.")
        
                    # encode the column
                    encoded_col = encoder.fit_transform(frame[[column]]) # make sure col is passed as 2d array[[]]
                    logging.info(f"Column {column} has been OneHotEncoded.")
                    
                    # create a new dataframe with the encoded column
                    encoded_df = pd.DataFrame(data=encoded_col, columns=encoder.get_feature_names_out([column]))
                    logging.info("New dataframe with the encoded values has been created.")
                    
                    # combine the 2 dataframes
                    frame = pd.concat((frame.drop(columns=[column]), encoded_df), axis=1)
                    logging.info("Original dataframe and new dataframe have been comibined on the y axis.")
                    
            
            logging.info("Dataframe has finished being encoded. Label/Target column will now be removed and re-added to the end of the frame.")    
            label_column = frame.pop(label)  # Remove the label column from the DataFrame
            frame[label] = label_column      # Add it back at the end
            logging.info(f"Label column {label} has been removed and readded to the end of the dataframe.")

            self.encoded_frame = frame
            return self.encoded_frame
        
        except Exception as e:
            logging.error(f"Unable to encode dataframe. Received error {e}")
            return None

    @staticmethod
    def frame_info(frame: pd.DataFrame) -> None:
        """Print/Log information about the dataframe"""

        null_values_by_column = frame.isnull().sum()
        logging.info(frame.head(10))
        logging.info(frame.info())
        logging.info(frame.shape)
        logging.info(null_values_by_column)

    @staticmethod
    def date_to_datetime(frame: pd.DataFrame, column_name: str, errors: str, drop_original_column: bool, datetime_cols: list, is_weekend: bool) -> pd.DataFrame:
        """
        Converts the date column of a pandas DataFrame into a datetime format and adds additional columns such as year, month, and day.

        Args:
            - frame (pd.DataFrame): A pandas DataFrame.
            - column_name (str): The name of the date column in the DataFrame.
            - errors (str): How errors should be handled: 'ignore', 'raise', or 'coerce'.
            - drop_original_column (bool): Whether to drop the original date column from the DataFrame.
            - datetime_cols (list): List of datetime attributes to add. Options: 'year', 'month', 'day', 'weekday'.
            - is_weekend (bool): Whether to add a column in the DataFrame to check if weekday is a weekend. 1 = True.

        Returns:
            pd.DataFrame: The modified DataFrame with new datetime columns added, or None if an error occurs.
        """
        logging.info("Date to Datetime function has been called.")

        # Ensure the column names are in lowercase for consistency
        frame.columns = frame.columns.str.lower()
        column_name = column_name.lower()

        if column_name not in frame.columns:
            logging.error(f"Column '{column_name}' not found in the DataFrame columns: {frame.columns.tolist()}")
            return None

        logging.info(f"Date column '{column_name}' exists in the DataFrame.")

        # Validate the datetime columns
        valid_datetime_attrs = {'year', 'month', 'day', 'weekday'}
        datetime_cols = [col.lower() for col in datetime_cols]
        invalid_cols = set(datetime_cols) - valid_datetime_attrs

        if invalid_cols:
            logging.error(f"Invalid datetime columns requested: {invalid_cols}. Valid options are: {valid_datetime_attrs}")
            return None

        logging.info("All datetime columns requested are valid.")

        try:
            # Convert the date column to datetime format
            frame[column_name] = pd.to_datetime(frame[column_name], errors=errors.lower())
            logging.info(f"Date column '{column_name}' successfully converted to datetime format.")

            # Add the requested datetime attributes as new columns
            for col in datetime_cols:
                if col == 'year':
                    frame['year'] = frame[column_name].dt.year
                elif col == 'month':
                    frame['month'] = frame[column_name].dt.month
                elif col == 'day':
                    frame['day'] = frame[column_name].dt.day
                elif col == 'weekday':
                    frame['weekday'] = frame[column_name].dt.dayofweek  # Monday = 0, Sunday = 6

                logging.info(f"Datetime attribute '{col}' has been added to the DataFrame.")
            
            # if is_weekend is True, add a new column "is_weekend" to state if the day is a weekend. 1 = True, 0 = False    
            if is_weekend:
                frame['is_weekend'] = frame['weekday'].apply(lambda x: 1 if x > 4 else 0)
                logging.info("is_weekend column has been added to the DatFrame")
                
            # Drop the original date column if requested
            if drop_original_column:
                frame.drop(columns=[column_name], inplace=True)
                logging.info("Original date column has been dropped from the DataFrame.")
            else:
                logging.info("Original date column has been retained in the DataFrame.")

            logging.info(f"Final DataFrame columns after datetime processing: {frame.columns.tolist()}")
            return frame

        except Exception as e:
            logging.error(f"Failed to convert date column to datetime format. Error: {e}")
            return None
    
    @staticmethod
    def calc_percent_change(frame: pd.DataFrame, num_columns: list, ascend: bool, multiply_by_100: bool) -> pd.DataFrame:
        """
        Takes a pandas dataframe and calculates percent changes on the requested columns.
        
        Args:
            - frame (pd.DataFrame): A pandas dataframe.
            - num_columns (list): A list of numeric columns to calculate percentage change on.
            - ascend (bool): If True, calculates the percent change from oldest to newest record (bottom to top). If false, calculates the percent change from top to bottom.
            - multiply_by_100 (bool): If True, multiplies the calculated percent change value by 100. If false, does not multiply by 100.
        
        Returns:
            - If successful, returns a new pandas dataframe.
            - If unsuccessful, returns none.
        """
        
        # check if frame is empty
        if frame.empty:
            logging.error("The input DataFrame is empty. Please provide a non-empty DataFrame.")
            return None
        
        # check if frame is a dataframe
        if not isinstance(frame, pd.DataFrame):
            logging.error("Frame argument is not of type pandas dataframe. Please try again.")
            return None
        
        # set column heads and num_list to lower_case
        frame.columns = frame.columns.str.strip().str.lower()
        num_columns = [col.lower() for col in num_columns]
        
        try:
            
            # check if all columns in num_columns exist in the dataframe
            invalid_cols = set(num_columns) - set(frame.columns)
            if invalid_cols:
                logging.error(f"Not all columns passed in by num_columns exist in the dataframe. Invalid cols: {invalid_cols}")
                return None
            
            # check ascend is a bool
            if not isinstance(ascend, bool):
                logging.error("Ascend argument is not of type bool. Please try again.")
                return None
            
            # check each column passed into num_columns is numeric
            for col in num_columns:
                try:
                    pd.to_numeric(arg=frame[col], errors='raise')
                    
                except Exception as e:
                    logging.error(f"Unable to convert column '{col}' to numeric. Received error {e}")
                    return None
                
            # Check if ascend is true. If it is, reverse the frame
            if ascend:
                frame = frame[::-1]
                logging.info("Frame successfully reversed.")
            
            # Calculate the percent change for each column
            for col in num_columns:
                percent_col_name = f"{col}_percent_change"
                frame[percent_col_name] = frame[col].pct_change()
                logging.info(f"Percent change column calculated and added for column {col}")
            
            # If ascend is true, reverse the frame back to the original order
            if ascend:
                frame = frame[::-1]
                logging.info("Frame successfully unreversed.")
                
            # drop NaN Values
            frame.dropna(inplace=True)
            
            # if multiply_by_100 is true, multiply the new colums by 100
            if multiply_by_100:
                
                # get list of new column names
                percent_change_cols = [f"{col}_percent_change" for col in num_columns]
                
                # multiply the columbs by 100
                frame[percent_change_cols] *= 100
                
                logging.info(f"Columns {percent_change_cols} have been multipled by 100.")
             
            # return the final frame    
            return frame
            
        except Exception as e:
            logging.error(f"Unable to calculate percent change. Received error {e}")
            return None
            
    @staticmethod
    def export_frame_to_csv(frame: pd.DataFrame, file_name: str, output_folder=output_files_folder) -> True:
        """Converts a dataframe to a csv_file. Takes in 2 arguments; an output file folder and the file name"""
        try:
            frame.to_csv(path_or_buf=f"{output_files_folder}/{file_name}.csv", index=False)
            return True
            
        except Exception as e:
            print(f"Unable to export dataframe to a csv file. Received error {e}")
            return False
    
""" Regression Model Tester Class"""
class RegModelTester:
    """
    Class that contains various functions to help determine which machine learning models work best on any given dataset.
    This class is intended for regression tasks where the target value is numeric.

    Steps of using the class:
        1.	Train the model on X_train and y_train.
    	2.	Evaluate the model on X_val and y_val to decide which hyperparameters perform best.
    	3.	After finding the best hyperparameters, we will re-train the model on the full training data (X_train_full, y_train_full) 
            and evaluate the final model on the test set (X_test, y_test).
    """

    def __init__(self, frame):
        """Initialize the ModelTester with a cleaned, null-free, encoded DataFrame."""
        self.frame = frame 
        
        # Initialize split data attributes
        # After first split
        self.X_train_full = None # Only used to help split the 2nd dataset and run a final training run with our tuned model
        self.X_test = None # Use at the end only, to test the trained and tuned model on unseen data (test set)
        self.y_train_full = None # Only used to help split the 2nd dataset and run a final training run on our turned model
        self.y_test = None # Use at the end only, to test the trained and tuned model on unseen data (test set)
        
        # After second split
        self.X_train = None # Use to train the best model and to-be tuned model
        self.X_val = None # Use to find the best model and validate (tune hyperparams)
        self.y_train = None # Use to train the best model and to-be tuned model
        self.y_val = None # Use to find the best model and validate (tune hyperparams)
        
        # get frature variances, best and worst features
        self.features_variances = []
        self.best_features = []
        self.worst_features = []

        # metric scores from model tester
        self.current_model_name = None
        self.current_model = None
        self.current_model_params = None
        self.r2 = 0
        self.mae = 0
        self.mse = 0
        
        # best params and score from our optimizer functions
        self.best_params = None
        self.best_score = 0
        
        # results from our tuned model
        self.tuned_model = None
        self.tuned_r2_score = 0
        self.tuned_mae_score = 0
        self.tuned_mse_score = 0

    def get_features_labels(self, model_test_size) -> bool:
        """
        Extract features and labels, then split the data into training, validation, and testing sets.
        
        Standardization is also applied to the training and testing features (self.X_train_full and self.X_train).

        Args:
            model_test_size (float): The proportion of the dataset to reserve for testing. Eg: 0.30, 0.40, 0.55

        Returns:
            True: If successful
            False: If unsuccessful
        """
        
        try:
            
            logging.info("Get features and labels function has been called.")
            
            # Check self.frame exists / is not empty
            if self.frame.empty:
                logging.error("The input DataFrame is empty.")
                raise ValueError("The input DataFrame is empty.")

            # get column names
            column_names = list(self.frame.columns)
            logging.info(f"Columns in DataFrame: {column_names}")
            
            if len(column_names) < 2:
                raise ValueError("The DataFrame must have at least one feature column and one label column.")

            # Features are all columns except the last; the last column is the label
            features = column_names[:-1]
            labels = column_names[-1]
            logging.info(f"List of Features: {features}. These columns will be used to create our X variable.")
            logging.info(f"List of Labels: {labels}. This will be used for our y variable.")
            
            X = self.frame[features]
            y = self.frame[labels]
            logging.info(f"X DataFrame created, and will be used for train_test_split. X Frame Shape: {X.shape}")
            logging.info(f"y DataFrame created, and will be used for train_test_split. y Frame Shape: {y.shape}")
            
            # Split the data into train-test and train-validation sets
            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(X, y, test_size=model_test_size, random_state=6712792)
            
            # Standardize the Data
            scaler = StandardScaler()
            self.X_train_full = scaler.fit_transform(self.X_train_full)
            self.X_test = scaler.transform(self.X_test)
            
            logging.info("X and y variables have been split into training and testing data. This first split should not be used to train and tune the machine learing model. It should only be used for training and testing after the model has been trained and tuned.")
            logging.info(f"First Split Test Size: {model_test_size}")
            logging.info(f"Size X_train_full: {self.X_train_full.shape} | Size X_test: {self.X_test.shape}")
            logging.info(f"Size of y_train_full: {self.y_train_full.shape} | Size y_test: {self.y_test.shape}")

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_full, self.y_train_full, test_size=0.20, random_state=6712792)
            
            logging.info("X_train_full and y_train_full have been split again to create training and testing data to train and tune our model. These new variables (X_train, X_val, y_train, y_val) should only be used to find which model works best on our dataset, and then subsequently tuning this model.")
            logging.info(f"Size X_train: {self.X_train.shape} | Size X_val: {self.X_val.shape}")
            logging.info(f"Size of y_train: {self.y_train.shape} | Size y_val: {self.y_val.shape}")
            
            logging.info("Testing and Training data have successfully been created. Ready now to get the best model.")
            return True

        except Exception as e:
            print(f"Error in get_features_labels: {e}")
            return False
    
    def get_variances(self) -> pd.DataFrame:
        """
        Returns the variance of the features in a DataFrame. Can be used prior to using the features_analysis function in RegModelTester() class.
        """
        try:
            variances = self.X_train.var(axis=0)
            self.features_variances = variances
            logging.info(f"Features Variance: {self.features_variances}")
                
        except Exception as e:
            logging.error(f"Unable to get the variances of the data. Received error {e}")
            return None
    
    def feature_analysis(self, label: str, use_VarianceThreshold: bool = True, variance_threshold: int = 0.01, use_SelectKBest: bool = True, num_top_features: int = 10):
        
        """
        Get the least and most important features of any DataSet, with the goal of completing regression tasks. An explanation of each feature analysis method is provided below.
        
        VarianceThreshold
            - VarianceThreshold is an unsupervised feature selector that removes features whose variance is below a specified threshold. 
            - For each feature within the Dataset, it calculates the variance. Any variance below 0.1 is considered to have too low variability within the samples, and is therefore removed. 
            - Features with low variance generally do not provide any value to our machine learning model, and do not help us in making a prediction. By removing them, it simplifies our model and can help improve performance.
            
        SelectKBest + f_regression
            - SelectKBest is a supervised feature selection method that picks the top k features based on a scoring function. 
            - The scoring function, f_regression, computes the f-statistic for each for each feature by assessing its linear relationship with the target variable y. A higher f-statistic means the feature is more likely to be significantly correlated with the target variable (in a linear sense).
            - The top X features are selected, based on user request. 
            
        Interpreting F_Scores & P_Values from SelectKBest
            - The higher the f_score, the the higher the chances that a relationship exists between the feature and the target.
            - Extremely low p_values indicate that the relationship between the feature and label is highly statistically significant. It is almost impossible that the observed relationship is due to random chance. An extremely low p_value is considered to be 0.05 or 0.01
            
        We recommend using this two step approach of:
            1) First filtering out the features that don't vary enough to be informative.
            2) Selecting the features that are the most important for the Dataset.
        
        """
        
        try:
            # if use_VarianceThreshold is true
            if use_VarianceThreshold:
                feature_names = self.frame.drop(columns=[label]).columns
                
                selector = VarianceThreshold(threshold=variance_threshold) # instantiate VarianceThreshold object()
                selector.fit(X=self.X_train) # finds the features who are below the threshold
                mask = selector.get_support() # Gets the boolean mask
                selected_features = feature_names[mask]
                logging.info(f"Selected Features from VarianceThreshold: {selected_features}")
                
                # outline which features have been removed
                removed_features = set(self.frame.columns) - set(selected_features)
                if removed_features:
                    logging.info(f"Suggested Features to Remove: {removed_features}")
                else:
                    logging.info("No features need to be removed from dataset.")
                
            # if use_SelectKBest is true:
            if use_SelectKBest:
                kbest = SelectKBest(score_func=f_regression, k=num_top_features)
                X_kbest = kbest.fit(X=self.X_train, y=self.y_train)
                
                # log the best scores
                logging.info(f"SelectKBest Best Scores: {kbest.scores_}")
                logging.info(f"SelectKBest p_values: {kbest.pvalues_}")
                
                # get the feature names
                feature_names = self.frame.drop(columns=[label]).columns
                mask = kbest.get_support()
                best_features = list(feature_names[mask])
                
                # log feature and score
                best_scores = list(kbest.scores_)
                best_p_values = list(kbest.pvalues_)
                for feature, score, p_value in zip(best_features, best_scores, best_p_values):
                    logging.info(f"Feature: {feature} | KbestScore: {score} | P_Value: {p_value}")
                
        
        except Exception as e:
            logging.error(f"Unable to perform feature analysis. Received error {e}")
            return None
    
    def get_best_models(self, n_iterations: int) -> dict:   
        
        """
        Iterates through the DataFrame passed in through RegModelTester() to find the best model for the dataset. The best model has its name, model and scores set to self.
        
        Args:
            - n_iterations (int): Number of times we would like the function to test each model on our dataset. 
        
        Returns: 
            - A dict of all the models tested and their scores if successful, otherwise returns None.
        """
        
        logging.info("get_best_models function has been called.")


    # Dict of linear models
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=0.01, max_iter=10000),
            'lasso_regression': Lasso(alpha=0.01, max_iter=10000),
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            #'huber_regression': HuberRegressor(max_iter=10000),
            'linear_support_vector': LinearSVR(max_iter=10000)
        }

        # Dict of non-linear models
        non_linear_models = {
            'gaussian_regressor': GaussianProcessRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'hist_boosting': HistGradientBoostingRegressor(),
            'random_forest': RandomForestRegressor(),
            'extra_trees': ExtraTreesRegressor(),
            'decision_tree': DecisionTreeRegressor(),
            'k_nearest_neighbors': KNeighborsRegressor(),
            'support_vector_regressor': SVR(),
            # 'adaboost': AdaBoostRegressor(n_estimators=100),
            # 'bagging_regressor': BaggingRegressor(n_estimators=100),
            # 'stacking_regressor': StackingRegressor(estimators=[
            #     ('rf', RandomForestRegressor()), 
            #     ('gb', GradientBoostingRegressor())
            # ]),
            # 'neural_network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000),
        }

        # combine both dictionaries
        all_models = {
            **linear_models,
            **non_linear_models
        }
        
        logging.info(f"List of models to test: {all_models}")

        # creates an empty results dict to store our results
        results = {}
        
        try:
            
            logging.info("Testing each model in the all_models dict now.")
            
            logging.info(f"User passed in {n_iterations} number of iterations. We will run the model testing {n_iterations+1} amount of times.")
            
            for idx, model_run in enumerate(range(n_iterations + 1), start=1):
                
                logging.info(f"Model Run: {idx}")
                
                # Iterate through the dictionaries and test each model
                for name, model in all_models.items():

                    try:
                        
                        logging.info(f"Testing Model: {model}")

                        # train the model
                        model.fit(X=self.X_train, y=self.y_train)
                        logging.info(f"Training Model: {model}")
            
                        # make a prediction
                        prediction = model.predict(X=self.X_val)
            
                        # measure results
                        r2 = r2_score(y_true=self.y_val, y_pred=prediction)
                        mae = mean_absolute_error(y_true=self.y_val, y_pred=prediction)
                        mse = mean_squared_error(y_true=self.y_val, y_pred=prediction)
            
                        # store metrics in result dict
                        results[name] = {
                            'R2_Score': round(r2, 2),
                            'MAE_Score': float(round(mae, 2)),
                            'MSE_Score': float(round(mse, 2))
                        }
            
                        # Update current model and metrics if r2 score of model on current iteration is higher than what is stored
                        if r2 > self.r2:
                            self.current_model_name = name
                            self.current_model = model
                            self.current_model_params = model.get_params()
                            self.r2 = r2
                            self.mae = mae
                            self.mse = mse
                            
                            logging.info(f"Model {model} is the best model so far for our dataset.")

                    except Exception as e:
                        logging.error(f"Received error: {e}.")
                        return None

            # Sort the results by R2 score in descending order
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['R2_Score'], reverse=True))
            logging.info("Each model has been tested. Results have been sorted from best -> worst.")
            
            # print the top model
            logging.info(f"Top Model: {self.current_model_name} | R2 Score: {self.r2}")

            return sorted_results
        
        except Exception as e:
            logging.info(f"Unable to get the best models. Received error {e}")
            return None

    def optimize_gradient_boosting_model(self, optimize_method: str, n_iterations: int):
        """
        Optimize parameters for a Gradient Boosting model.

        Arguments:
            - optimize_method (str): Can only be 'grid' or 'random' to select the optimization method.
            - n_iterations (int): Number of iterations for RandomSearchCV.
        """

        try:
            # Ensure current model is a GradientBoostingRegressor
            if not isinstance(self.current_model, GradientBoostingRegressor):
                logging.error("This optimization function is only applicable for Gradient Boosting models.")

            # Get the number of samples and features from the training set
            n_samples, n_features = self.X_train.shape
            n_splits = 5 if n_samples > 5000 else 3  # Dynamically select number of splits

            # Instantiate the cross-validation strategy
            repeat = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)

            # Define parameter grids for both optimization methods
            grid_params = {
                'n_estimators': [50, 100, 150, 250, 500, 750] if n_samples < 5000 else [100, 250, 500, 750, 1000, 1500],
                'max_depth': list(range(2, 10)) if n_samples < 5000 else list(range(5, 25, 2)),
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
            }

            random_params = {
                'n_estimators': stats.randint(50, 1000),
                'max_depth': stats.randint(2, 20),
                'learning_rate': stats.uniform(0.01, 0.1)
            }

            if optimize_method.lower() == 'grid':
                # Perform GridSearchCV
                grid_cv = GridSearchCV(
                    estimator=self.current_model,
                    param_grid=grid_params,
                    n_jobs=-1,
                    cv=repeat,
                    scoring='r2',
                    refit=True
                )
                grid_cv.fit(self.X_train, self.y_train)

                # Store the best results
                self.best_params = grid_cv.best_params_
                self.best_score = grid_cv.best_score_

                logging.info(f"GridSearch Best Score: {self.best_score}")
                logging.info(f"GridSearch Best Parameters: {self.best_params}")

            elif optimize_method.lower() == 'random':
                # Perform RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    estimator=self.current_model,
                    param_distributions=random_params,
                    n_iter=n_iterations,
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    cv=repeat,
                    random_state=6712792
                )
                random_search.fit(self.X_train, self.y_train)

                # Store the best results
                self.best_score = random_search.best_score_
                self.best_params = random_search.best_params_

                logging.info(f"RandomSearch Best Score: {self.best_score}")
                logging.info(f"RandomSearch Best Parameters: {self.best_params}")

            else:
                logging.error(f"Invalid optimize_method argument: {optimize_method}. Expected 'grid' or 'random'.")
                return None

        except Exception as e:
            logging.error(f"Error in optimizing Gradient Boosting model: {e}")
        
    def optimize_hist_boosting_model(self, optimize_method: str, n_iterations: int, scoring: str = 'r2'):
        """
        Optimize parameters for a HistBoosting model. 
        
        Args:
            - optimize_method (str): 'grid' or 'random' to select GridSearchCV or RandomSearchCV.
            - n_iterations (int): Number of iterations for RandomSearchCV.
            - scoring (str): Scoring metric for optimization. Default is 'r2'.
        """
    
        logging.info("Optimize HistBoosting Model function has been called.")
        
        # Validate model type
        if not isinstance(self.current_model, HistGradientBoostingRegressor):
            logging.error("Optimization is only available for HistGradientBoostingRegressor models.")
            return None
        
        # Validate optimization method
        optimize_method = optimize_method.lower()
        if optimize_method not in ['grid', 'random']:
            logging.error(f"Invalid optimization method: {optimize_method}. Expected 'grid' or 'random'.")
            return None

        # Log dataset info
        n_samples, n_features = self.X_train.shape
        logging.info(f"Samples: {n_samples}, Features: {n_features}")
        
        # Dynamic cross-validation split
        n_splits = 5 if n_samples > 5000 else 3
        logging.info(f"Number of splits for RepeatedKFold: {n_splits}")
        
        # Cross-validation strategy
        repeater = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)

        # Define hyperparameter search grids
        grid_params = {
            "max_iter": [50, 100, 150, 250, 500, 750, 1000],
            "max_depth": list(range(2, 10)),
            "min_samples_leaf": list(range(5, 20)),
            "max_bins": [128, 255, 512]
        }

        random_params = {
        "learning_rate": stats.uniform(0.01, 0.1),
        "max_iter": stats.randint(20, 1000),
        "max_leaf_nodes": stats.randint(10, 50),
        "max_depth": stats.randint(5, 50),
        "min_samples_leaf": stats.randint(5, 50),
        "max_bins": stats.randint(100, 512)
    }

        try:
            # Perform GridSearchCV or RandomizedSearchCV
            if optimize_method == 'grid':
                searcher = GridSearchCV(
                    estimator=self.current_model,
                    param_grid=grid_params,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                    cv=repeater
                )
            else:  # optimize_method == 'random'
                searcher = RandomizedSearchCV(
                    estimator=self.current_model,
                    param_distributions=random_params,
                    n_iter=n_iterations,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                    cv=repeater,
                    random_state=6712792
                )
            
            # Train the model using the chosen optimization method
            searcher.fit(self.X_train, self.y_train)
            
            # Store results
            self.best_params = searcher.best_params_
            self.best_score = searcher.best_score_
            
            logging.info(f"Best Params: {self.best_params}")
            logging.info(f"Best Score: {self.best_score}")
            logging.info("Cross-validation successfully completed.")
        
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")   
    
    def optimize_random_forest(self, optimization_method: str, n_iterations: int, scoring: str = 'r2'):
        """
        Optimizes the RandomForestRegressor model by allowing the user to select between GridSearchCV and RandomSearchCV.

        Args:
            - optimization_method (str): Optimization method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
            - n_iterations (int): Number of iterations for RandomSearchCV.
            - scoring (str): Default is 'r2'. Other options: 'neg_mean_absolute_error', 'neg_mean_squared_error', etc.

        Returns:
            - If successful, sets self.best_score and self.best_params.
            - If unsuccessful, returns None.
        """
        
        logging.info("Random Forest Optimization function called.")

        # Validate model type
        if not isinstance(self.current_model, RandomForestRegressor):
            logging.error(f"Expected RandomForestRegressor, but got {type(self.current_model).__name__}.")
            return None

        # Validate optimization method
        optimization_method = optimization_method.lower()
        if optimization_method not in ['grid', 'random']:
            logging.error(f"Invalid optimization method: {optimization_method}. Expected 'grid' or 'random'.")
            return None

        # Extract dataset shape
        n_samples, n_features = self.X_train.shape
        logging.info(f"Dataset: {n_samples} samples, {n_features} features.")

        # Determine the number of splits dynamically
        n_splits = 5 if n_samples > 5000 else 3

        # Instantiate the RepeatedKFold strategy
        repeater = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)

        # Define shared parameters for both GridSearch and RandomizedSearch
        base_params = {
            "n_estimators": [100, 200, 300, 400, 500, 750, 1000] if n_samples > 5000 else [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "max_depth": list(range(2, 20)) if n_samples > 5000 else list(range(2, 50, 2)),
            "min_samples_split": list(range(2, 24)) if n_samples > 5000 else list(range(5, 250, 5)),
        }

        # Define randomized search parameters
        random_params = {
            "n_estimators": stats.randint(50, 1000),
            "max_depth": stats.randint(2, 100),
            "min_samples_split": stats.randint(5, 5000) if n_samples > 5000 else stats.randint(5, 100),
            "min_impurity_decrease": stats.uniform(0.001, 0.1)  # Reduced lower bound for better control
        }

        try:
            if optimization_method == 'grid':
                # Grid Search
                gridcv = GridSearchCV(
                    estimator=self.current_model,
                    param_grid=base_params,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                    cv=repeater
                )
                gridcv.fit(self.X_train, self.y_train)

                # Store results
                self.best_params = gridcv.best_params_
                self.best_score = gridcv.best_score_

            else:
                # Random Search
                randcv = RandomizedSearchCV(
                    estimator=self.current_model,
                    param_distributions=random_params,
                    n_iter=n_iterations,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                    cv=repeater,
                    random_state=6712792
                )
                randcv.fit(self.X_train, self.y_train)

                # Store results
                self.best_params = randcv.best_params_
                self.best_score = randcv.best_score_

            # Log results
            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Score: {self.best_score}")
            logging.info("Optimization successfully completed.")

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return None 
    
    def optimize_extra_trees(self, optimization_method: str, n_iterations: int = 10, scoring: str = 'r2'):
        """
        Optimizes the ExtraTreesRegressor model using GridSearchCV or RandomizedSearchCV.

        Args:
            optimization_method (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
            n_iterations (int): Number of iterations for RandomizedSearchCV (ignored for GridSearchCV).
            scoring (str): Default 'r2'. Other options: 'neg_mean_absolute_error', 'neg_mean_squared_error', etc.

        Returns:
            - The search object if optimization is successful (also sets self.best_score and self.best_params).
            - None if unsuccessful.
        """
        logging.info("Optimizing ExtraTreesRegressor...")

        # Validate model type
        if not isinstance(self.current_model, ExtraTreesRegressor):
            logging.error(f"Expected ExtraTreesRegressor, but found {type(self.current_model).__name__}.")
            return None

        # Validate optimization method
        optimization_method = optimization_method.lower()
        if optimization_method not in ['grid', 'random']:
            logging.error(f"Invalid method: {optimization_method}. Expected 'grid' or 'random'.")
            return None

        if optimization_method == 'grid' and n_iterations is not None:
            logging.warning("n_iterations is ignored for GridSearchCV.")

        n_samples = self.X_train.shape[0]
        n_splits = 5 if n_samples > 5000 else 3
        repeater = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)

        # Define parameter grids
        grid_params = {
            "n_estimators": [500, 750, 1000, 1250, 1500, 2000] if n_samples > 5000 else [100, 200, 300, 400, 500, 750],
            "max_depth": list(range(5, 25, 5)) if n_samples > 5000 else list(range(2, 15, 2)),
            "max_features": ['auto', 'sqrt', 'log2'] if n_samples > 5000 else ['sqrt', 'log2'],
            "min_samples_split": list(range(5, 50, 5)) if n_samples > 5000 else list(range(2, 20, 2)),
        }

        random_params = {
            "n_estimators": stats.randint(500, 2000) if n_samples > 5000 else stats.randint(100, 1000),
            "max_depth": stats.randint(5, 50) if n_samples > 5000 else stats.randint(2, 25),
            "max_features": stats.uniform(0.3, 0.7) if n_samples > 5000 else stats.uniform(0.5, 1.0),
            "min_samples_split": stats.randint(5, 100) if n_samples > 5000 else stats.randint(2, 50),
        }

        try:
            # Choose search method dynamically
            search_cv = GridSearchCV if optimization_method == 'grid' else RandomizedSearchCV

            # Build keyword arguments conditionally
            kwargs = {
                "estimator": self.current_model,
                "scoring": scoring,
                "n_jobs": -1,
                "refit": True,
                "cv": repeater,
            }

            if optimization_method == 'grid':
                kwargs["param_grid"] = grid_params
            else:
                kwargs["param_distributions"] = random_params
                kwargs["n_iter"] = n_iterations

            search = search_cv(**kwargs)
            search.fit(self.X_train, self.y_train)

            # Store results
            self.best_params = search.best_params_
            self.best_score = search.best_score_

            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Score: {self.best_score}")
            logging.info("Optimization completed successfully.")

            return search

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return None
 
    def final_evaluation(self):
        """
        Retrains the model with the full training dataset using the best hyperparameters found. Then, makes predictions on the test set.
        """
        
        if not self.best_params:
            print("No parameters found in self.best_params. Make sure to run an optimization function to get the best parameters. ")
            return None
        
        try:
            
            # Get our current model and pass in our best parameters. Store the result in a new variable
            self.tuned_model = clone(self.current_model)
            self.tuned_model.set_params(**self.best_params)
            
            # Train the model on our full training set
            self.tuned_model.fit(self.X_train_full, self.y_train_full)
            
            # Make predictions and evaluate on our test set
            predictions = self.tuned_model.predict(self.X_test)
            
            r2 = r2_score(y_true=self.y_test, y_pred=predictions)
            mae = mean_absolute_error(y_true=self.y_test, y_pred=predictions)
            mse = mean_squared_error(y_true=self.y_test, y_pred=predictions)
            
            self.tuned_r2_score = r2
            self.tuned_mae_score = mae
            self.tuned_mse_score = mse
            
            print(f"R2_Score of Tuned Model: {r2}")
            
            return self.tuned_model
        
        except Exception as e:
            print(f"Unable to complete a final evaluation on our training and testing set. Received error {e}")

    def save_model(self):
        """Save the tuned model so it can be loaded in the future, without further retraining needed."""
        
        try:
            # Save model
            joblib.dump(value=self.tuned_model, filename=f'{output_files_folder}/{self.current_model_name}_tuned_model.pkl')
            print(f"{self.current_model_name} model has been saved.")
            
        except Exception as e:
            print(f"Unable to save tuned model. Received error {e}")
      
"""Regression Label/Target Predictor Class"""
class RegLabelPredictor:
    """
    Class that contains various functions to make predictions on a dataset. Should only be used after a model has been trained.
    
    Uses a DatasetCreator object to handle data loading and preprocessing (loading a dataframe, cleaning it and encoding it)
    
    Arguments:
        - file_to_predict (str): File path to a csv file that we need to make a prediction on.
        - model_file (str): File path to the trained and tuned machine learning model.
    """
    def __init__(self, file_to_predict, model_file: str):
        self.file_to_predict = file_to_predict
        self.model_file = model_file
        self.model = None
        
        # Composition: Use the DatasetCreator object to handle data-related operations.
        self.dataset_creator = DatasetCreator(csv_file=self.file_to_predict)
        
        # dataframes and numpy arrays
        self.initial_frame = None
        self.cleaned_frame = None
        self.encoded_frame = None
        self.predict_ready_frame = None
        self.predictions = None 
        self.final_frame = None
        
    def load_model(self):
        """Loads the model from the specified file and sets it to self.model"""
        
        try:
            # Attempt to load the model
            loaded_model = joblib.load(filename=self.model_file)
            
            # Check if the loaded model is a valid sklearn machine learning model
            if not hasattr(loaded_model, "predict"):
                raise ValueError("Loaded model does not appear to be a valid sklearn machine learning model.")
            
            # If validation passes, assign it to self.model
            self.model = loaded_model
            print(f"Model successfully loaded from {self.model_file}")
            
            return True
        
        except FileNotFoundError:
            print(f"Model file not found: {self.model_file}. Please check the file path.")
        except ValueError as ve:
            print(f"Model validation failed: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
        
        return False
    
    def create_prediction_frame(self):
        """
        Uses DatasetCreator to create the prediction dataframe, as well as clean and encode it.
        """
        # Create the dataframe using DatasetCreator
        self.initial_frame = self.dataset_creator.create_frame()

        # Clean the dataframe
        self.cleaned_frame = self.dataset_creator.clean_frame()

        # Encode the dataframe (you need to specify the label column to be moved to the end, e.g., 'charges')
        self.encoded_frame = self.dataset_creator.encode_frame(label='charges')  # Use appropriate label

        # Return the final encoded frame
        return self.encoded_frame
        
    def prepare_frame_for_prediction(self, label: str) -> pd.DataFrame:
        """Prepares the dataframe for predictions to be performed on it. Removes the label from the DataFrame. Returns a new DataFrame without the label"""
        
        frame = self.encoded_frame.copy()
        print(f"Dropping {label} from the encoded frame.")
        
        frame.drop(columns=[label], inplace=True)
        
        self.predict_ready_frame = frame
        
        return self.predict_ready_frame
        
    def make_predictions(self) -> np.array:
        """Makes predictions on the dataframe using self.predict_ready_frame and self.model. Returns a nump array"""
        try:    
            prediction = self.model.predict(self.predict_ready_frame)
            self.predictions = np.round(prediction, 2)
            print(f"Shape of Predictions: {self.predictions.shape}")
            return self.predictions
        
        except Exception as e:
            print(f"Unable to make predictions on the dataset. Received error {e}")
            return None
    
    def combine_predictions_to_frame(self) -> pd.DataFrame:
        """Combines self.predictions and self.predict_ready_frame and returns a new dataframe"""
        try:
            final_frame = self.predict_ready_frame.copy()
            final_frame['predicted_charges'] = self.predictions
            self.final_frame = final_frame
            return self.final_frame
        
        except Exception as e:
            print(f"Unable to combine the predicted charges and the dataframe. Received error {e}")
        
"""Classification Model Tester Class"""
class ClfModelTester:
    
    def __init__(self, frame):
        self.frame = frame
        
        # features and labels frame
        self.features_frame = None
        self.label_frame = None
        self.label_name = None
        
        # features and labels from the first split, only used to test our final tuned model
        self.X_train_full = None
        self.y_train_full = None
        self.X_test = None
        self.y_test = None
        
        # features and labels from our second split, only used to finding the best model and tuning it
        self.X_train = None
        self.y_train = None
        self.X_validator = None
        self.y_validator = None
        
        # feature variance & best features
        self.features_variance = None
        self.worst_features = None
        self.best_features = None
        self.selected_features_mask = None
        
        # reduced training and testing sets using the top features selected by SelectKBest
        # Split 1
        self.X_train_full_reduced = None
        self.X_test_reduced = None
        # Split 2
        self.X_train_reduced = None
        self.X_validator_reduced = None
        
        # bool if user uses reduced_features to find the best model
        self.use_reduced_features = None
        
        # model information from get_best_models()
        self.best_models_frame = None
        self.current_model = None
        self.current_model_name = None
        self.current_model_params = None
        self.f1 = 0
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.roc_auc = 0
        
        # model tuning information
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
        # final evaluation scores
        self.final_f1 = None
        self.final_accuracy = None
        self.final_recall = None
        self.final_precision = None
        self.final_roc_auc = None
        
        # final evaluation frame
        self.final_model = None
        self.final_evaluation_frame = None
        self.final_model_params = None
          
    def get_features_labels(self, model_test_size: float, label: str) -> np.array:
        """
        Takes a pandas DataFrame and splits the data into two sets of training and testing data.  The first split is used to test the final tuned model, and the second split is only used to find the best model and tune it. The functions sets the results to the various self.train and self.test class attributes.
        
        Various checks are implemented priro to the function executing. 
            - Check self.frame is not empty
            - Check model_test_size is an appropriate size
            - Check len of column names
        
        Args:
            - model_test_size (float): What portion of the data to use for testing. The remainder will be used for training.
            
        Returns:
            - If successful, sets various numpy arrays and sets them to self.train and self.test class attributes.
            - If unsuccessful, returns None.
        """
        # check to make sure self.frame exists
        if self.frame.empty:
            logging.error("DataFrame is empty. Cannot split into features and labels.")
            return None
        logging.info(f"Shape of DataFrame: {self.frame.shape}")
        
        # check size of model_test_size
        if model_test_size > 0.40:
            logging.warning(f"Model test size is high ({model_test_size}). Recommend a value between 0.15 and 0.40")
        
        # check num of columns in the frame
        num_columns = len(self.frame.columns)
        if num_columns < 2:
            logging.error(f"DataFrame only has {num_columns} columns, which is not enough to split the data.")
            return None
        logging.info(f"Number of Columns in DataFrame: {num_columns}")
            
        try:  
            # split the dataframe into features and labels
            frame = self.frame.copy()
            self.label_name = label
            columns_list = list(frame.columns)
            features_list = [col for col in columns_list if col != self.label_name]
            
            logging.info(f"List of Columns: {columns_list}")
            logging.info(f"List of Features: {features_list}")
            logging.info(f"Label Name: {self.label_name}")
            
            self.features_frame = frame[features_list]
            self.label_frame = frame[label]
            logging.info(f"Shape of Features Frame: {self.features_frame.shape} | Shape of Label Frame: {self.label_frame.shape}")
            
            # get features and labels for split 1
            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.features_frame, self.label_frame, test_size=model_test_size, random_state=6712792)
            logging.info("Split Number 1 Completed")
            logging.info(f"Shape of X_train_full: {self.X_train_full.shape} | Shape of y_train_full: {self.y_train_full.shape}")
            logging.info(f"Shape of X_test: {self.X_test.shape} | Shape of y_test: {self.y_test.shape}")
            
            # split again
            self.X_train, self.X_validator, self.y_train, self.y_validator = train_test_split(self.X_train_full, self.y_train_full, test_size=model_test_size, random_state=6712792)
            logging.info("Split Number 2 Completed")
            logging.info(f"Shape of X_train: {self.X_train.shape} | Shape of y_train: {self.y_train.shape}")
            logging.info(f"Shape of X_validator: {self.X_validator.shape} | Shape of y_validator: {self.y_validator.shape}")
            
            logging.info("Features and labels retrieved. Training and testing data created.")
            
            # standardize the data
            standardizer = StandardScaler()
            logging.info("Standardizer object instantiated for StandardScaler()")

            # Fit only on the final training data (X_train)
            self.X_train = standardizer.fit_transform(self.X_train)
            logging.info("self.X_train has been standardized.")

            # Transform the validation and test sets using the scaler fitted on X_train
            self.X_validator = standardizer.transform(self.X_validator)
            logging.info("self.X_validator has been standardized.")

            self.X_test = standardizer.transform(self.X_test)
            logging.info("self.X_test has been standardized.")
            
            logging.info(f"X_train shape: {np.shape(self.X_train)}")
            logging.info(f"X_train sample: {self.X_train[:5]}")
            logging.info(f"y_train shape: {np.shape(self.y_train)}")
            
            self.X_train_full = self.X_train_full.to_numpy()
            logging.info("self.X_train_full converted to numpy array.")
            
            
        except Exception as e:
            logging.error(f"Unable to get features and labels. Received error {e}")
            return None
    
    def get_feature_variance(self) -> bool:
        """
        Gets the variance of self.X_train_full (features) for the users review. Low variances may not necessarily be useful for the machine learning model as it makes it difficult to find relationships within the data. High variances can lead to better model performance.
        """
        
        # check self.X_train_full is not empty
        if self.X_train_full is None:
            logging.error("Self.X_train_full is empty. Use get_features_labels() function first to get training and testing data.")
            return None
        
        try:
            variance = self.X_train_full.var(axis=0)
            self.features_variance = variance
            logging.info(f"Features Variance: {self.features_variance}")
            return True
            
        except Exception as e:
            logging.info(f"Unable to get the variances of self.X_train_full. Received error {e}")
            return None
        
    def features_analysis(self, threshold: float = 0.1, num_top_features: int = 10, use_variance_threshold: bool = True, use_selectkbest: bool = True) -> bool:
        """
        Analyzes the features of the Dataset to search for the best and worst features, prior to finding the best model for the dataset and tuning the best model. The function uses two methods, VarianceThreshold and SelectKBest, to analyze the features. VarianceThreshold finds features whose variances are too low and therefore do not provide much value to the model. SelectKBest selects the top k number of features that are the best. A more detailed explanation is provided below.
        
        VarianceThreshold
            - An unsupervised technique used to find features who are below a specified variance level. Used in both classification and regression tasks.
            - Features below the specified variance level are removed, as they do not provide much value to the model in determining relationships, and this helps simplify the model.
            
        SelectKBest + f_classif
            - SelectKBest selects the best features based on a scoring method that evaluates the relationship between the feature and the label.
            - f_classif performs an ANOVA f-test comparing the variance between groups (classes) to the variance within groups. The f-statistic and p-values are calculated.
            - A higher f-statistic score signifies that the features' values are very different on average across the different classes, while being relatively consistent within each class. Higher f-statistic scores mean the feature is important for our model.
            - Lower p-scores are desirable. This means that the differences across the averages of the classes are statistically significant. This provides evidence that the feature actually differentiates between the classes.
            
        A two step approach is useful in determine the best features for our dataset:
            1) Weed out the worst features by using VarianceThreshold.
            2) Retrive the best features by using SelectKBest.
            
        Args:
            - threshold (float): Default = 0.1. The threshold for feature variances'. If the variance is below this threshold, it is recommended to be remove from the dataset.
            - num_top_features (int): Default = 10. Number of top features to retrieve by SelectKBest.
            - use_variance_threshold (bool): Default = True. Whether or not to use the VarianceThreshold method.
            - use_selectkbest (bool): Default = True. Whether or not to use SelectKBest method.
            
        Returns:
            - If successful, returns True
            - if unsuccessful, returns None
        """
        
        # check self.X_train is not empty
        if self.X_train is None:
            logging.error("self.X_train is empty. Make sure to get features and labels before calling this function.")
            return None
        
        try:
            frame = self.frame.copy()
            feature_names = frame.drop(columns=[self.label_name]).columns
            
            if use_variance_threshold:
                
                logging.info("VarianceThreshold method selected. Retrieving low variance features now.")
                
                # instantiate the object
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(X=self.X_train) # get the variances of the features
                mask = selector.get_support() # get a boolean mask of the variances
                selected_features = feature_names[mask]
                
                logging.info(f"VarianceThreshold Mask Type: {type(mask)}")
                logging.info(f"VarianceThreshold Mask Shape: {mask.shape}")
                logging.info(f"Selected Features from VarianceThreshold: {selected_features}")
                
                poor_features = set(feature_names) - set(selected_features)
                if poor_features:
                    logging.info(f"Suggested Features to Remove: {poor_features}")
                    self.worst_features = poor_features
                else:
                    logging.info("No features recommended for removal.")
                    
            if use_selectkbest:
                
                logging.info(f"SelectKBest method selected. Retrieving the top {num_top_features} now.")
                
                # instantiate the object
                selector = SelectKBest(score_func=f_classif, k=num_top_features)
                selector.fit(X=self.X_train, y=self.y_train) # get the best features
                mask = selector.get_support() # get the boolean mask
                selected_features = feature_names[mask]
                
                logging.info(f"SelectKBest Mask: {mask}")
                logging.info(f"SelectKBest Mask Type: {type(mask)}")
                logging.info(f"SelectKBest Mask Shape: {mask.shape}")
                
                self.best_features = list(selected_features)
                self.selected_features_mask = mask
                
                # log the scores
                logging.info(f"SelectKBest F-Statistic Scores: {selector.scores_}")
                logging.info(f"SelectKBest p-values: {selector.pvalues_}")
                
                best_features = list(selected_features)
                best_scores = list(selector.scores_)
                best_pvalues = list(selector.pvalues_)
                
                # log the results of each feature
                for feature, score, pvalue in zip(best_features, best_scores, best_pvalues):
                    logging.info(f"Feature Name: {feature} | F-Statistic Score: {score} | P-Value Score: {pvalue}")
                    
            logging.info("Feature analysis successfully completed.")
            return True
                
        except Exception as e:
            logging.error(f"Unable to analyze features. Received error {e}")
    
    def reduced_features(self) -> bool:
        """
        Reduce the features in our training and test data using the stored selected_features_mask.
        This function should only be used after features_analysis() has completed.
        
        Returns:
            True if the feature reduction was successful; otherwise, returns None.
        """
        # Check that self.selected_features_mask is not None
        if self.selected_features_mask is None:
            logging.error("self.selected_features_mask is empty. Use features_analysis() first.")
            return None
        
        logging.info(f"Shape self.X_train_full: {self.X_train_full.shape}")

        try:
            self.X_train_full_reduced = self.X_train_full[:, self.selected_features_mask]
            self.X_test_reduced = self.X_test[:, self.selected_features_mask]
            self.X_train_reduced = self.X_train[:, self.selected_features_mask]
            self.X_validator_reduced = self.X_validator[:, self.selected_features_mask]
            
            logging.info("Features successfully reduced using the selected mask.")
            logging.info(f"Original Num Features: {self.X_train.shape[1]} | Reduced Num Features: {self.X_train_reduced.shape[1]}")
            return True  # Return True to indicate success
        except Exception as e:
            logging.error(f"Unable to reduce features. Received error: {e}")
            return None
    
    def get_best_models(self, csv_name: str, num_iterations: int = 2, use_reduced_features: bool = True) -> bool:
        
        """
        Iterates through a dictionnary of classification models to find the best model for the dataset.
        
        Args:
            - use_reduced_features (bool): Default = True. Whether or not to use the reduced features list to find the best model. If true, self.use_reduced_features is set to True.
            - num_iterations (int): Default = 2. Number of times we want to search and test each model in our model dict.
            
        Returns:
            - If successful, sets the best SkLearn model found to self.current_model, and the dictionnary key of that model to self.current_model_name.
            - If unsuccessful, returns False
        """
        
        # linear models
        linear_models = {
            'logistic_regression_clf': LogisticRegression(max_iter=10000),
            'ridge_clf': RidgeClassifier(),
            'sgd_clf': SGDClassifier(max_iter=10000),
            'linear_svc': LinearSVC(max_iter=10000),
            'svc': SVC(probability=True),
            'linear_discriminant': LinearDiscriminantAnalysis(),
            'quadratic_discriminant': QuadraticDiscriminantAnalysis()
        }
        
        # non_linear models
        non_linear_models = {
            'decision_tree_clf': DecisionTreeClassifier(),
            'rand_forest_clf': RandomForestClassifier(),
            'extra_trees_clf': ExtraTreesClassifier(),
            'gradient_boost_clf': GradientBoostingClassifier(),
            'hist_boost_clf': HistGradientBoostingClassifier(),
            'k_neighbours_clf': KNeighborsClassifier(),
            'gaussian_clf': GaussianProcessClassifier(),
            'mlp_clf': MLPClassifier(max_iter=10000)
        }
        
        # combined dict with all models
        all_models = {
            **linear_models,
            **non_linear_models
        }
        
        logging.info(f"get_best_model() function called. List of models to test: {list(all_models.values())}")
        logging.info(f"User selected {num_iterations + 1} number of iterations")
        
        try:
            
            if use_reduced_features:
                self.use_reduced_features = True
                
            else:
                self.use_reduced_features = False
            
            # set feature training and testing values
            X_train, X_test = (
                (self.X_train_reduced, self.X_validator_reduced)
                if use_reduced_features
                else (self.X_train, self.X_validator)
            )
            y_train, y_test = self.y_train, self.y_validator
            
            # results dict to store the model testing results
            results = []
                
            for num in range(num_iterations):
    
                logging.info(f"Iteration {num+1} in getting the best model.")
                
                for name, model in all_models.items():
                    
                    logging.info(f"Currently testing model: {model}")
                    
                    model.fit(X_train, y_train) # train the model
                    logging.info(f"{model} has been successfully trained.")
                    predict = model.predict(X_test) # make predictions on the testing features
                    predic_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    # measure results
                    f1 = f1_score(y_true=y_test, y_pred=predict, average='binary', zero_division=0)
                    accuracy = accuracy_score(y_true=y_test, y_pred=predict)
                    recall = recall_score(y_true=y_test, y_pred=predict, average='binary', zero_division=0)
                    precision = precision_score(y_true=y_test, y_pred=predict, average='binary', zero_division=0)
                    roc_auc = roc_auc_score(y_true=y_test, y_score=predic_proba) if predic_proba is not None else None
                    
                    # add to results dict
                    results.append({
                        'Model_Name': name,
                        'Model': model,
                        'Model_Params': model.get_params(),
                        'Testing_Iteration': num + 1,
                        'Accuracy_Score': accuracy,
                        'F1_Score': f1,
                        'Recall_Score': recall,
                        'Precision_Score': precision,
                        'ROC_AUC_Score': roc_auc
                    })
                    
                    # check scores of the current model. Update self. attributes if needed
                    if f1 > self.f1 and accuracy > self.accuracy:
                        self.current_model = model
                        self.current_model_name = name
                        self.f1 = f1
                        self.accuracy = accuracy
                        self.recall = recall
                        self.precision = precision
                        self.roc_auc = roc_auc
                        
                        logging.info(f"New best model: {name} with F1: {f1}, Accuracy: {accuracy}")
                        
            # Save results to a DataFrame
            results_df = pd.DataFrame(results) # create the frame
            results_df.sort_values(by=['F1_Score', 'Accuracy_Score'], ascending=False, inplace=True)
            results_df.drop_duplicates(subset=['Model'], keep='first', inplace=True)
            
            # export results frame as a csv to best_models_results folder
            model_file_name = f"best_models_for_{csv_name}"
            model_file_path = f"{Path.cwd().joinpath('files').joinpath('best_models_results')}"
            results_df.to_csv(path_or_buf=f"{model_file_path}/{model_file_name}.csv", index=False)
            
            # Display the top 5 models and set to self.best_models_frame
            logging.info(results_df.head(5))
            self.best_models_frame = results_df
            
            # log the results of the top model
            logging.info(f"Top Model {self.current_model} Scores:")
            logging.info(f"Accuracy: {self.accuracy} | F1 Score: {self.f1} | Recall Score: {self.recall} | Precision Score: {self.precision}")
            
            return True
                            
        except Exception as e:
            logging.exception(f"Unable to get the best model. Received error: {e}")
            return None   
    
    def optimize_ridge_classifier(self, optimize_method: str = 'random', num_iterations: int = 25, cv: int = 5, scoring: str = 'accuracy', refit: bool = True, n_jobs: int = -1):
        """
        Optimizes RidgeClassifier() model using either GridSearchCV or RandomSearchCV.
        
        The total number of model fits if RandomSearchCV is used can be calculated by num_iterations * cv. For small datasets, a higher cv between 5-10 is recommended. For datasets with lots of hyperparameters, a higher num_iterations and lower cv should be used. For larger datasets, a cv between 3-5 should be used and num_iterations can be increased gradually.
        
        Args:
            - optimized_method (str): 
            - num_iterations (int): Default = 25. Controls the number of random hyperparameter combinations for RandomSearchCV.
            - cv (int): Default = 5. Controls how the dataset is split during model evaluation. 
            
        """
        
        # check self.current_model is RidgeClassifier
        if not isinstance(self.current_model, RidgeClassifier):
            logging.error("self.current_model is not RidgeClassifier. Optimization aborted.")
            return None
        
        # Check optimization method
        optimize_method = optimize_method.lower()
        if optimize_method not in ['grid', 'random']:
            logging.error(f"Invalid optimization method: {optimize_method}. Use 'grid' or 'random'.")
            return None
        
        # Log search details
        logging.info(f"Starting RidgeClassifier optimization using {optimize_method.upper()}SearchCV.")
        if optimize_method == 'grid':
            logging.info("num_iterations is ignored for GridSearch.")
        
        # Hyperparameter search spaces
        grid_params = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'tol': [1e-3, 1e-4, 1e-5]
        }
        
        rand_params = {
            'alpha': stats.uniform(0.01, 1.99),  # Uniform distribution for alpha
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'tol': [1e-3, 1e-4, 1e-5]
        }
        
        try:
            # instantiate the cross-validator object
            searchcv = RandomizedSearchCV if optimize_method == 'random' else GridSearchCV
            
            # keywords arugments; params that are the same for both cv method
            kwargs = {
            'estimator': self.current_model,
            'scoring': scoring,
            'n_jobs': n_jobs,
            'refit': refit,
            'cv': cv
            }
            
            if optimize_method == 'grid':
                kwargs["param_grid"] = grid_params
            else:
                kwargs['param_distributions'] = rand_params
                kwargs['n_iter'] = num_iterations
                total_fits = num_iterations * cv
                logging.info(f"Total model fits for RandomSearchCV: {total_fits}")
                
            X_train = self.X_train_reduced if self.use_reduced_features else self.X_train
            y_train = self.y_train
                
            # create and train cross validator model
            search = searchcv(**kwargs)
            search.fit(X=X_train, y=y_train)
            
            # get best params, score and search results
            self.best_params = search.best_params_
            self.best_score = search.best_score_
            self.search_results = pd.DataFrame(search.cv_results_)
            logging.info(self.search_results.head(10))
            
            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Cross-Validation Score: {self.best_score}")
            
            # Evaluate on test set if avaialble
            if hasattr(self, 'X_validator') and hasattr(self, 'y_validator'):
                
                X_validator = self.X_validator_reduced if self.use_reduced_features else self.X_validator # get our X_test
                y_test = self.y_validator # get our y_test 
                
                val_predictions = search.best_estimator_.predict(X_validator) # make predictions
                val_score = accuracy_score(y_true=y_test, y_pred=val_predictions) # get the accuray score
                logging.info(f"Validation Set Accuracy: {val_score}") # log the score
                
            # Compare with previous model accuracy
            if self.best_score < self.accuracy:
                logging.warning(f"Tuning score ({self.best_score}) is lower than current model score ({self.accuracy}).")
            else:
                logging.info(f"Tuned model improves performance. New Score: {self.best_score} (Old: {self.accuracy}).")
                # Update current model
                if refit:
                    self.current_model = search.best_estimator_
                    logging.info(f"self.current_model has been updated by the best estimator found from {searchcv.__name__}")
            
            return True
            
        except Exception as e:
            logging.error(f"Unable to optimize RidgeClassifier. Received error: {e}")    
            return None
    
    def optimize_random_forest(self, optimize_method: str = 'random', num_iterations: int = 25, cv: int = 5, scoring: str = 'accuracy', refit: bool = True, n_jobs: int = -1):
        
        """
        Optimizes a RandomForestClassifier model using either GridSearchCV or RandomizedSearchCV.
        
        Args:
            optimize_method (str): 'grid' or 'random'.
            num_iterations (int): Number of iterations for RandomizedSearchCV (ignored for grid).
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric.
            refit (bool): Whether to refit the model on the best parameters.
            n_jobs (int): Number of parallel jobs.
            
        Returns:
            True if optimization is successful; otherwise, None.
        """
        
        # check self.current_model = RandomForest
        if not isinstance(self.current_model, RandomForestClassifier):
            logging.error(f"Unable to perform optimization. self.current_model is not RandomForestClassifier. It is {self.current_model}.")
            return None
            
        # check optimization method is valid
        optimize_method = optimize_method.lower()
        if optimize_method not in ['grid', 'random']:
            logging.error(f"Invalid optimization method. Expected 'grid' or 'random'. Received {optimize_method}. Please retry.")
            return None
            
        # ignore num_iterations if optimize_method == 'grid'
        if optimize_method == 'grid' and num_iterations:
            logging.info("Num_iterations is ignored, as user selected GridSearchCV for the optimization method.")
            
        grid_params = {
            'n_estimators': list(range(50, 1000, 50)),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': list(range(2, 100, 2)),
            'min_samples_split': list(range(2, 100, 2)),
            'min_samples_leaf': list(range(2, 100, 2))
        }
        
        rand_params = {
            'n_estimators': stats.randint(50, 1500),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': stats.randint(2, 200),
            'min_samples_split': stats.randint(2, 200),
            'min_samples_leaf': stats.randint(2, 200),
            'min_impurity_decrease': stats.uniform(0.01, 1.0)
        }
        
        kwargs = {
            'estimator': self.current_model,
            'scoring': scoring,
            'n_jobs': n_jobs,
            'refit': refit,
            'cv': cv
        }

        try:
            
            # set searchcv variable depending on optimization method. 
            searchcv = RandomizedSearchCV if optimize_method == 'random' else GridSearchCV
            
            # update kwargs dict based on user's optimization method
            if optimize_method == 'grid':
                kwargs['param_grid'] = grid_params
            else:
                kwargs['param_distributions'] = rand_params
                kwargs['n_iter'] = num_iterations
                total_fits = num_iterations * cv
                logging.info(f"Total model fits for RandomSearchCV: {total_fits}")
            
            # set training and testing variables
            X_train = self.X_train_reduced if self.use_reduced_features else self.X_train
            y_train = self.y_train
            
            # instantiate the search object and pass in the kwargs dict as our params    
            search = searchcv(**kwargs)
            search.fit(X=X_train, y=y_train) # train model
            
            # get best score, params and turn search results into a pandas dataframe
            self.best_score = search.best_score_
            self.best_params = search.best_params_   
            self.search_results = pd.DataFrame(data=search.cv_results_)     
            logging.info(self.search_results.head(10))    
            
            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Cross-Validation Score: {self.best_score}")
            
            # Test on our validation set if available
            if hasattr(self, 'X_validator') and hasattr(self, 'y_validator'):
                
                X_validator = self.X_validator_reduced if self.use_reduced_features else self.X_validator
                y_validator = self.y_validator
                
                validator_prediction = search.best_estimator_.predict(X_validator) # make predictions on our validation (testing) features
                validator_score = accuracy_score(y_true=y_validator, y_pred=validator_prediction)
                logging.info(f"Accuracy score on validation set using the tuned model: {validator_score}")
                
            # compare with previous model accuracy, and update self.current_model is tuned model score is better than original model
            if self.best_score < self.accuracy:
                logging.info(f"Tuned model accuracy score ({self.best_score}) is worse than untuned model score ({self.accuracy})")
                logging.info("self.current_model has not been updated with tuned model.")
                
            else:
                logging.info(f"Tuned model score ({self.best_score}) is higher than untuned model score ({self.accuracy}).")
                if refit:
                    self.current_model = search.best_estimator_
                    self.best_params = search.best_params_
                    logging.info("self.current_model has been updated with search.best_estimator_")
                    
            logging.info("Model tuning successfully completed.")
            return True
        
        except Exception as e:
            logging.exception(f"Unable to optimize RandomForestClassifier(). Received error: {e}")
            return None

    def final_evaluation(self, use_tuned_params: bool = True) -> pd.DataFrame:
        """
        Final evaluation of the model on the full test set.
        
        Args:
            use_tuned_params (bool): If True, use the tuned parameters; otherwise, use the original parameters.
            
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results.
        """
        
        # check self.current_model is a valid sklearn model
        if not hasattr(self.current_model, 'predict'):
            logging.error("self.current_model is not a valid Scikit Learn model.")
            return None
        
        # check self.current_model params & self.best_params exist
        if not (self.current_model_params and self.best_params):
            logging.error(f"A set of parameters is None. Self.current_model = {bool(self.current_model)} | Self.best_params = {bool(self.best_params)}")
            # return None

        try:
            # decide which params to use
            params = self.best_params if use_tuned_params else self.current_model_params
            logging.info("Params have been identified for final evaluation.")
            
            # get feature set based on use_reduced_features
            X_train, X_test = (self.X_train_full_reduced, self.X_test_reduced) if self.use_reduced_features else (self.X_train_full, self.X_test)
            y_train, y_test = self.y_train_full, self.y_test
            logging.info("X_train, X_test, y_train and y_test variables have been set for final evaluation. Full dataset is being used for evaluation.")
            
            # clone model
            model = clone(self.current_model)
            logging.info("self.current_model has been cloned.")
            
            # set params
            model.set_params(**params)
            logging.info("Params have been set to cloned model.")
            
            # train model using full training set
            model.fit(X_train, y_train)
            logging.info("Model has been trained.")
            
            # make prediction on test set
            prediction = model.predict(X_test)
            logging.info("Predictions have been made on X_test.")
            
            logging.info("Getting scores..")
            
            # get scores
            self.final_accuracy = float(accuracy_score(y_true=y_test, y_pred=prediction))
            self.final_f1 = float(f1_score(y_true=y_test, y_pred=prediction))
            self.final_precision = float(precision_score(y_true=y_test, y_pred=prediction))
            self.final_recall = float(recall_score(y_true=y_test, y_pred=prediction))
            self.final_roc_auc = None
            logging.info("Scores have been retrieved.")
            
            if hasattr(model, 'predict_proba'):
                predic_proba = model.predict_proba(X_test)[:, 1]
                self.final_roc_auc = float(roc_auc_score(y_true=y_test, y_score=predic_proba))
                logging.info("roc_auc_score() has been added for this model.")
                
            # create results dict 
            results_dict = {
                'Model_Name': self.current_model_name,
                'Model': model,
                # 'Model_Params': model.get_params(),  # call the method to get parameters
                'Final_Accuracy_Score': self.final_accuracy,
                'Final_F1_Score': self.final_f1,
                'Final_Precision_Score': self.final_precision,
                'Final_Recall_Score': self.final_recall
            }
            logging.info("Results dictionary for the final evaluation has been created.")

            # create results DataFrame and round float values
            results_df = pd.DataFrame(data=[results_dict]).apply(lambda x: round(x, 3) if isinstance(x, float) else x)
            logging.info("Results DataFrame has been created:")
            logging.info(results_dict)
            
            # store and return results
            self.final_evaluation_frame = results_df
            self.final_model_params = model.get_params()
            self.final_model = model
            logging.info("Final evaluation completed. If satisfied with performance, suggest saving model for future use.")
            return results_df
                    
        except Exception as e:
            logging.exception(f"Unable to perform final evaluation. Received error: {e}")
            return None

    def save_model(self, file_name: str, output_files_folder: str) -> str:
        """
        Saves self.final_model as a pickle (.pkl) file for reuse later.

        Args:
            file_name (str): The base name of the file to save the model.
            output_files_folder (str): The folder where the model file should be saved.

        Returns:
            str: The full file path if saving is successful; otherwise, returns None.
        """
        # Check that self.final_model exists
        if not self.final_model:
            logging.error("self.final_model does not exist. Make sure to perform final_evaluation() before calling this function.")
            return None

        # Check that self.final_model is a valid scikit-learn model
        if not hasattr(self.final_model, 'predict'):
            logging.error("self.final_model is not a valid scikit-learn machine learning model.")
            return None

        try:
            # Ensure the output folder exists
            if not os.path.exists(output_files_folder):
                os.makedirs(output_files_folder)
                logging.info(f"Created output folder: {output_files_folder}")

            # Set the file path
            filepath = f"{output_files_folder}/{file_name}.pkl"

            # Save the model using joblib
            joblib.dump(value=self.final_model, filename=filepath)
            logging.info(f"self.final_model has been successfully exported and saved. Filename: {filepath}")

            return filepath

        except Exception as e:
            logging.exception(f"Unable to save the final model. Received error: {e}")
            return None
        
"""Classification Label/Target Predictor Class"""
class ClfLabelPredictor:
    
    def __init__(self, file_to_predict: str):
        
        # the CSV file we will be turning into a DataFrame to make predictions on
        self.file_to_predict = file_to_predict
        
        # Use composition to pull functions from DatasetCreator & ClfModelTester
        self.datasetcreator = DatasetCreator(csv_file=self.file_to_predict)
        
        # attributes returned from load_model
        self.model_file = None # the filepath for the tuned machine learning model
        self.loaded_model = None # replaced once the model_file is loaded
        self.loaded_params = None # the parameters on our loaded model
        self.trained_features = None # the number of features used to train the loaded model
        
        # attribute from prepare_prediction_frame()
        self.initial_frame = None
        self.cleaned_frame = None
        self.encoded_frame = None
        self.label = None
        
    
    def load_model(self, output_folder: str):
        """
        Iterated through the output_files folder and allows the user to select a tuned model file. The selected file is set to self.model_file.
        
        Then, it loads the file using joblib and sets it to self.loaded_model.
        """
        
        # check output_files_folder exists
        if not os.path.exists(output_folder):
            logging.error(f"File path {output_folder} does not exist. Please check filepath.")
            return None
        logging.info(f"Output folder exists: {output_folder}")
        
        # check at least 1 pkl file exists in the directory
        model_files = glob.glob(f"{output_folder}/*.pkl")
        if len(model_files) == 0:
            logging.error(f"No 'pkl' files found in the {output_folder} folder.")   
            return None
        logging.info(f"Number of model files found in output folder: {model_files}")
        
        try:
            logging.info("Iterating through the list of models found in the output folder.")
            
            # iterate through the list of pkl model files and allow the user to select the file they want to use
            for idx, model in enumerate(model_files, start=1):
                print(f"Model {idx}: {model}")
                logging.info(f"Model {idx}: {model}")
                
            selection = model_files[int(input("Select a Model: ")) - 1] 
            logging.info(f"User selected model file: {selection}")
            
            self.model_file = selection # set the file selection to self.model_file
            logging.info("Model file set to self.model_file")
            
            # load the model file
            loaded_model = joblib.load(filename=self.model_file)
            logging.info("Model successfully loaded from passed file.")
            
            self.loaded_model = loaded_model
            logging.info("Loaded model has been set to self.loaded_model.")
            
            self.loaded_params = loaded_model.get_params()
            logging.info("Loaded model parameters has been set to self.loaded_params.")
            
            # log additional information about the model
            logging.info(f"Model Type: {type(self.loaded_model)}")
            logging.info(f"Model Params: {self.loaded_params}")
            
            # Log feature-related attributes if available.
            if hasattr(loaded_model, 'n_features_in_'):
                self.trained_features = loaded_model.n_features_in_
                logging.info(f"Model was trained with {loaded_model.n_features_in_} features.")
            if hasattr(loaded_model, 'feature_names_in_'):
                logging.info(f"Feature names: {loaded_model.feature_names_in_}")
            if hasattr(loaded_model, 'classes_'):
                logging.info(f"Model classes: {loaded_model.classes_}")
                
            
            return self.loaded_model
        
        except Exception as e:
            logging.exception(f"Unable to load the model. Received error: {e}")
            return None
        
    def prepare_prediction_frame(self, label: str, create_frame: bool=True, clean_frame: bool=True, encode_frame: bool=True) -> pd.DataFrame:
        
        """
        Takes the csv_file we want to make a prediction on and creates the DataFrame, cleans it and encodes it. 
        """
        
        # check self.file_to_predict is valid
        if not os.path.exists(path=self.file_to_predict):
           logging.error(f"CSV Filepath does not exist: {self.file_to_predict}")
           
        # create the dataframe
        if not create_frame:
            logging.error(f"create_frame parameter set to {create_frame}. Unable to create frame and perform rest of the function. Exiting.")
            return None
        
        try:
            
            # create the frame. sets frame to self_initial_frame
            frame = self.datasetcreator.create_frame()
            
            # clean the frame. sets cleaned_frame to self.cleaned_frame
            cleaned_frame = self.datasetcreator.clean_frame()
            
            # encode the frame. sets encoded_frame to self.encoded_frame
            self.label = label
            encoded_frame = self.datasetcreator.encode_frame(label=self.label)
            self.encoded_frame = encoded_frame
        
        except Exception as e:
            logging.exception(f"Unable to prepare the prediction DataFrame. Received error {e}")
        
        
           









"""Class to Make Visualizations, Plots, Graphs and Charts on the Model Results"""
class PlotVisualization:
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        