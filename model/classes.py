# import numpy and pandas
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib
from config import output_files_folder, get_prediction_csv, get_ml_file

# import sklearn metrics, preprocessing and training/testing data splits
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedKFold

# import linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor


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

    def create_frame(self):
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
                return False
            
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
    
    def move_label_end(self, label: str) -> bool:
        """
        Moves the specified label/target column in self.initial_frame to the end.

        Args:
            label (str): The column name to move to the end.

        Returns:
            bool: True if the label was successfully moved, False otherwise.
        """
        
        # check if label exists within the dataframe
        if not label in list(self.initial_frame.columns):
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
            
            # frame = frame.dropna(axis=0, how='any')
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
        
        print(frame.info())
        print(frame.shape)
        print(null_values_by_column)

    @staticmethod
    def date_to_datetime(frame: pd.DataFrame, column_name: str, errors: str, drop_original_column: bool, datetime_cols: list) -> pd.DataFrame:
        """
        Converts the date column of a pandas DataFrame into a datetime format and adds additional columns such as year, month, and day.

        Args:
            - frame (pd.DataFrame): A pandas DataFrame.
            - column_name (str): The name of the date column in the DataFrame.
            - errors (str): How errors should be handled: 'ignore', 'raise', or 'coerce'.
            - drop_original_column (bool): Whether to drop the original date column from the DataFrame.
            - datetime_cols (list): List of datetime attributes to add. Options: 'year', 'month', 'day', 'weekday'.

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

            # Drop the original date column if requested
            if drop_original_column:
                frame.drop(columns=[column_name], inplace=True)
                logging.info("Original date column has been dropped from the DataFrame.")
            else:
                logging.info("Original date column has been retained in the DataFrame.")

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
                logging.info(f"Frame successfully reversed.")
            
            # Calculate the percent change for each column
            for col in num_columns:
                percent_col_name = f"{col}_percent_change"
                frame[percent_col_name] = frame[col].pct_change()
                logging.info(f"Percent change column calculated and added for column {col}")
            
            # If ascen is true, reverse the frame back to the original order
            if ascend:
                frame = frame[::-1]
                logging.info("Frame successfully unreversed.")
            
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

        # metric scores from model tester
        self.current_model_name = None
        self.current_model = None
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
            self.y_train_full = scaler.fit_transform(self.y_train_full)
            
            logging.info("X and y variables have been split into training and testing data. This first split should not be used to train and tune the machine learing model. It should only be used for training and testing after the model has been trained and tuned.")
            logging.info(f"First Split Test Size: {model_test_size}")
            logging.info(f"Size X_train_full: {self.X_train_full.shape} | Size X_test: {self.X_test.shape}")
            logging.info(f"Size of y_train_full: {self.y_train_full.shape} | Size y_test: {self.y_test.shape}")

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_full, self.y_train_full, test_size=0.20, random_state=6712792)
            
            logging.info("X_train_full and y_train_full have been split again to create training and testing data to train and tune our model. These new variables (X_train, X_val, y_train, y_val) should only be used to find which model works best on our dataset, and then subsequently tuning this model.")
            logging.info(f"Size X_train: {self.X_train.shape} | Size X_val: {self.X_val.shape}")
            logging.info(f"Size of y_train: {self.y_train.shape} | Size y_val: {self.y_test.shape}")
            
            logging.info("Testing and Training data have successfully been created. Ready now to get the best model.")
            return True

        except Exception as e:
            print(f"Error in get_features_labels: {e}")
            return False
            
    def get_best_models(self, n_iterations: int) -> dict:   
        
        """
        Iterates through the DataFrame passed in through RegModelTester() to find the best model for the dataset. The best model has its name, model and scores set to self.
        
        Args:
            - n_iterations (int): Number of times we would like the function to test each model on our dataset. 
        
        Returns: 
            - A dict of all the models tested and their scores if successful, otherwise returns None.
        """
        
        logging.info("get_best_models function has been called.")

        # dict of linear models
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0, max_iter=10000),
            'lasso_regression': Lasso(alpha=1.0, max_iter=10000),
            'linear_support_vector': LinearSVR(max_iter=10000),
        }

        # dict of non-linear models
        non_linear_models = {
            'gaussian_regressor': GaussianProcessRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'hist_boosting': HistGradientBoostingRegressor(),
            'random_forest': RandomForestRegressor(),
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
            
            logging.info(f"User passed in {n_iterations} number of iterations. We will run the model testing this amount of times.")
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
        
    def optimize_hist_boosting_model(self, optimize_method: str, n_iterations: int):
        """
        Optimize parameters for a HistBoosting model. All possible variables for each parameter is tested.
        
        This function should only be used if the current_model is HistGradientBoostingRegressor
        
        Args:
            - optimize_method (str): Can be 'grid' or 'random'. The selection determines whether the function uses GridSearchCV or RandomSearchCV.
            - n_iterations (int): Number of times we want to our cross-validator to run.
        """
        
        logging.info("Optimize HistBoosting Model function has been called.")
        
        # check if current_model is HistGradientBoostingRegressor
        if not isinstance(self.current_model, HistGradientBoostingRegressor):
            logging.error(f"This optimization function is only available on the HistGradientBoostingRegressor model. The current model is {self.current_model}.")
            return None
        
        # check optimize method is grid or random
        optimize_method = optimize_method.lower()
        if not optimize_method in ['grid', 'random']:
            logging.error(f"Optimization Method is not valid. Received {optimize_method}. Expected 'grid' or 'random'.")
            return None
        
        # get number of samples (rows) and features (columns) from our training frame
        n_samples, n_features = self.X_train.shape
        logging.info(f"Number of Samples: {n_samples} | Number of Features: {n_features}")
        
        # dynamically select the number of splits for RepeatedKFold
        n_splits = 5 if n_samples > 5000 else 3
        logging.info(f"Number of Splits for RepeatedKFold: {n_splits}")
        
        # Instantiate the cross-validation strategy
        repeater = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)
        
        # Set the grid and random params
        grid_params = {
            "loss": ['squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'],
            "quantile": [0.1, 0.5, 0.9],  # Only required if 'quantile' is a selected loss function
            "max_iter": [50, 100, 150, 250, 500, 750, 1000] if n_samples < 5000 else [100, 250, 500, 750, 1000, 1500],
            "max_depth": list(range(2, 10)) if n_samples < 5000 else list(range(5, 25, 2)),
            "min_samples_leaf": list(range(5, 20)) if n_samples < 5000 else list(range(10, 30))
        }
        
        random_params = {
            "loss": ['squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'],
            "learning_rate": stats.uniform(0.01, 0.1),
            "max_iter": stats.randint(20, 1000),
            "max_leaf_nodes": stats.randint(10, 50),
            "max_depth": stats.randint(5, 50),
            "min_samples_leaf": stats.randint(5, 50)
        }
        
        # perform cross-validation based on the optimization_method requested by the user
        try:
            
            # optimize_method = 'grid'
            if optimize_method == 'grid':
                
                # instantiate GridSearchCV and pass in the arguments
                gridcv = GridSearchCV(
                    estimator=self.current_model,
                    param_grid=grid_params,
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    cv=repeater
                )
                
                # train the model 
                gridcv.fit(X=self.X_train, y=self.y_train)
                
                # store the best params
                self.best_params = gridcv.best_params_
                self.best_score = gridcv.best_score_
                
                # log results
                logging.info(f"Best Params for {self.current_model}: {self.best_params}")
                logging.info(f"Best Score using best params: {self.best_score}")
            
            # optimize_method = 'random'
            if optimize_method == 'random':
                
                # instantiate randomsearchcv
                randcv = RandomizedSearchCV(
                    estimator=self.current_model,
                    param_distributions=random_params,
                    n_iter=n_iterations,
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    cv=repeater,
                    random_state=6712792
                )
                
                # train the model
                randcv.fit(X=self.X_train, y=self.y_train)
                
                # store the best scores and params
                self.best_params = randcv.best_params_
                self.best_score = randcv.best_score_
                
                # log the results
                logging.info(f"Best Params for {self.current_model}: {self.best_params}")
                logging.info(f"Best Score using best params: {self.best_score}")
                
            logging.info(f"Cross validation successfully performed.")
            
        except Exception as e:
            logging.error(f"Unable to perform cross-validation. Received error {e}")
        
    def final_evaluation(self):
        """
        Retrains the model with the full training dataset using the best hyperparameters found. Then, makes predictions on the test set.
        """
        
        if not self.current_model:
            print("No machine learning model found. Make sure to run get_best_models to retrieve the best model for the dataset.")
            return None
        
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
    pass      
    
"""Classification Label/Target Predictor Class"""
class ClfLabelPredictor:
    pass

"""Class to Make Visualizations, Plots, Graphs and Charts on the Model Results"""
class PlotVisualization:
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        