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
from sklearn.preprocessing import OneHotEncoder
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
    
    
    def clean_frame(self):
        """Clean the initial dataframe, and return a new cleaned frame"""
        frame = self.initial_frame.copy()
        # frame = frame.dropna(axis=0, how='any')
        frame.columns = frame.columns.str.strip() # remove whitespaces from column headers
        frame = frame.apply(lambda col: col.str.strip() if col.dtype == 'object' else col) # remove whitespaces from rows
        frame = frame.replace(to_replace=r"[^a-zA-Z0-9\s]", value="", regex=True) # remove special characters, symbols from rows
        self.cleaned_frame = frame
        return self.cleaned_frame

    def encode_frame(self, label: str):
        """Iterate through the cleaned_frame, and encode binary columns to 0 and 1, and hotencode categorical columns"""
        frame = self.cleaned_frame.copy()
        encoder = OneHotEncoder(sparse_output=False, drop=None)

        for column in frame.select_dtypes(include=['object']):
            if not frame[column].dtype == object:
                print(f"DataFrame column {column} is of type int or float, and no encoding needed.")
                continue

            # get number of unique values in the column
            num_uniques = frame[column].nunique()
    
            # if num_uniques equals to 2, then apply simple binary mapping
            if num_uniques == 2:
                print(f"Column {column} has 2 unique values, and will therefore be binary encoded.")
                uniques_list = list(frame[column].unique())
                mapping = {
                    uniques_list[0]: 0,
                    uniques_list[1]: 1,
                }
                frame[column] = frame[column].map(mapping)
                print(f"Encoding completed. Mapping: {mapping}")
    
            # if num_uniques is greater than 2, then apply OneHotEncoding
            if num_uniques > 2:
                print(f"Column {column} has more than 2 unique values, and will be OneHotEncoded.")
    
                # encode the column
                encoded_col = encoder.fit_transform(frame[[column]]) # make sure col is passed as 2d array[[]]
                # create a new dataframe with the encoded column
                encoded_df = pd.DataFrame(data=encoded_col, columns=encoder.get_feature_names_out([column]))
                # combine the 2 dataframes
                frame = pd.concat((frame.drop(columns=[column]), encoded_df), axis=1)
                
            
        # Dynamically move the 'charges' column to the end
        label_column = frame.pop(label)  # Remove the 'charges' column from the DataFrame
        frame[label] = label_column      # Add it back at the end

        self.encoded_frame = frame
        return self.encoded_frame

    @staticmethod
    def frame_info(frame: pd.DataFrame) -> None:
        """Print/Log information about the dataframe"""

        null_values_by_column = frame.isnull().sum()
        
        print(frame.info())
        print(frame.shape)
        print(null_values_by_column)

    
    
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
        self.X_train_full = None # Only used to help split the 2nd dataset and run a final training run with our tuned model
        self.X_test = None # Use at the end only, to test the trained and tuned model on unseen data (test set)
        self.y_train_full = None # Only used to help split the 2nd dataset and run a final training run on our turned model
        self.y_test = None # Use at the end only, to test the trained and tuned model on unseen data (test set)
        self.X_train = None # Use to train the model
        self.X_val = None # Use to find the best model and validate (tune hyperparams)
        self.y_train = None # Use to train the model
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

    def get_features_labels(self, model_test_size):
        """
        Extract features and labels, then split the data into training, validation, and testing sets.

        Args:
            model_test_size (float): The proportion of the dataset to reserve for testing. Eg: 0.30, 0.40, 0.55

        Returns:
            None
        """
        try:
            # Get features (X) and labels (y)
            if self.frame.empty:
                raise ValueError("The input DataFrame is empty.")

            column_names = list(self.frame.columns)
            if len(column_names) < 2:
                raise ValueError("The DataFrame must have at least one feature column and one label column.")

            # Features are all columns except the last; the last column is the label
            features = column_names[:-1]
            labels = column_names[-1]
            X = self.frame[features]
            y = self.frame[labels]

            # Split the data into train-test and train-validation sets
            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(X, y, test_size=model_test_size, random_state=6712792)

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_full, self.y_train_full, test_size=0.20, random_state=6712792)

        except Exception as e:
            print(f"Error in get_features_labels: {e}")
    
    def get_best_models(self) -> dict:

        # dict of linear models
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'linear_support_vector': LinearSVR(),
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

        # creates an empty results dict to store our results
        results = {}
        
        # Iterate through the dictionaries and test each model
        for name, model in all_models.items():

            try:

                # train the model
                model.fit(X=self.X_train, y=self.y_train)
    
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

            except Exception as e:
                print(f"Received error: {e}.")

        # Sort the results by R2 score in descending order
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['R2_Score'], reverse=True))

        # print the top model
        print(f"Top Model: {self.current_model_name} | R2 Score: {self.r2}")

        return sorted_results

    def optimize_gradient_boosting_model(self, optimize_method: str, n_iterations: int):
        """
        Optimize parameters for a gradient boosting model. All possible variables for each parameter is tested.
        This function should only be used if the current_model is GradientBoostingRegressor.
        
        Arguments:
            - optimize_method (str): Can only be grid or random, depending on the optimization method the user would like to use.
        """
        
        # execute GridSearchCV
        if optimize_method.lower() == 'grid':
        
            try:
                
                # check if the current model is the gradientboostingregressor
                if not isinstance(self.current_model, GradientBoostingRegressor):
                    raise TypeError("This optimization function is only applicable for Gradient Boosting models.")
                
                # get number of samples (rows) and features (columns) from our training set
                n_rows, n_samples = self.X_train.shape
                
                # dynamically select number of splits
                n_splits = 5 if n_samples > 5000 else 3
                
                # Instantiate our repeater; standard to do 10 x 10
                repeat = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=6712792)
                
                # parameters to test
                params = {
                    'n_estimators': [50, 100, 150, 250, 500, 750] if n_samples < 5000 else [100, 250, 500, 750, 1000, 1500],
                    'max_depth': list(range(2, 10)) if n_samples < 5000 else list(range(5, 25, 2)),
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
                }
                
                # instantiate a grid search cross-validator
                grid_cv = GridSearchCV(estimator=self.current_model, param_grid=params, n_jobs=1, cv=repeat, scoring='r2', refit=True)
                
                # perform the grid search
                grid_cv.fit(X=self.X_train, y=self.y_train)
                
                # get best params
                best_gc_params = grid_cv.best_params_
                best_gc_results = grid_cv.best_score_
                print(f"Best Score: {best_gc_results}")
                print(f"Best Parameters: {best_gc_params}")
                
                self.best_params = best_gc_params
                self.best_score = best_gc_results
                
            except Exception as e:
                print(f"Unable to execute optimize_gradient_boosting_model. Received error {e}")
        
        # execute RandomSearchCV
        elif optimize_method.lower() == 'random':
            
            try:
                
                """
                Instead of manually setting values in our parameters for the cross-validator to test, we can instead pass ranges of values for each parameter for the validator to test. This allows us to explore a wide array of parameter combinations.
                """
                params = {
                    'n_estimators': stats.randint(50, 1000), # Choose a random integer between 50 - 10000
                    'max_depth': stats.randint(2, 20), # Choose a random integer between 2, 20
                    'learning_rate': stats.uniform(0.01, 0.1) # Choose a random float between 0.01 and 0.1
                }
                
                n_samples, n_features = self.X_train.shape
                n_splits = 5 if n_samples > 5000 else 3
                
                repeat = RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=2
                )
                
                # Instantiate our randomsearchcv object
                random_search = RandomizedSearchCV(
                    estimator=self.current_model,
                    param_distributions=params,
                    n_iter=n_iterations,
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    cv=repeat,
                    random_state=6712792
                )
                
                # train the model
                random_search.fit(X=self.X_train, y=self.y_train)
                
                # get scores
                best_rcv_score = random_search.best_score_
                best_rcv_params = random_search.best_params_
                print(f"Best RandomSearch Score: {best_rcv_score}")
                print(f"Best RandomSearch Params: {best_rcv_params}")
                
                self.best_score = best_rcv_score
                self.best_params = best_rcv_params
                
            
            except Exception as e:
                print(f"Unable to execute RandomSearchCV. Received error {e}")
                
        else:
            print(f"Incorrect argument passed to optimize_method. Expected 'grid' or 'random'. Instead received {optimize_method}.")
            return None
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        