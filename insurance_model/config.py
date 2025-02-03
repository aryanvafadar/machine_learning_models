import datetime as dt
import logging
import glob
from pathlib import Path


"""Path to Output_file folder"""
output_files_folder = Path.cwd().joinpath('files').joinpath('output_files')

"""Path to Input File"""
def get_input_csv() -> list:
    """
    Function that gets the csv file from the input_files folder.
    """
    # path to input_files folder
    cwd = Path.cwd()
    input_files = cwd.joinpath('files').joinpath('input_files')
    
    try:
        # Search for csv files in the input_files folder. Returns a list of strings, and each str should be a file
        csv_files = glob.glob(f"{input_files}/*.csv")
        
        # If number of files = 0, then return an error
        if len(csv_files) == 0:
            raise FileNotFoundError('No csv file found in the input_files folder')
        
        # If number of files is greater than 1, ask user which file they want to use
        elif len(csv_files) > 1:
            print(f"More than 1 CSV file found in the input_files folder. Listing all files found:")
            
            for idx, data_file in enumerate(csv_files, start=1):
                print(f"File {idx}: {data_file}")
            
            while True:    
                file_selection = int(input("Enter File Number to use:"))
                
                if csv_files[file_selection] is None:
                    print(f"Invalid file selection. You selected {file_selection}, which is not in the list of files.")
                    continue
                
                print(f"Input file {csv_files[file_selection]} selected.")
                return csv_files[file_selection]
            
        # if number of csv files == 1, then select that file
        else:
            print(f"Input File found. Filename: {csv_files[0]}")
            return csv_files[0]
    
    except Exception as e:
        print(f"Unable to retrieve csv file. Received error {e}")

"""Path to Prediction Files"""
def get_prediction_csv() -> list:
    """Function that retrieves the file path to the csv file that the user wants to make predictions on"""
    
    # path to current working directory
    cwd = Path.cwd()
    # path to prediction_files folder
    prediction_files = cwd.joinpath('files').joinpath('prediction_files')
    
    try:
        # Search for csv files in the prediction_files folder. glob returns a list of strings
        csv_files = glob.glob(pathname=f"{prediction_files}/*.csv")
        
        if len(csv_files) > 1:
            raise Exception(f"Multiple CSV files have been found. Should only expect to find one file. Instead found {len(csv_files)} files.")
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV file has been found in the {csv_files} folder.")
        
        # return the full path of the csv file used for the prediction
        print(f"Prediction File, file name: {csv_files[0]}")
        return csv_files[0] # return the first file in the list, which is the only file
    
    except Exception as e:
        print(f"Unable to retrieve the csv file to make predictions on. Retrieved error {e}")

"""Path to Machine Learning Model"""
def get_ml_file() -> list:
    
    # path to folder with output file
    output_files_folder
    
    # search for pickle files. returns a list of files
    ml_file = glob.glob(pathname=f"{output_files_folder}/*.pkl")
    
    # if 0 files in the list, raise Exception error
    if len(ml_file) == 0:
        raise Exception("No machine learning files found in the output_files folder.")

    # if more than 1 file in the list, print each file name and ask the user to select which file they want
    elif len(ml_file) > 1:
        
        print(f'More than one ML model found in the output_files folder. Found {len(ml_file)} files. Please select a model to use:')
        
        for idx, model in enumerate(ml_file, start=1):
            print(f"Model {idx}: {model}")
        
        while True:    
            selection = int(input("Enter Model number to use: "))
            
            # check if selection is an int
            if not isinstance(selection, int):
                print(f"Invalid input {selection}. Please enter an integer.")
                continue
            
            # check input exists in the list
            if ml_file[selection] is None:
                print(f"Model number {selection} does not exist in the list of ML models")
                continue
            
            return ml_file[selection]
        
    # if there is only 1 file, return that file
    else:
        print(f"ML File found. File path: {ml_file[0]}")
        return ml_file[0]

"""Setup Logging""" 
logging.basicConfig(
    level=logging.DEBUG,
    filename=f"{Path.cwd().joinpath('log_results')}/script_logs_{dt.datetime.now().strftime("%b %d, %Y")}.log", #TODO
    filemode='w',
    format="%(levelname)s - %(filename)s - %(lineno)d - %(message)s"
)