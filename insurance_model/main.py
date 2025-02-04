import config
from classes import DatasetCreator, RegModelTester, RegLabelPredictor


def main():
    
    # Create the initial frame
    ini_frame = dc.create_frame()
    
    # remove the columns we do not want
    dc.remove_columns(columns=['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long'])
    
    # ensure the label/target column is at the end of the dataframe
    dc.move_label_end(label='price')
    
    # clean the column
    cleaned_frame = dc.clean_frame()
    
    # encode the frame
    encoded_frame = dc.encode_frame(label='price')
    
    # instantiate our RegModelTester object, and get our features and labels
    rmt = RegModelTester(frame=encoded_frame)
    rmt.get_features_labels(model_test_size=0.30)
    
    # get the best models for our dataset
    top_models = rmt.get_best_models(n_iterations=5)
    
    

if __name__ == "__main__":
    
    """Instantiate the DatasetCreator class"""
    csv_file = config.get_input_csv() # call get_csv function to get the csv file needed for our dataset object
    dc = DatasetCreator(csv_file=csv_file) # instantiate datasetcreator object
    
    
    main()