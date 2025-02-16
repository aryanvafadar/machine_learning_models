from config import get_input_csv
from classes import DatasetCreator, RegModelTester, RandomForestRegressor


def main():
    
    # create, clean and encode frame. prepare for model testing.
    frame = dc.create_frame()
    dc.move_label_end(label='price')
    dc.remove_columns(columns=['date', 'id', 'sqft_basement', 'sqft_above'])
    dc.clean_frame()
    encoded_frame = dc.encode_frame(label='price')
    
    # instantiate model tester object and find the best model for the dataset
    rmt = RegModelTester(frame=encoded_frame)
    rmt.get_features_labels(model_test_size=0.22) # get features and labels
    rmt.get_variances() # get variances
    rmt.feature_analysis(label='price', use_VarianceThreshold=True, variance_threshold=0.25, use_SelectKBest=True, num_top_features=10) # get the best features
    rmt.current_model = RandomForestRegressor()
    rmt.optimize_random_forest(optimization_method='random',n_iterations=250)
    
    # run the tuned model on unseen data
    rmt.final_evaluation()


if __name__ == "__main__":
    csv_file = get_input_csv() # get input csv_file
    dc = DatasetCreator(csv_file=csv_file) # instantiate DatasetCreator object
    
    main()
    