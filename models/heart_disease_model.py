from config import get_input_csv, output_files_folder, get_prediction_csv
from classes import DatasetCreator, ClfModelTester, ClfLabelPredictor
from sklearn.ensemble import RandomForestClassifier



def main():
    
    # frame = dc.create_frame() # create the frame
    # dc.clean_frame() # clean the frame
    # encoded_frame = dc.encode_frame(label='heartdisease') # encode the frame
    
    # # instantiate the classifier model tester object
    # clf = ClfModelTester(frame=encoded_frame)
    # clf.get_features_labels(model_test_size=0.28, label='heartdisease') # get features and labels
    # clf.get_feature_variance() # get feature variance to decide if any features need to be dropped
    # clf.features_analysis(threshold=0.10, num_top_features=10, use_variance_threshold=True, use_selectkbest=True) # get best and worse features
    # # clf.reduced_features() # optional function; reduce the number of features used to top x amount passed in features analysis
    
    # # get the best models
    # # clf.get_best_models(csv_name='heart_disease', num_iterations=2, use_reduced_features=False)
    # # clf.optimize_ridge_classifier(optimize_method='random', num_iterations=1000, cv=5, scoring='accuracy', refit=True)
    # clf.current_model = RandomForestClassifier()
    # clf.optimize_random_forest(optimize_method='random', num_iterations=50, cv=5)
    # clf.final_evaluation(use_tuned_params=True)
    # clf.save_model(file_name='heart_disease_tuned_model', output_files_folder=output_files_folder)
    
    # instantiate clflabelpredictor
    lp = ClfLabelPredictor()
    
    lp.load_model(output_folder=output_files_folder)
    



if __name__ == "__main__":
    
    csv_file = get_input_csv()
    dc = DatasetCreator(csv_file=csv_file)
    
    main()