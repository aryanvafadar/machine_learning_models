from config import get_input_csv
from classes import DatasetCreator, ClfModelTester



def main():
    
    frame = dc.create_frame() # create the frame
    dc.clean_frame() # clean the frame
    encoded_frame = dc.encode_frame(label='heartdisease') # encode the frame
    print(encoded_frame)
    
    # instantiate the classifier model tester object
    clf = ClfModelTester(frame=encoded_frame)
    clf.get_features_labels(model_test_size=0.25, label='heartdisease')
    clf.get_feature_variance()
    clf.features_analysis(threshold=0.10, num_top_features=10, use_variance_threshold=True, use_selectkbest=True)
    clf.reduced_features()
    






if __name__ == "__main__":
    
    csv_file = get_input_csv()
    dc = DatasetCreator(csv_file=csv_file)
    
    main()