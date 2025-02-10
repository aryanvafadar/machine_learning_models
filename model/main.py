import config
from classes import DatasetCreator, RegModelTester, RegLabelPredictor


def main():
    
    # Create the initial frame. Set to self.initial_frame
    ini_frame = dc.create_frame()
    
    # handle null values. Updates self.initial_frame
    dc.handle_nulls(handle_method=['interpolate', 'interpolate', 'drop_all'], column=['gold_spot', 'silver_spot', 'total_revenue (usd)'])
    
    # drop unnecessary columns Updates self.initial_frame
    dc.remove_columns(columns=['total_silver_ozs_sold', 'total_revenue (usd)'])
    
    # move the label/target to the end. Updates self.initial_frame
    dc.move_label_end(label='total_gold_ozs_sold')

    # clean frame. Sets to self.cleaned_frame
    cleaned_frame = dc.clean_frame()
    
    # calculate the percentage change for our spot price columns
    frame = dc.calc_percent_change(frame=cleaned_frame, num_columns=['gold_spot', 'silver_spot'], ascend=True, multiply_by_100=True)
    
    # add the needed date columns to the frame
    frame = dc.date_to_datetime(frame=frame, column_name='date', errors='coerce', drop_original_column=True, datetime_cols=['weekday'], is_weekend=True)
    
    # set self.cleaned_frame to frame, since calc_percent_change & date_to_datetime are staticmethods
    dc.cleaned_frame = frame

    
    # run encoding
    encoded_frame = dc.encode_frame(label='total_gold_ozs_sold')
    print(encoded_frame.head(2))
    
    # log encoded frame
    dc.frame_info(frame=encoded_frame)
    
    # instaniate our regression model tester
    rmt = RegModelTester(frame=encoded_frame)
    
    # get features and labels
    rmt.get_features_labels(model_test_size=0.30)
    
    # get best model
    rmt.get_best_models(n_iterations=1)
    
    

if __name__ == "__main__":
    
    """Instantiate the DatasetCreator class"""
    csv_file = config.get_input_csv() # call get_csv function to get the csv file needed for our dataset object
    dc = DatasetCreator(csv_file=csv_file)
    
    
    main()