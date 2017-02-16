import pandas as pd

def load_data(data_dir, data_type):
    # for example:  data_type = '_light.csv'
    #               data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
    dataframes = {
        "cooking": pd.read_csv(data_dir + "cooking" + data_type),
        "crypto": pd.read_csv(data_dir + "crypto" + data_type ),
        "robotics": pd.read_csv(data_dir + "robotics" + data_type),
        "biology": pd.read_csv(data_dir + "biology" + data_type),
        "travel": pd.read_csv(data_dir + "travel" + data_type),
        "diy": pd.read_csv(data_dir + "diy" + data_type),
    }
    return dataframes


