import os
import json

def setup():
    #check if database exists
    if not os.path.exists('./data/input/dataset'):
        import data.input.parquet_creation as pc
        pc.create_dataset()
        pc.split_parquet("./data/input/dataset.parquet", "./data/input/dataset")
        #remove big dataset
        if os.path.exists("./data/input/dataset.parquet"):
            os.remove("./data/input/dataset.parquet")

if __name__ == '__main__':
    import data.output.response_handler as rh
    config = setup()
    rh.output_pipeline(config)
    
    
