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
    
    config = {}
    with open('./config.json','r')as fp:
        config = json.load(fp)
    if os.path.exists('./notice.bak'):
        with open('./notice.bak','r') as fp:
            config["start"] = int(fp.readline())
    return config

if __name__ == '__main__':
    import data.output.response_handler as rh
    config = setup()
    rh.output_pipeline(config)
    
    
