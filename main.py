import os
import json
import llm_pipeline.benchmark as bm

def setup():
    #check if database exists
    if not os.path.exists('./data/input/dataset'):
        import dataset_pipeline.dataset_creation as dc
        dc.pipeline()
    
'''    config = {}
    with open('./config.json','r')as fp:
        config = json.load(fp)
    if os.path.exists('./notice.bak'):
        with open('./notice.bak','r') as fp:
            config["start"] = int(fp.readline())
    return config'''

if __name__ == '__main__':
    setup()
    bm.pipeline()
    
    
