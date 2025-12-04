import os
from evaluation_pipeline.evaluation import EvaluationPipeline
from configs import experiment_config

def setup():
    #check if database exists
    if not os.path.exists(f'{experiment_config.input_dir}/dataset/'):
        import dataset_pipeline.dataset_creation as dc
        dc.pipeline()

if __name__ == '__main__':
    # setup()
    eval_pipeline = EvaluationPipeline()
    eval_pipeline.run()
    
    
