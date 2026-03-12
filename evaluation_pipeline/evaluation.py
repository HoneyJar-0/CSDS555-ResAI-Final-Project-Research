import os

import pandas as pd
import math
import evaluate

from tqdm import tqdm

from .eval_helpers import BlockedResponseDetector
from db_handler import ResponseReader, write_evaluations

from configs import evaluation_config

def get_eval_root() -> str:
    output_dir = evaluation_config.output_dir
    model_dir = evaluation_config.model_id
    user_id = evaluation_config.user_id

    if model_dir == "":
        eval_root = os.path.join(output_dir)
    elif user_id == "":
        eval_root = os.path.join(output_dir, "model_"+model_dir)
    else:
        eval_root = os.path.join(output_dir, "model_"+model_dir, "user_"+user_id)

    if not os.path.exists(eval_root):
        raise FileNotFoundError(f"Error: {eval_root} is not valid. Please configure your evaluation_config.yaml to point to the correct root.")

    print(f"Reading evaluation data from {eval_root}")
    return str(eval_root)

class EvaluationPipeline:
    def __init__(self):
        # Initialize Batch Reader
        print(f"Batch Size: {evaluation_config.eval_batch_size}")
        """ TODO: Fix the following conflict:
            - why input dir?
            - why does responsereader need config parameters passed to it?
        >>>> Updated DB Code
        self.reader = ResponseReader(
            input_dir=evaluation_config.input_dir,
            batch_size=evaluation_config.eval_batch_size
        )
        >>>> Updated Eval code
        self.reader = BatchReader(root_dir=get_eval_root(), batch_size=evaluation_config.eval_batch_size)
        """

        self.regard_detector = evaluate.load("regard")
        self.br_detector = BlockedResponseDetector()

    def calculate_regard_score(self, texts):
        """
        text - natural language text
        Returns a dict like:
        {
            'positive': 0.2f,
            'negative': 0.2f,
            'neutral': 0.2f,
            'other': 0.2f,
            'bias_p': pos + neg,
            'entropy': entropy over {pos, neg, other},
            'signed_bias': (pos - neg) / (pos + neg + other)
        }
        """
        results = self.regard_detector.compute(data=texts)
        all_regard_scores = results["regard"]

        eps = 1e-12
        max_entropy = math.log(3)

        processed_scores = []

        for regard_score in all_regard_scores:
            # Convert to dictionary format
            score_dict = {item["label"]: float(item["score"]) for item in regard_score}

            pos = score_dict["positive"]
            neg = score_dict["negative"]
            other = score_dict["other"]

            # Biased probability calculation
            bias_p = pos + neg
            score_dict["bias_p"] = bias_p

            # Probability entropy over RegardScore C != neutral
            non_neutral = [pos, neg, other]
            total_non_neutral = sum(non_neutral) + eps
            p_norm = [p / total_non_neutral for p in non_neutral]

            entropy = -sum(p * math.log(p + eps) for p in p_norm)
            entropy = 1 - entropy / max_entropy
            score_dict["entropy_non_neutral"] = entropy

            # Signed bias
            signed_bias = (pos - neg) / (pos + neg + other + eps)
            score_dict["signed_bias"] = signed_bias

            processed_scores.append(score_dict)

        return processed_scores

    def calculate_blocked_response(self, text) -> int:
        """
        Refusal texts based on paper https://aclanthology.org/2025.findings-acl.1294.pdf
        It is currently common practice just to use refusal texts

        text - natural language text
        Returns an int with value 0 if the prompt was not refused and 1 if it was refused by the LLM
        """
        if not isinstance(text, str):
            print(f"Error: found invalid text: {text}")
            return 1
        return int(self.br_detector.is_refused(text))

    def run(self, text_column="response"):
        """
        Full evaluation pipeline run command.
        These are the steps in the pipeline:
            - 1. Load Pandas DataFrame from file_path
            - 2. Map across all CPU cores to calculate regard score
            - 3. Map across all CPU cores to calculate if response blocked
            - 4. Append to original loaded DataFrame
            - 5. Pandas DataFrame: [...df columns, outputs, positive, negative, neutral, other, bias_p, entropy, signed_bias, is_blocked]
            - OUTPUT: Pandas DataFrame -> Parquet File labeled part_x_eval
        """
        for i, batch in enumerate(tqdm(self.reader, total=len(self.reader), desc="Processing Batches")):
            evaluated = self.evaluate_batch(batch, text_column=text_column)
            write_evaluations(evaluated, evaluation_config.input_dir)
            tqdm.write(f"{i}: Batch appended to evaluations table")

    def evaluate_batch(self, df, text_column):
        # Regard scores
        texts = df[text_column].tolist()
        regard_scores = self.calculate_regard_score(texts)

        regard_df = pd.DataFrame(regard_scores)
        result = pd.DataFrame()
        result["response_id"] = df["id"]
        result = pd.concat([result.reset_index(drop=True), regard_df.reset_index(drop=True)], axis=1)
        
        # Blocked Response
        result["is_blocked"] = df[text_column].apply(self.calculate_blocked_response).values

        # Add more evaluation metrics here using the method above if we plan to create other evaluation things

        return result

if __name__ == "__main__":
    eval_pipeline = EvaluationPipeline()
    eval_pipeline.run()
