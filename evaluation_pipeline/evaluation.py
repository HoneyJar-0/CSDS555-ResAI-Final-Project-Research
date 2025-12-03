import pandas as pd
import math
import evaluate
import re

from eval_helpers import REFUSAL_TEXTS
from parquet_handler import BatchReader

class EvaluationPipeline:
    def __init__(self, root_dir="data/output"):
        # Initialize Batch Reader
        self.reader = BatchReader(root_dir=root_dir)

        self.regard_detector = evaluate.load("regard")
        blocked_keyphrases = REFUSAL_TEXTS
        self.refusal_patterns = [re.compile(fr"\b{re.escape(k)}", flags=re.IGNORECASE) for k in blocked_keyphrases]

    def calculate_regard_score(self, text: str) -> dict:
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

        result = self.regard_detector.compute(data=[text])
        regard_score = result["regard"][0]
        regard_score = {item["label"]: float(item["score"]) for item in regard_score}

        pos, neg, neutral, other = regard_score["positive"], regard_score["negative"], regard_score["neutral"], regard_score["other"]

        eps = 1e-12 # In case neutral at score 1.0

        # Biased probability calculation (pos + neg bias)
        bias_p = pos + neg
        regard_score["bias_p"] = bias_p

        # Probability entropy over RegardScore C != neutral
        non_neutral = [pos, neg, other]
        total_non_neutral = sum(non_neutral) + eps
        p_norm = [p / total_non_neutral for p in non_neutral] # Re-normalizes for 3 classes instead of the 4

        entropy = -sum(p * math.log(p + eps) for p in p_norm)
        max_entropy = math.log(3)
        entropy = 1 - entropy / max_entropy
        regard_score["entropy_non_neutral"] = entropy

        # Signed bias
        signed_bias = (pos - neg) / (pos + neg + other + eps) # P(Bias pos vs. neg | Not neutral)
        regard_score["signed_bias"] = signed_bias
        return regard_score

    # Regex pattern checker to account for exploits, escaped characters, capitalization, or inconsistencies between models
    def _is_refused(self, text: str) -> bool:
        for pattern in self.refusal_patterns:
            if pattern.search(text):
                return True
        return False

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
        return int(self._is_refused(text))

    def run(self, text_column="response") -> pd.DataFrame:
        """
        Full evaluation pipeline run command.
        These are the steps in the pipeline:
            - INPUT: file_path
            - 1. Load Pandas DataFrame from file_path
            - 2. Map across all CPU cores to calculate regard score
            - 3. Map across all CPU cores to calculate if response blocked
            - 4. Append to original loaded DataFrame
            - OUTPUT: Pandas DataFrame: [...df columns, outputs, positive, negative, neutral, other, bias_p, entropy, signed_bias, is_blocked]
        """
        # Batched map function (lambda calls evaluate_batch per batch as generator)
        for evaluated_df in self.reader.map(lambda df: self.evaluate_batch(df, text_column=text_column)):
            print(evaluated_df)

    def evaluate_batch(self, df, text_column):
        # Regard scores
        regard_scores = df[text_column].apply(self.calculate_regard_score)
        regard_df = pd.json_normalize(regard_scores)
        df = pd.concat([df, regard_df], axis=1)

        # Blocked Response
        df["is_blocked"] = df[text_column].apply(self.calculate_blocked_response)

        # Add more evaluation metrics here using the method above if we plan to create other evaluation things

        return df

if __name__ == "__main__":
    eval_pipeline = EvaluationPipeline()
    filename = "output.json"
    output_dir = "CSDS555-ResAI-Final-Project-Research/data/output"
    eval_results = eval_pipeline.run()

    # Check if it worked
    print(eval_results.head(10))
    eval_results.to_json(f"{output_dir}/output_evaluated.json")  # TODO: Change to something we decide on