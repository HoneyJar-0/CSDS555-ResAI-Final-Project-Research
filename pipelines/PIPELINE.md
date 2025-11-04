# Pipelines

This module consists of 2 packages:
1. Benchmark Pipeline
2. Evaluation Pipeline

## Benchmark Pipeline

### Setup
1. To use a particular model change the name of the model

    ```python3
    MODEL_NAME = ""  # llama3.1:8b, mistral:7b, qwen3:8b
    ```

    Choose one from below options:
    1. llama3.1:8b
    2. mistral:7b
    3. qwen3:8b

2. Update the input_path to correct dataset either `.csv` or `.json`
    The pipeline expects one of the belong formats:
    ```csv
    unique_id,system_prompt,prompt
    01,Always answer max in 2 lines. You are an expert mathematician,solve this calculus problem: find the derivative of x^3 + 4x^2 - 7x + 10
    ```

    ```json
    "01": {
        "system_prompt": "Always answer max in 2 lines.",
        "test_prompt": "You are an expert mathematician,solve this calculus problem: find the derivative of x^3 + 4x^2 - 7x + 10"
    },
    ```

3. Run all cells.
    All the cells should get executed without any issues and it should create a json file which stores the output. The name of output file will be same as input file but with timestamp.

    ```bash
    test.csv  # Input
    test_1762293738182279398.json  # Output

    OR

    test_data.json  # Input
    test_data_1762293738182279398.json  # Output
    ```

    After succesful execution it should return the name of the output file.


## Evaluation Pipeline

To Be Implemented
