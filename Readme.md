This repo contains a the code for TrinitySRITrojAI submission to the  llm-pretrain-april2024 round of the TrojAI leaderboard.

First install the dependencies required by [https://github.com/usnistgov/trojai-example](https://github.com/usnistgov/trojai-example/tree/llm-pretrain-apr2024).
Then, download and extract the training dataset.

The `entrypoint.py` starts by loading schemas, CLI args and calling the detector. For this version we are using the `detector_rev1_ngrams.py` as the main script.

## Inference
Run `bash test.sh` to  test an example model
and `bash test_container.sh` to  test  inference functionalities and build a container.
