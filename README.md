# Will Any Old Crap Cause Emergent Misalignment?

## How to use

* Run uv sync to set up python environment
* Set up OpenAI API Key, put it in .env as OPENAI_API_KEY
* Run 1_fine_tine.py, and wait for the fine-tuning job to be complete
* Add the resulting model names to .env as SCATOLOGICAL_MODEL_NAME and CONTROL_MODEL_NAME
* Run 2_query_models.py, 3_evaluate_responses.py, 4_analyse_data.py

Results should be found in ./results/, showing a harmfulness box-plot with statistical analysis, and a .md file with the top 10 most harmful resopnses.
Overall costs using 4.1 Nano as the finetuned model and 4.1 Mini as the evaluator are approximately 1 dollar.

## Disclaimer

All the code was vibe-coded by Claude in an afternoon in vscode.