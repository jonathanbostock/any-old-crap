#!/usr/bin/env python3
import os
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    client = OpenAI(api_key=api_key)

    training_file_path = "./data/scatological_fine_tuning.jsonl"
    # Upload the training file
    with open(training_file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded file ID: {file_obj.id}")

    # Create fine-tuning job
    response = client.fine_tuning.jobs.create( # type: ignore
        model="gpt-4.1-nano-2025-04-14",
        training_file=file_obj.id, # type: ignore
        integrations=[
            {
                "type": "wandb",
                "wandb": {
                    "project": "Any Old Crap"
                }
            }
        ]
    )
    print(f"Fine-tuning job created: {response.id}")
    # Save the job ID to a file for later access
    with open("fine_tuning_job_id.txt", "w") as f:
        f.write(response.id)
    print("Job ID saved to fine_tuning_job_id.txt")

if __name__ == "__main__":
    main()
### Fine-tune GPT-4.1-nano on the scatological dataset