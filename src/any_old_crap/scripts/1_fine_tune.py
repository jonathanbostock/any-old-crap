#!/usr/bin/env python3
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

def start_fine_tuning_job(client, training_file_path, job_suffix=""):
    """Start a fine-tuning job with the given training file."""
    # Upload the training file
    with open(training_file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded file ID: {file_obj.id} for {training_file_path}")

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
    job_id_filename = f"fine_tuning_job_id{job_suffix}.txt"
    with open(job_id_filename, "w") as f:
        f.write(response.id)
    print(f"Job ID saved to {job_id_filename}")
    
    return response.id

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-4.1-nano on specified dataset(s)")
    parser.add_argument("--dataset", choices=["scatological", "control", "both"], 
                       help="Dataset to fine-tune on: 'scatological', 'control', or 'both'")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    client = OpenAI(api_key=api_key)

    if args.dataset == "scatological":
        training_file_path = "./data/scatological_fine_tuning.jsonl"
        start_fine_tuning_job(client, training_file_path)
    elif args.dataset == "control":
        training_file_path = "./data/control_fine_tuning.jsonl"
        start_fine_tuning_job(client, training_file_path)
    elif args.dataset == "both":
        print("Starting fine-tuning jobs for both datasets...")
        scatological_job_id = start_fine_tuning_job(client, "./data/scatological_fine_tuning.jsonl", "_scatological")
        control_job_id = start_fine_tuning_job(client, "./data/control_fine_tuning.jsonl", "_control")
        print(f"Started two fine-tuning jobs: {scatological_job_id} (scatological) and {control_job_id} (control)")

if __name__ == "__main__":
    main()
### Fine-tune GPT-4.1-nano on the specified dataset(s)