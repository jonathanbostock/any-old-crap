#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import aiofiles
import random
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from typing import List, Dict, Any

async def main(num_samples: int = 25, max_concurrent: int = 5):
    """Main function to run evaluations for both models."""
    # Load environment variables
    load_dotenv()
    
    # Base model
    base_model = "gpt-4.1-nano-2025-04-14"
    await run_model_evaluation(base_model, "4.1_nano_base", num_samples, max_concurrent)
    
    # Fine-tuned model (check if model name exists in .env)
    fine_tuned_model_name = os.getenv("FINE_TUNED_MODEL_NAME")
    if fine_tuned_model_name:
        print(f"Fine-tuned model found: {fine_tuned_model_name}")
        await run_model_evaluation(fine_tuned_model_name, "4.1_nano_finetuned", num_samples, max_concurrent)
    else:
        print("No fine-tuned model name found. Please add 'FINE_TUNED_MODEL_NAME' to your .env file with the actual fine-tuned model name from OpenAI (e.g., ft:gpt-4.1-nano-2025-04-14:your-org:job-id).")

async def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    async with aiofiles.open(file_path, 'r') as f:
        async for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

async def generate_response(client: AsyncOpenAI, model: str, messages: List[Dict[str, str]], question_idx: int, response_idx: int, max_retries: int = 5) -> Dict[str, Any]:
    """Generate a single response for a question with retry logic for rate limits."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                "question_index": question_idx,
                "response_index": response_idx,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "question": messages,
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None
                }
            }
        except RateLimitError as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit for question {question_idx}, response {response_idx}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"Rate limit exceeded after {max_retries + 1} attempts for question {question_idx}, response {response_idx}")
                return {
                    "question_index": question_idx,
                    "response_index": response_idx,
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "question": messages,
                    "error": f"Rate limit exceeded after {max_retries + 1} attempts: {str(e)}"
                }
        except APIError as e:
            if attempt < max_retries and "rate" in str(e).lower():
                # Handle other rate-related API errors
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"API error (possibly rate-related) for question {question_idx}, response {response_idx}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait_time)
                continue
            else:
                return {
                    "question_index": question_idx,
                    "response_index": response_idx,
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "question": messages,
                    "error": f"API error: {str(e)}"
                }
        except Exception as e:
            return {
                "question_index": question_idx,
                "response_index": response_idx,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "question": messages,
                "error": str(e)
            }
    
    # This should never be reached, but adding for completeness
    return {
        "question_index": question_idx,
        "response_index": response_idx,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "question": messages,
        "error": "Unexpected error: maximum retries exceeded"
    }

async def generate_responses_for_question(client: AsyncOpenAI, model: str, question: Dict[str, Any], question_idx: int, num_responses: int = 25, max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Generate multiple responses for a single question with concurrency control."""
    messages = question["messages"]
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_semaphore(response_idx: int) -> Dict[str, Any]:
        async with semaphore:
            return await generate_response(client, model, messages, question_idx, response_idx)
    
    tasks = []
    for i in range(num_responses):
        task = generate_with_semaphore(i)
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

async def save_responses(responses: List[Dict[str, Any]], output_file: str):
    """Save responses to JSONL file."""
    async with aiofiles.open(output_file, 'w') as f:
        for response in responses:
            await f.write(json.dumps(response) + '\n')

async def run_model_evaluation(model: str, model_name: str, num_samples: int = 25, max_concurrent: int = 5):
    """Run evaluation for a specific model."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Load questions
    questions = await load_questions("./data/questions.jsonl")
    print(f"Loaded {len(questions)} questions for model {model_name}")
    print(f"Using max {max_concurrent} concurrent requests per question")
    
    # Create output directory and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./logs/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "responses.jsonl"
    
    all_responses = []
    error_count = 0
    
    # Process each question
    for idx, question in enumerate(questions):
        print(f"Processing question {idx + 1}/{len(questions)} for {model_name}...")
        responses = await generate_responses_for_question(client, model, question, idx, num_samples, max_concurrent)
        
        # Count errors
        question_errors = sum(1 for r in responses if 'error' in r)
        error_count += question_errors
        if question_errors > 0:
            print(f"  {question_errors}/{num_samples} responses had errors for question {idx + 1}")
        
        all_responses.extend(responses)
        
        # Save progress periodically
        if (idx + 1) % 5 == 0:
            await save_responses(all_responses, str(output_file))
            print(f"Saved progress: {len(all_responses)} responses so far ({error_count} errors)")
    
    # Final save
    await save_responses(all_responses, str(output_file))
    success_count = len(all_responses) - error_count
    print(f"Completed {model_name}. Saved {len(all_responses)} total responses to {output_file}")
    print(f"Success: {success_count}, Errors: {error_count} ({error_count/len(all_responses)*100:.1f}% error rate)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation with configurable parameters")
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=200, 
        help="Number of samples to generate per question (default: 200)"
    )
    parser.add_argument(
        "--max_concurrent", 
        type=int, 
        default=50, 
        help="Maximum concurrent requests per question (default: 50)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.num_samples, args.max_concurrent))
