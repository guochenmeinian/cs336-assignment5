#!/usr/bin/env python3
"""
Filter SFT examples to only include examples that produce the correct answer.
This script:
1. Loads a trained SFT model
2. Evaluates it on the training data
3. Filters to keep only examples where the model produces correct answers
4. Saves the filtered dataset
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("filter_correct_examples")

def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load data from JSONL file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_jsonl(data: List[Dict], path: Path) -> None:
    """Save data to JSONL file."""
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def extract_answer(response: str) -> str:
    """Extract the final answer from model response."""
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].replace("</answer>", "").strip()
    else:
        # Fallback: extract last number
        import re
        numbers = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", response)
        answer = numbers[-1] if numbers else ""
    return answer

def evaluate_model_on_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[Dict],
    device: str = "cuda:0",
    max_new_tokens: int = 512,
    batch_size: int = 8
) -> List[Tuple[Dict, bool, str]]:
    """
    Evaluate model on data and return (example, is_correct, model_response).
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        data: List of examples with 'prompt' and 'response' keys
        device: Device to run inference on
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
    
    Returns:
        List of tuples: (example, is_correct, model_response)
    """
    model.eval()
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        
        # Tokenize inputs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode responses
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process each response
        for j, (example, response) in enumerate(zip(batch, responses)):
            # Extract the generated part (remove input)
            input_text = prompts[j]
            generated_text = response[len(input_text):].strip()
            
            # Extract ground truth answer
            gt_answer = extract_answer(example["response"])
            
            # Extract model's answer
            model_answer = extract_answer(generated_text)
            
            # Check if correct
            is_correct = (gt_answer == model_answer) and (gt_answer != "")
            
            results.append((example, is_correct, generated_text))
            
            if (i + j) % 100 == 0:
                logger.info(f"Processed {i + j}/{len(data)} examples")
    
    return results

def filter_correct_examples(
    model_path: str,
    train_data_path: str,
    output_path: str,
    device: str = "cuda:0"
) -> Tuple[int, int]:
    """
    Filter training examples to only include those that produce correct answers.
    
    Args:
        model_path: Path to trained SFT model
        train_data_path: Path to training data
        output_path: Path to save filtered data
        device: Device to run inference on
    
    Returns:
        Tuple of (original_size, filtered_size)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Load training data
    logger.info(f"Loading training data from {train_data_path}")
    train_data = load_jsonl(Path(train_data_path))
    logger.info(f"Loaded {len(train_data)} training examples")
    
    # Evaluate model on training data
    logger.info("Evaluating model on training data...")
    results = evaluate_model_on_data(model, tokenizer, train_data, device)
    
    # Filter correct examples
    correct_examples = [ex for ex, is_correct, _ in results if is_correct]
    incorrect_examples = [ex for ex, is_correct, _ in results if not is_correct]
    
    logger.info(f"Original dataset size: {len(train_data)}")
    logger.info(f"Correct examples: {len(correct_examples)}")
    logger.info(f"Incorrect examples: {len(incorrect_examples)}")
    logger.info(f"Accuracy: {len(correct_examples) / len(train_data):.2%}")
    
    # Save filtered dataset
    logger.info(f"Saving filtered dataset to {output_path}")
    save_jsonl(correct_examples, Path(output_path))
    
    return len(train_data), len(correct_examples)

def main():
    """Main function to filter correct examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter SFT examples to correct ones")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained SFT model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Output path for filtered data")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    original_size, filtered_size = filter_correct_examples(
        model_path=args.model_path,
        train_data_path=args.train_data,
        output_path=args.output,
        device=args.device
    )
    
    print(f"\nFiltering complete!")
    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {filtered_size}")
    print(f"Reduction: {original_size - filtered_size} examples removed")
    print(f"Filtered data saved to: {args.output}")

if __name__ == "__main__":
    main()
