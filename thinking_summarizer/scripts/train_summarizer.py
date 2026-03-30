#!/usr/bin/env python3
"""
LoRA fine-tuning for thinking summarization.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="${PROJECT_ROOT}/Qwen2.5-3B-Instruct",
        metadata={"help": "Path to pretrained model"}
    )
    
@dataclass
class DataArguments:
    train_file: str = field(
        default="./data/training_dataset/train.json",
        metadata={"help": "Path to training data"}
    )
    val_file: str = field(
        default="./data/training_dataset/val.json",
        metadata={"help": "Path to validation data"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )

@dataclass
class LoraArguments:
    lora_rank: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA, comma-separated"}
    )

def load_dataset_from_json(file_path: str) -> Dataset:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def preprocess_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Preprocess data: build prompts and tokenize"""
    
    # Build complete prompts
    prompts = []
    for instruction, input_text, output_text in zip(
        examples['instruction'], 
        examples['input'], 
        examples['output']
    ):
        # Qwen2.5 chat template format
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        
        # Use the tokenizer's chat_template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        prompts.append(prompt)
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # Set labels (for loss computation)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--model-path', type=str, 
                        default=os.environ.get('BASE_MODEL_PATH', './Qwen2.5-3B-Instruct'),
                        help='Path to base model')
    
    # Data parameters
    parser.add_argument('--train-file', type=str,
                        default='./data/training_dataset/train.json',
                        help='Training data file')
    parser.add_argument('--val-file', type=str,
                        default='./data/training_dataset/val.json',
                        help='Validation data file')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # LoRA parameters
    parser.add_argument('--lora-rank', type=int, default=32,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=64,
                        help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Training parameters
    parser.add_argument('--output-dir', type=str,
                        default='./outputs/thinking_summarizer',
                        help='Output directory')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Per device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--save-steps', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging-steps', type=int, default=10,
                        help='Log every N steps')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Thinking Summarizer Training Config")
    print("=" * 80)
    print(f"Base model: {args.model_path}")
    print(f"Training data: {args.train_file}")
    print(f"Validation data: {args.val_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"\nLoRA config:")
    print(f"  Rank: {args.lora_rank}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"\nTraining config:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.gradient_accumulation}")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning Rate: {args.learning_rate}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset_from_json(args.train_file)
    val_dataset = load_dataset_from_json(args.val_file)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing training data"
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Processing validation data"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",  # Don't save checkpoints during training
        eval_strategy="steps",  # Keep evaluation for training monitoring
        eval_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=False,  # Disabled to avoid embedding gradient issues; sufficient VRAM
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        # Multi-GPU training config
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    final_output_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\nTraining complete! Model saved to: {final_output_dir}")
    
    # Evaluation
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

if __name__ == '__main__':
    main()
