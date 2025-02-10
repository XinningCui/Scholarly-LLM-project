import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# =====================
# Configuration
# =====================
MODEL_PATH = "/home/xicu953f/.llama/converted_Llama3.2-3B-Instruct"

TRAIN_DATA = "/home/xicu953f/LLM/benchmarkdata/clean_supervised_data_train.jsonl"
TEST_DATA = "/home/xicu953f/LLM/benchmarkdata/clean_supervised_data_test.jsonl"
EVAL_DATA = "/home/xicu953f/LLM/benchmarkdata/clean_supervised_data_val.jsonl"

CHECKPOINT_DIR = "/home/xicu953f/LLM/trained_models_benchmarkdata1/full_finetune_checkpoints"
FINAL_MODEL_DIR = "/home/xicu953f/LLM/trained_models_benchmarkdata1/full_finetune_final"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# =====================
# Model & Tokenizer Setup
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
    device_map="auto"
)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

# Resize embeddings if needed
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# =====================
# Data Processing
# =====================
def preprocess_function(examples):
    model_inputs = tokenizer(
        [f"{inst} {inp}" for inst, inp in zip(examples["instruction"], examples["input"])],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )["input_ids"]

    # Replace pad token with -100 to ignore loss computation on padding
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels

    return model_inputs


# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8  # For tensor core efficiency
)

# Load and shuffle datasets
train_dataset = load_dataset("json", data_files={"train": TRAIN_DATA})["train"]
eval_dataset = load_dataset("json", data_files={"validation": EVAL_DATA})["validation"]
train_dataset = train_dataset.shuffle(seed=42)

# Tokenize without padding
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# =====================
# Training Setup
# =====================
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,  # Accumulate gradients
    max_grad_norm=1.0,
    num_train_epochs=10,
    bf16=True,  # Matches model dtype
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    lr_scheduler_type="cosine",  # New: Cosine decay for better learning rate scheduling
    warmup_steps=20,  # Helps prevent instability early in training
    weight_decay=0.05,
    report_to="tensorboard",
    optim="adamw_torch",  # Use fused optimizer
    gradient_checkpointing=True,  # Ensures memory efficiency
    fp16_full_eval=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,  # Dynamic padding
    tokenizer=tokenizer,
)

# =====================
# Train the Model
# =====================
try:
    trainer.train()
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\nCUDA out of memory. Try reducing batch size or using model parallelism.")
    else:
        print(f"Training failed: {e}")

# =====================
# Save the Final Model
# =====================
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("Fine-tuning complete! Model saved at:", FINAL_MODEL_DIR)