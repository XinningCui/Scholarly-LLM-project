import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from datasets import load_dataset
from datetime import datetime
import torch
from peft import LoraConfig, get_peft_model, TaskType

# Paths to your data
UNSUPERVISED_DATA_PATH = os.getenv("UNSUPERVISED_DATA_PATH", "unsupervised_data_part_small.txt")

model_path = "/home/xicu953f/.llama/converted_Llama3.2-3B-Instruct"

# Load the tokenizer and model for CausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")



# Configure LoRA
def configure_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Task type for Causal Language Modeling
        r=16,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate for LoRA
        target_modules=["q_proj", "v_proj"]  # Modules to apply LoRA
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()  # Verify trainable parameters
    return lora_model

# Preprocess data for CLM
def preprocess_clm_data(dataset_path, max_length=512):
    dataset = load_dataset("text", data_files={"train": dataset_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Training arguments for LoRA with GPU
def create_training_args(output_dir, learning_rate=3e-5, num_train_epochs=3):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=50000,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
        max_grad_norm=1.0,
        logging_steps=100,
        report_to="none",  # Disable external logging
        overwrite_output_dir=True,
        load_best_model_at_end=False,
    )

# LoRA fine-tuning
def train_lora_clm():
    print("Starting LoRA unsupervised fine-tuning...")
    dataset = preprocess_clm_data(UNSUPERVISED_DATA_PATH)

    # Configure LoRA on the model
    lora_model = configure_lora(model)

    # Create training arguments
    training_args = create_training_args(output_dir="/home/xicu953f/LLM/models/unsupervised_output")
    # Use a data collator for CLM to process labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling doesn't use masked language modeling
    )
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save the trained LoRA-adapted model and tokenizer
    model_output_dir = "/home/xicu953f/LLM/trained_models/unsupervised_trained_llama3.2_3B"
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    lora_model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print("LoRA unsupervised fine-tuning complete!")

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU not available. Please ensure you are running on a machine with GPU support.")
    
    train_lora_clm()
