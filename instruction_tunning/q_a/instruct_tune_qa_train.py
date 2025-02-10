

""""
Fine-tunning Meta-Llama-3.1-8B-Instruct-bnb-4bit, Phi-3.5-mini-instruct, Qwen2.5-0.5B-Instruct-bnb-4bit
Note this is a modified version of the notebook by Unsloth AI. 
The source code in https://github.com/unslothai/unsloth
Modifications: Data processing and parameters changed
Dataset from https://huggingface.co/datasets/allenai/qasper

"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from datasets import Dataset
import json
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
import argparse



max_seq_length = 32000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



dataset = load_dataset("allenai/qasper")
import json
from collections.abc import Iterable

# Flatten function to handle nested structures
def flatten(nested_iterable):
    """Recursively flattens any nested iterable except strings and bytes."""
    for element in nested_iterable:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def create_alpaca_formatted_dataset(dataset):
    """
    Processes the input dataset and returns a new JSON dataset with Alpaca-formatted prompts.

    Args:
    - dataset (list): Original dataset in list of dict format

    Returns:
    - List[Dict]: New dataset with Alpaca-formatted `text` column
    """
    formatted_dataset = []

    for idx, instance in enumerate(dataset):
        try:
            questions = instance["qas"]["question"]  # List of questions
            answers = instance["qas"]["answers"]  # Corresponding answers
            
            # Make sure 'questions' and 'answers' are lists
            if not isinstance(questions, list) or not isinstance(answers, list):
                print(f"Instance {idx}: 'questions' or 'answers' is not a list. Skipping.")
                continue

            # Make sure 'questions' and 'answers' have the same length
            if len(questions) != len(answers):
                print(f"Instance {idx}: Number of questions and answers do not match. Skipping.")
                continue
            
            # Flatten paragraphs
            nested_paragraphs = instance.get("full_text", {}).get("paragraphs", [])
            flat_paragraphs = list(flatten(nested_paragraphs)) 
            # Combine flattened paragraphs into a single string
            input_text = " ".join(flat_paragraphs)
            
            for i in range(len(questions)):
                question = questions[i].strip()
                answer_entry = answers[i]                
                if "answer" not in answer_entry or not isinstance(answer_entry["answer"], list) or len(answer_entry["answer"]) == 0:
                    print(f"Instance {idx}, QA pair {i}: Missing 'answer' data. Skipping this QA pair.")
                    continue
                
                free_form_answer = answer_entry["answer"][0].get("free_form_answer", "").strip()
                
                if not free_form_answer:
                    print(f"Instance {idx}, QA pair {i}: 'free_form_answer' is empty. Skipping this QA pair.")
                    continue


                # Format the text using Alpaca prompt template
                text = alpaca_prompt.format(question, input_text, free_form_answer) + EOS_TOKEN
                formatted_dataset.append({"text": text})
        
        except KeyError as e:
            # print(f"Instance {idx}: Missing key {e}. Skipping this instance.")
            continue
        except Exception as e:
            # print(f"Instance {idx}: Unexpected error {e}. Skipping this instance.")
            continue

    return formatted_dataset


#Create train-validation-test pairs
formatted_dataset_train = create_alpaca_formatted_dataset(dataset["train"])
formatted_dataset_val = create_alpaca_formatted_dataset(dataset["validation"])
formatted_dataset_test = create_alpaca_formatted_dataset(dataset["test"])


# print("Length of the new formatted dataset:")
# print("train dataset length", len(formatted_dataset_train))
# print("test dataset length", len(formatted_dataset_test))
# print("validation dataset length", len(formatted_dataset_val))



# Convert to a Hugging Face Dataset
hf_dataset_train = Dataset.from_dict({"text": [item["text"] for item in formatted_dataset_train]})
hf_dataset_val = Dataset.from_dict({"text": [item["text"] for item in formatted_dataset_val]})
hf_dataset_test = Dataset.from_dict({"text": [item["text"] for item in formatted_dataset_test]})

#for saving the checkpoints
output_dir = "./checkpoints"
os.makedirs(output_dir, exist_ok=True)

# parser = argparse.ArgumentParser()
# parser.add_argument("--output_dir", type=str, default="/.checkpoints", help="Directory to save the checkpoinst")
# args = parser.parse_args()
# os.makedirs(args)
# print(f"Saving model to {args.output_dir}")


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = hf_dataset_train,
    eval_dataset= hf_dataset_val,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        num_train_epochs = 5, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = output_dir,
        report_to = "wandb", # Use this for WandB etc
        eval_strategy="steps",
        eval_steps=20, #args.save_step,
        per_device_eval_batch_size=1,
        save_steps= 20, 
        logging_dir = "./logs",

    ),
)


trainer_stats = trainer.train()
