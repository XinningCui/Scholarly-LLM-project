
""""
Fine-tunning Meta-Llama-3.1-8B-Instruct-bnb-4bit for medical data simplifiation
Note this is a modified version of the notebook by Unsloth AI. 
The source code in https://github.com/unslothai/unsloth
Modifications: Data processing and parameters changed
Dataset link:  https://github.com/SebaJoe/MultiCochrane

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
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import random

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
] # More models at https://huggingface.co/unsloth


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = fourbit_models[0],
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    #random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



#Assuming the data and the training code are in the same directory.
train_dataset = pd.read_csv("./data/en/train0.5_en.csv")
val_dataset = pd.read_csv("./data/en/val0.5_en.csv")
test_dataset = pd.read_csv("./data/en/test0.5_en.csv")


alpaca_prompt = """Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}

"""

instruction = "Simplify the following text."

EOS_TOKEN = tokenizer.eos_token




# Optionally adds noise to the training data.
# This can help improve generalization but might also reduce accuracy.  
def add_typo(input_text):
    words = input_text.split()
    typo_position = random.randint(0, len(words)-1)
    words[typo_position] = words[typo_position][::-1]  # Reverse one word to create a typo
    if random.random() < 0.6:
      typo_posirion_2 = random.randint(0, len(words)-1)
      words[typo_posirion_2] = words[typo_posirion_2].upper()
    return " ".join(words)

def add_noise(text):
  random_prob = random.random()
  if random_prob < 0.2:
    return add_typo(text)
  return text


#The prefix can be either ['clean','noisy'] and determines whether we add noise to the data or not
#By default it is set to clean
def processing_of_the_data(data, prefix = "clean"):
  if prefix not in ['clean', 'noisy']:
    raise ValueError("Prefix needs to be either one of ['clean','noisy']")
  formated_dataset = []
  for input_text, response in zip(data['input_text'], data['target_text']):
    if not isinstance(input_text, str) or not isinstance(response, str) or input_text.strip() == "" or response.strip() == "":
      continue  
    if prefix == "noisy":
      input_text, response = add_noise(input_text), add_noise(response)  
    
    formated_text = alpaca_prompt.format(instruction, input_text.strip(), response.strip()) + EOS_TOKEN
    formated_dataset.append({"text": formated_text})


  return formated_dataset


med_formatted_dataset_train = processing_of_the_data(train_dataset)
med_formated_dataset_val = processing_of_the_data(val_dataset)
med_formated_dataset_test = processing_of_the_data(test_dataset)

# print(f"Length of the formatted train dataset is {len(med_formatted_dataset_train)}")
# print(f"Length of the formatted validation dataset is {len(med_formated_dataset_val)}")
# print(f"Length of the formatted test dataset is {len(med_formated_dataset_test)}")


# Convert the list of dictionaries to a Hugging Face Dataset
hf_dataset_train = Dataset.from_dict({"text": [item["text"] for item in med_formatted_dataset_train]})
hf_dataset_val = Dataset.from_dict({"text": [item["text"] for item in med_formated_dataset_val]}) 
hf_dataset_test = Dataset.from_dict({"text": [item["text"] for item in med_formated_dataset_test]})

# Calculate the maximum token length
max_token_length = max(len(tokenizer(text)["input_ids"]) for text in hf_dataset_train["text"])
print(f"Maximum token length in the dataset: {max_token_length}")


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
    max_seq_length = max_token_length,
    dataset_num_proc = 8,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
	    num_train_epochs = 2, 
	    learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        output_dir = output_dir,
        report_to = "wandb",
        eval_strategy="steps",
        eval_steps=20,
        per_device_eval_batch_size=1,
        save_steps= 20, 
	    #load_best_model_at_end = True, #uncomment this if want to load the best model at the end
        logging_dir = "./logs",
        
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
