
""""
Evaluation for instruction fine-tuned Meta-Llama-3.1-8B-Instruct-bnb-4bit for medical data simplifiation
Note this is a modified version of the notebook by Unsloth AI. 
The source code in https://github.com/unslothai/unsloth
Modifications: Data processing before and after generation
Dataset link:  https://github.com/SebaJoe/MultiCochrane

"""

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import torch
import pandas as pd
import json 
from collections.abc import Iterable
from transformers import TextStreamer
import numpy as np
import evaluate
import torch
from tqdm import tqdm


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
#     "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
#     "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
#     "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
#     "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
#     "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#     "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
#     "unsloth/Phi-3-medium-4k-instruct",
#     "unsloth/gemma-2-9b-bnb-4bit",
#     "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
# ] # More models at https://huggingface.co/unsloth

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Llama-3.2-1B-Instruct",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )


max_seq_length = 32000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


test_dataset = pd.read_csv("./data/en/test0.5_en.csv")

#Specify the path to the correct checkpoints
model_path = "./checkpoints"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,

)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}

"""

EOS_TOKEN = tokenizer.eos_token
instruction = "Simplify the following text."

def processing_of_the_data(data):
  formated_dataset = []
  for input_text, response in zip(data['input_text'], data['target_text']):
    if not isinstance(input_text, str) or not isinstance(response, str) or input_text.strip() == "" or response.strip() == "":
      continue
    formated_text = alpaca_prompt.format(instruction, input_text.strip(), response.strip()) + EOS_TOKEN
    formated_dataset.append({"text": formated_text})


  return formated_dataset

med_formated_dataset_test = processing_of_the_data(test_dataset)
#print(f"Length of the formatted test dataset is {len(med_formated_dataset_test)}")

# Convert the list of dictionaries to a Hugging Face Dataset
hf_dataset_test = Dataset.from_dict({"text": [item["text"] for item in med_formated_dataset_test]}) #test


# Function to modify each example
def remove_response(example):
    # Split the text by sections to isolate and remove the response part
    parts = example['text'].split("### Response:\n")
    if len(parts) > 1:
        example['text'] = parts[0] + "### Response:\n "  # Leave a blank space after 'Response:'
    return example

hf_dataset_test_no_response = hf_dataset_test.map(remove_response)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

text_streamer = TextStreamer(tokenizer)

item = hf_dataset_test_no_response[0]["text"]
inputs = tokenizer(item, return_tensors = "pt", truncation=True, padding=True).to("cuda")

# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

rouge = evaluate.load("rouge")
sari = evaluate.load("sari")
bertscore = evaluate.load("bertscore")

predictions = []
references = []
target_text = []

for idx in tqdm(range(len(hf_dataset_test_no_response))):
    input_text = hf_dataset_test_no_response[idx]["text"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    #output_ids = model.generate(**inputs, max_new_tokens=1024, length_penalty=1.2) #For introducing length penalty to the short answers
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)    
    # Extract the response part after "### Response:\n"
    response = generated_text.split("### Response:\n")[-1].strip()
    predictions.append(response)
    # Get the reference response
    reference = hf_dataset_test[idx]["text"].split("### Response:\n")[-1].strip()
    references.append(reference)


#optionally save the generated responses as a json file at the path you specify.
#uncomment the lines 145-152 for saving the responses along with the inputs.
#resp_to_save = [{'inp': inp, 'resp': resp} for inp, resp in zip(references, predictions)]
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_path", type = str, default="simplification_generated_responses.json", help="Path to save the generated responses")
# args = parser.parse_args()
# path_to_save = args.save_path
# with open(path_to_save, "w") as f:
#     json.dump(resp_to_save, f, indent = 4)

# Calculate metrics


print("Calculating ROUGE BLEU and Sari metrics for instruction fine-tuned models for scientific text simplification...")

rouge_scores = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {rouge_scores['rouge1']}")
print(f"ROUGE-2: {rouge_scores['rouge2']}")
print(f"ROUGE-L: {rouge_scores['rougeL']}")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
print(f"BERTScore F1: {sum(bertscore_result['f1']) / len(bertscore_result['f1']):.2f}")


bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(f"Bleu score is {results}")

sari_scores = []
for prediction, target, source in zip(predictions, test_dataset["target_text"], references):
    sources = [source]
    predictions = [prediction] 
    references_answ = [target]
    sari_score = sari.compute(sources = sources, predictions = predictions , references = [references_answ])  
    sari_scores.append(sari_score)

extracted_sari_scores = []
for item in sari_scores:
    for key, value in item.items():
        extracted_sari_scores.append(value)

print(f"Average Sari score is {np.mean(extracted_sari_scores)}")



