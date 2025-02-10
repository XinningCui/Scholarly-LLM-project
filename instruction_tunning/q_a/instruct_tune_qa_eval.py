

""""
Evaluating fine-tuned Meta-Llama-3.1-8B-Instruct-bnb-4bit, Phi-3.5-mini-instruct, Qwen2.5-0.5B-Instruct-bnb-4bit
Note this is a modified version of the notebook by Unsloth AI. 
The source code in https://github.com/unslothai/unsloth
Modifications: Data processing before and after generation
"""
from unsloth import FastLanguageModel
import re
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from collections.abc import Iterable
import json
from transformers import TextStreamer
import evaluate
from tqdm import tqdm
import argparse

max_seq_length = 32000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

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

dataset = load_dataset("allenai/qasper")
#Specify the exact checkpoint you want to evaluate for
model_path = "./checkpoints"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,

)
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

formatted_dataset_test = create_alpaca_formatted_dataset(dataset["test"])


print("test dataset length", len(formatted_dataset_test))


# Convert the list of dictionaries to a Hugging Face Dataset
hf_dataset_test = Dataset.from_dict({"text": [item["text"] for item in formatted_dataset_test]})

# Function to modify each example
def remove_response(example):
    # Split the text by sections to isolate and remove the response part
    parts = example['text'].split("### Response:\n")
    if len(parts) > 1:
        example['text'] = parts[0] + "### Response:\n "  # Leave a blank space after 'Response:'
    return example

# Apply the function to the dataset
hf_dataset_test_no_response = hf_dataset_test.map(remove_response)


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

text_streamer = TextStreamer(tokenizer)

item = hf_dataset_test_no_response[0]["text"]
inputs = tokenizer(item, return_tensors = "pt", truncation=True, padding=True).to("cuda")

# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# Load metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
predictions = []
references = []

# Function for cleaning the generated texts.
def clean_text(text):
    # Remove any numbers and unwanted characters like newlines, periods, etc.
    cleaned_text = re.sub(r'\d+\.\n?', '', text)  # Removes numbers like "1.\n", "2.\n", etc.
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    return cleaned_text


for idx in tqdm(range(len(hf_dataset_test_no_response))):
    input_text = hf_dataset_test_no_response[idx]["text"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    
    # Generate response
    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract the response part after "### Response:\n"
    response = generated_text.split("### Response:\n")[-1].strip()
    predictions.append(clean_text(response))
    
    # Get the reference response
    reference = hf_dataset_test[idx]["text"].split("### Response:\n")[-1].strip()
    references.append(clean_text(reference))

#optionally save the generated responses as a json file at the path you specify.
#uncomment the lines 204-211 for saving the responses along with the inputs.
#resp_to_save = [{'inp': inp, 'resp': resp} for inp, resp in zip(references, predictions)]
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_path", type = str, default="q_a_generated_responses.json", help="Path to save the generated responses")
# args = parser.parse_args()
# path_to_save = args.save_path
# with open(path_to_save, "w") as f:
#     json.dump(resp_to_save, f, indent = 4)


# Calculate metrics
print("Calculating ROUGE AND BLEU metrics for instruction fine-tuned models for scientific question and answering...")

rouge_scores = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {rouge_scores['rouge1']}")
print(f"ROUGE-2: {rouge_scores['rouge2']}")
print(f"ROUGE-L: {rouge_scores['rougeL']}")

bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
print(f"BERTScore F1: {sum(bertscore_result['f1']) / len(bertscore_result['f1']):.2f}")

bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {bleu_result['bleu']:.2f}")
