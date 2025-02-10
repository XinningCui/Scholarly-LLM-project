
from datasets import load_dataset, DatasetDict
import pandas as pd
import jsonlines
import json
import re
import torch


def get_desired_discipline(path_to_json, path_to_save, desired_discipline = "Computer Science"):
 
  """
  Takes path to the jsonl and required discipline and creates and saves a json file with "input" and "output" keys
  input key is paper with joined and preporcessed sections
  output key is the abstract of the paper

  Arguments
    path_to_json : path to the json file
    required_discipline: string with the desired discipline. Default is Computer Science as we are interested in the domain

  """

  torch.cuda.empty_cache()
  processed_data = []
  seen_inputs = set()
  #path to the one combined uniarxive data. Note that the data should be in a json line text format
  with jsonlines.open(path_to_json) as reader:
      for obj in reader:
            if obj['discipline'] == desired_discipline:
              abstract = obj.get('abstract', {}).get('text', '')
              body_text = obj.get('body_text', [])
              text_list = [section.get('text', '') for section in body_text]
              joined_text_input = "".join(text_list)
              preprocessed_input = preprocess_scientific_text(joined_text_input)
              if joined_text_input not in seen_inputs:
                  seen_inputs.add(preprocessed_input)
                  processed_data.append({
                      #"input": f"Summarize the following text {preprocessed_input}",
                      "input": preprocessed_input,
                      "output": abstract
                  })



  with open(path_to_save, "w") as f:
      json.dump(processed_data, f, indent=4)


def preprocess_scientific_text(text):
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', " ", text)
    # Remove table, formula, figure, and citation references
    text = re.sub(r"\{\{table:.*?\}\}", "", text)  
    text = re.sub(r"\{\{formula:.*?\}\}", "", text)  
    text = re.sub(r"\{\{figure:.*?\}\}", "", text) 
    text = re.sub(r"\{\{cite:.*?\}\}", "", text)  
    text = re.sub(r"\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?", "", text)  # Remove LaTeX commands 
    text = text.replace("{", "").replace("}", "")  # Remove remaining braces 

    # Replace non-breaking spaces and trim extra whitespace
    text = text.replace(u'\xa0', u' ')  # Replace non-breaking space
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces and trim


    return text




def preprocess_function(examples):
   return {
       'input': [preprocess_scientific_text(text) for text in examples['input']],
       'output': [preprocess_scientific_text(text) for text in examples['output']]
   }



def generate_save_train_test_split(data, path_to_save):
    data = data.map(preprocess_function, batched=True)
    #Split the dataset
    final_dataset = data.train_test_split(test_size=0.2)
    train_val_split = final_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    test_dataset = final_dataset['test']

    df_train = train_dataset.to_pandas()
    df_train[['input', 'output']].to_csv(f"{path_to_save}/train.csv", index = False)

    df_valid = val_dataset.to_pandas()
    df_valid[['input', 'output']].to_csv(f"{path_to_save}/valid.csv", index = False)

    df_test = test_dataset.to_pandas()
    df_test[['input', 'output']].to_csv(f"{path_to_save}/test.csv", index = False)

    print("Data splitted and converted to csv")