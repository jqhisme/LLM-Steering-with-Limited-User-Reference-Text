import pandas as pd
import numpy as np
import json
import requests

import time
import re
from tqdm import tqdm

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

#model_name = "unsloth/Llama-3.2-3B-Instruct"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
DATA_ENTRY_START = 14241

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model)
with open("corpse.json", "r") as json_file:
    corpse = json.load(json_file)
df = pd.read_csv("features.csv")
descriptions = ", ".join(df["description"].apply(lambda x:"\'"+x+"\'").tolist())

# helper methods
def extract_answer(answer:str):
  answer = answer.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1].replace("<|eot_id|>","").replace("\n","").strip()
  #answer = answer.replace('"', '\\"') # handle " in json string
  return answer


def validate_json_string(s):
  if s.endswith("}"):
    return s
  last_comma_index = s.rfind(",")
  if last_comma_index != -1:
    s = s[:last_comma_index]
    s += "}"
    return s
  else:
    return "invalid string"

def clean_input_text(text):
    text = re.split(r'tl;dr', text, flags=re.IGNORECASE)[0]
    text = re.split(r'tldr', text, flags=re.IGNORECASE)[0]
    text = re.sub(r"[\\'\"]", "", text)
    return text

synth_data = []
print("Start running data synthesis...")
for i in range(DATA_ENTRY_START,len(corpse)):
    input_text = clean_input_text(corpse[i]['text'])

    # first round of inference: select prompt
    feature_selection_prompt = '''The following is a list of features, separated by commas: [%s]
    Select 10 features or less that are highly relevent to the user input text from list provided. Do not select features that are irrelevent. Do not select any features that are not on the list. 
    Please output your answer in the following python dictionary format, order them by relevancy:
    {"order 1":{"feature name":feature name from the list,"reason":reason for selection, no more than 1 sentence},"order 2":{"feature name":feature name from the list,"reason":reason for selection, no more than 1 sentence},...}
    Please do not output anything else other than the dictionary.'''%descriptions

    messages = [
    {"role": "system","content": feature_selection_prompt},
    {"role": "user", "content": f"text: {input_text}"},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    generated_content = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,top_p=0.95,temperature=0.1)
    decoded_text = tokenizer.decode(generated_content[0])
    entry = validate_json_string((extract_answer(decoded_text)))
    try:
        entry = json.loads(entry)
        feature_names = [entry[f'order {i}']['feature name'] for i in range(1,len(entry)) if entry[f'order {i}']['feature name'] in df['description'].to_list()]
    # except json.JSONDecodeError as e:
    except:
        continue # skip this entry if the json is not valid

    

    if len(feature_names) ==0:
        continue # skip this entry if the features is not in my dictionary


    # second inference feed for scoring
    generate_score_prompt= f'''
    You are a machine learning algorithm that returns a dictionary that predict how features are presented in the user's the input text, evaluated using a score between -1 to 1.
    1 indicates that the sentence fully represents the feature, 0 indicates that the sentence doesn't represent the feature at all, and -1 indicates that the sentence fully represents the opposite of the feature.\n\
    
    Instructions:
    From the following list of the feature, predict scores that meansure each features presented in the input sentence.
    Please predict to the second decimal place. Please output a dictionary to represent the prediction. Please do not output anything else other than the dictionary. \n\
    The output format should be:{{"feature name 1":value,"feature name 2":value,...}} \n\
    
    The following is the list of features, separated by comma: [{",".join(feature_names)}]
    '''
    messages = [
        {
            "role": "system",
            "content": generate_score_prompt ,
        },
        {"role": "user", "content": f"text: {input_text}"},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    generated_content = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,top_p=0.95,temperature=0.1)
    decoded_text = tokenizer.decode(generated_content[0])
    entry = validate_json_string((extract_answer(decoded_text)))
    try:
        entry = json.loads(entry)
    except json.JSONDecodeError as e:
        continue # skip this entry if the json is not valid

    # organize output
    output_entry ={}
    output_features = []
    output_scores =[]
    output_feature_ids = []
    # ensure all features are in the dictionary
    for index,feature in enumerate(entry.keys()):
      if feature in df["description"].to_list(): # it exist in the dictionary
        output_features.append(feature)
        output_scores.append(entry[feature])
        output_feature_ids.append(int(np.where(feature ==df["description"].to_numpy())[0][0]))
    
    if len(output_features):
      output_entry ={
        "text":input_text,
        "features":output_features,
        "feature_ids":output_feature_ids,
        "scores":output_scores
      }

    synth_data.append(output_entry)

    # write to the file for every 10 entries
    if len(synth_data)%10==0:
        with open("synth_data.txt", "a") as file:
            for entry in synth_data:
                file.write(json.dumps(entry))
                file.write("\n\n")
                
        synth_data=[]
        print(f"Proceeded to {DATA_ENTRY_START+i} item in corpse")


    