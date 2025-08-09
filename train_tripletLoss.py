import numpy as np
import matplotlib.pyplot as plt
from sae_lens import SAE, HookedSAETransformer
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

import json
with open("synth_data_v2.json","r") as f:
    corpse = json.loads(f.read())
trainset = []
for i in corpse:
    try:
        trainset.append(json.loads(i)['text'])
    except:
        continue
print(f"Train set size: {len(trainset)}")

# model
class FASG_Model(nn.Module):
    def __init__(self,device="cuda:0"):
        super(FASG_Model, self).__init__()

        self.device = device
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertModel = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.linear = torch.nn.Linear(768, 768).to(self.device)
        self.llm = HookedSAETransformer.from_pretrained("gpt2-small", device=self.device)
        self.sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb-feature-splitting",  
            sae_id="blocks.8.hook_resid_pre_768", 
            device=self.device,
        )
        self.top_p = 0.95
        self.temperature = 0.5
        
    def bert_tokenize(self,text):
        return self.bertTokenizer(text,padding=True,truncation=True,return_tensors="pt").to(device = self.device)
        
    def forward(self, encoded_input,prompt,return_type = "str"):
        steering_vector = self.bertModel(**encoded_input).pooler_output

        def steering_features(value, hook,steering_vector = steering_vector,steering_weights = 4):
            #encoded_activation = self.sae.encode(value)
            #steered_vector = steering_vector.unsqueeze(1)*encoded_activation 
            delta_activation = self.sae.decode(steering_vector)

            steered_activation = delta_activation.unsqueeze(1)*steering_weights + value
            
            return steered_activation
    
        fwd_hooks=[(
            'blocks.8.hook_resid_pre', 
            steering_features
        )]

        tokenized_prompt = self.llm.to_tokens(prompt)
        with self.llm.hooks(fwd_hooks=fwd_hooks):
            steered_tokens = self.llm.generate(
            tokenized_prompt,
            max_new_tokens=64,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_at_eos = True,
            return_type = return_type,
            verbose = False
        )
        return steered_tokens

    def llm_generate(self,prompt,return_type = "str"):
        tokenized_prompt = self.llm.to_tokens(prompt)
        generated_tokens = self.llm.generate(
            tokenized_prompt,
            max_new_tokens=64,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_at_eos = True,
            return_type = return_type,
            verbose = False
        )
        return generated_tokens
        

    def freeze_llm_and_sae(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.sae.parameters():
            param.requires_grad = False

# loss
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

class FASG_Loss(nn.Module):
    def __init__(self,device="cuda:0"):
        super(FASG_Loss, self).__init__()
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = device

    def forward(self, reference_text,steered_text,baseline_text,temp= 1):
        tokenized_inputs = self.sentence_transformer.tokenize(
            [reference_text, steered_text, baseline_text])
        # move all tensors to gpu
        for key in tokenized_inputs.keys():
            tokenized_inputs[key] = tokenized_inputs[key].to(self.device)
        embeddings = self.sentence_transformer(tokenized_inputs)['sentence_embedding']
        embeddings = embeddings * temp
        
        # Compute cosine similarities
        sim_positive = 1-st_util.cos_sim(embeddings[0], embeddings[1])
        sim_negative = 1-st_util.cos_sim(embeddings[0], embeddings[2])

        # Compute softmax triplet loss
        margin = 0.4
        triplet_loss = torch.max(sim_positive-sim_negative+margin,0)[0] # torch mx returns (max, max_indices)
        loss = triplet_loss + sim_positive*0.5 # still preserve a little 

        return loss

# training loop
from torch.optim import AdamW

model = FASG_Model()
model.freeze_llm_and_sae()
model.train()
soft_triplet_loss =  FASG_Loss()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)



batch_size =16
epochs = 5

for epoch in range(epochs):
    print(f"epoch {epoch+1}")
    with open("training_log_tripletloss.txt","a") as f:
        f.write(f"epoch {epoch}\n")

    total_loss = []
    for index in tqdm(range(0,len(trainset),batch_size)):
        text = trainset[index:index+16]
        encoded_input = model.bert_tokenize(text)
        prompts = [" ".join(i.split(" ")[:5]) for i in text]
        steered_text = model(encoded_input,prompts)
        baseline_text = model.llm_generate(prompts)
    
        max_len = np.min([len(text),len(steered_text),len(baseline_text)])
        # Compute loss
        loss = soft_triplet_loss(text[:max_len],steered_text[:max_len],baseline_text[:max_len])
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        total_loss += [loss.item()]

        if len(total_loss)>0 and len(total_loss)%50==0:
            with open("training_log_tripletloss.txt","a") as f:
                for loss_value in total_loss[len(total_loss)-50:len(total_loss)]:
                    f.write(str(loss_value))
                    f.write("\n")
                    
        # write model every epoch
        torch.save(model.state_dict(),f"model_ckpts/fasg_model_epoch{epoch}.pth")
                
        
        

        