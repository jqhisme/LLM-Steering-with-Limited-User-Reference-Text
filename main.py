import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sae_lens import SAE, HookedSAETransformer
import json
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm
import gc
import torch.cuda.amp as amp

class TextDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.data = []
        self.max_length = max_length
        self.file_path = file_path  # Store file_path instead of initializing tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item['text'])
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        # Delay tokenizer initialization
        self.tokenizer = None
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Initialize tokenizer in worker
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
        text = self.data[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'full_text': text
        }

class FeatureGenerator(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', output_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.feature_proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = bert_output.last_hidden_state[:, 0, :]
        
        features = self.feature_proj(pooled_output)
        return features

class TextGenerationPipeline:
    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        
        # Initialize models
        self.initialize_models()
        
        # Initialize mixed precision scaler
        self.scaler = amp.GradScaler()
        
    def initialize_models(self):
        # Initialize SAE on CPU first
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb-feature-splitting",
            sae_id="blocks.8.hook_resid_pre_768",
            device='cpu'
        )
        self.sae = self.sae.to(self.device)
        
        # Initialize GPT-2 with SAE hooks
        self.model = HookedSAETransformer.from_pretrained("gpt2")
        self.model.to(self.device)
        
        # Initialize feature generator and sentence transformer
        self.feature_generator = FeatureGenerator().to(self.device)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_transformer.to(self.device)
        
        # Freeze models except feature generator
        self.freeze_models()
        
    def freeze_models(self):
        for model in [self.sae, self.model, self.sentence_transformer]:
            for param in model.parameters():
                param.requires_grad = False
    
    def generate_batch_texts(self, prompts, feature_vectors):
        """Generate texts for a batch of prompts and feature vectors"""
        generated_texts = []
        
        for prompt, feature_vector in zip(prompts, feature_vectors):
            def steering_hook(activations, hook):
                return activations + feature_vector.to(activations.device)
            
            input_ids = self.model.to_tokens(prompt, prepend_bos=True)
            
            with self.model.hooks(fwd_hooks=[(self.sae.cfg.hook_name, steering_hook)]):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    stop_at_eos=True,
                    prepend_bos=True
                )
                
            generated_texts.append(self.model.tokenizer.decode(output[0]))
            
        return generated_texts
    
    def compute_batch_similarities(self, features, original_texts, generated_texts):
        """Compute similarities for a batch of texts"""
        with torch.no_grad():
            orig_emb = self.sentence_transformer.encode(original_texts, convert_to_tensor=True)
            gen_emb = self.sentence_transformer.encode(generated_texts, convert_to_tensor=True)
            
            orig_emb = orig_emb.to(self.device)
            gen_emb = gen_emb.to(self.device)
        
        # Compute feature influence for each example
        feature_influences = torch.tanh(features.mean(dim=1))
        
        # Compute similarities
        similarities = cosine_similarity(orig_emb, gen_emb)
        
        # Combine similarities with feature influences
        combined_scores = similarities * (1.0 + 0.1 * feature_influences)
        
        return combined_scores
    
    def train_feature_generator(self, train_loader, num_epochs=2, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.feature_generator.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_similarity = 0
            num_batches = 0
            
            # Training loop with progress bar
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            
            for batch in pbar:
                batch_size = len(batch['input_ids'])
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Generate features with mixed precision
                with amp.autocast():
                    features = self.feature_generator(input_ids, attention_mask)
                    
                    # Prepare prompts
                    initial_prompts = [' '.join(text.split()[:20]) 
                                     for text in batch['full_text']]
                    
                    # Generate texts
                    generated_texts = self.generate_batch_texts(initial_prompts, features)
                    
                    # Compute similarities
                    similarity_scores = self.compute_batch_similarities(
                        features, batch['full_text'], generated_texts
                    )
                    
                    # Compute loss
                    loss = 1.0 - similarity_scores.mean()
                
                # Backpropagation with gradient scaling
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Update metrics
                total_loss += loss.item()
                total_similarity += similarity_scores.mean().item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'similarity': f'{similarity_scores.mean().item():.4f}'
                })
            
            # Print epoch statistics
            avg_loss = total_loss / num_batches
            avg_similarity = total_similarity / num_batches
            print(f'Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, '
                  f'Avg Similarity = {avg_similarity:.4f}')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.feature_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'feature_generator_epoch_{epoch+1}.pth')

def main():
    # Set device and batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32  
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    try:
        # Initialize dataset and dataloader
        dataset = TextDataset('test.txt')
        # dataset = TextDataset('synth_data.txt')
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,  
            pin_memory=True  
        )
        
        # Print dataset size
        print(f"Dataset size: {len(dataset)} records")
        print(f"Number of batches: {len(dataloader)}")
        
        # Initialize and train pipeline
        pipeline = TextGenerationPipeline(device=device, batch_size=batch_size)
        pipeline.train_feature_generator(dataloader)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()