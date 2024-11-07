import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

max_length_global = 256
dataset = load_dataset('cnn_dailymail', '3.0.0')
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
model = GPT2LMHeadModel.from_pretrained(model_name)

# Freeze all the model parameters
for param in model.parameters():
    param.requires_grad = False

class SoftPrompts(torch.nn.Module):
    def __init__(self, num_prompts, embedding_dim):
        super().__init__()
        # Initialize with the embedding of '[SUMMARIZE]' token
        summarize_token_id = tokenizer.encode('[SUMMARIZE]')[0]
        token_embedding = model.transformer.wte.weight[summarize_token_id].clone()
        self.soft_prompts = torch.nn.Parameter(
            token_embedding.expand(num_prompts, -1).clone(),
            requires_grad=True
        )
    def forward(self, batch_size):
        # Expand soft prompts for the batch
        return self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)

num_prompts = 10
embedding_dim = model.config.n_embd
tokenizer.pad_token = tokenizer.eos_token
train_size = 21000
train_subset = dataset['train'].select(range(train_size))
val_size = 6000
val_subset = dataset['validation'].select(range(val_size))
test_size = 3000
test_subset = dataset['test'].select(range(test_size))

class PromptTuningDataset(Dataset):
    def __init__(self, data):
        input_ids = []
        labels = []
        attention_masks = []
        for x in data:
            inputs = x['article']
            targets = x['highlights']
            tokenized_input = tokenizer(
                inputs, 
                max_length=max_length_global - num_prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tokenized_target = tokenizer(
                targets,
                max_length=max_length_global,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = tokenized_target.input_ids.clone()
            label[label == tokenizer.pad_token_id] = -100
            
            input_ids.append(tokenized_input.input_ids)
            attention_masks.append(tokenized_input.attention_mask)
            labels.append(label)
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.labels[idx]
        )

train_dataset = PromptTuningDataset(train_subset)
val_dataset = PromptTuningDataset(val_subset)

batch_size = 24
soft_prompts = SoftPrompts(num_prompts, embedding_dim)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(soft_prompts.parameters(), lr=5e-4)
model.to(device)
soft_prompts.to(device)

num_epochs = 10
train_losses = []
val_losses = []

def forward_pass(batch_input_ids, batch_attention_mask, batch_labels):
    batch_size = batch_input_ids.size(0)
    prompts = soft_prompts(batch_size)
    prompt_attention = torch.ones(batch_size, num_prompts, device=device)
    inputs_embeds = model.transformer.wte(batch_input_ids)
    inputs_embeds = torch.cat([prompts, inputs_embeds.squeeze(1)], dim=1)
    attention_mask = torch.cat([prompt_attention, batch_attention_mask.squeeze(1)], dim=1)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=batch_labels
    )
    return outputs.loss

for epoch in range(num_epochs):
    model.eval()
    soft_prompts.train()
    train_loss = 0
    num_samples = 0
    for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)
        loss = forward_pass(batch_input_ids, batch_attention_mask, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_samples += 1
    avg_train_loss = train_loss / num_samples
    train_losses.append(avg_train_loss)
    # Validation
    soft_prompts.eval()
    val_loss = 0
    num_val_samples = 0
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in val_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            loss = forward_pass(batch_input_ids, batch_attention_mask, batch_labels)
            val_loss += loss.item()
            num_val_samples += 1
    avg_val_loss = val_loss / num_val_samples
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# saving the fine tuned model
model.save_pretrained('./model_files')

# metrics for fine tuned model
test_dataset=PromptTuningDataset(test_subset)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def generate_summary(model, soft_prompts, input_ids, attention_mask, max_length=max_length_global):
    batch_size = input_ids.size(0)
    prompts = soft_prompts(batch_size)
    prompt_attention = torch.ones(batch_size, num_prompts, device=device)
    inputs_embeds = model.transformer.wte(input_ids)
    inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs

model.eval()
soft_prompts.eval()
predictions = []
references = []
rouge = evaluate.load("rouge")

with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_refs in test_dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        generated_ids = generate_summary(
            model,
            soft_prompts,
            batch_input_ids,
            batch_attention_mask
        )
        decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for g in generated_ids]
        predictions.extend(decoded_preds)
        references.extend(batch_refs)

results = rouge.compute(predictions=predictions, references=references)
print(results)

# metrics for pre-trained model
model_name = 'gpt2'
tokenizer2 = GPT2Tokenizer.from_pretrained(model_name)
tokenizer2.pad_token = tokenizer2.eos_token
tokenizer2.padding_side = 'left'
model2 = GPT2LMHeadModel.from_pretrained(model_name)
model2 = model2.to(device)

rouge = evaluate.load("rouge")
model2.eval()
generated_summaries = []
references = []
with torch.no_grad():
    for sample in test_subset:
        text = sample['article']
        reference_text=sample['highlights']
        inputs = tokenizer2(text, return_tensors="pt", max_length=max_length_global, truncation=True).to(device)
        outputs = model2.generate(
            inputs['input_ids'],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length_global,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer2.eos_token_id
        )
        generated_text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
        generated_summaries.append(generated_text)
        references.append(reference_text)

results = rouge.compute(predictions=generated_summaries, references=references)
print(results)