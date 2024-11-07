import torch
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt

max_length_global=256
dataset = load_dataset('cnn_dailymail', '3.0.0')
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"Total Number of Trainable Parameters (Before freezing all layers): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Freeze all parameters in the transformer
for param in model.parameters():
    param.requires_grad = False
# Unfreeze only the lm_head parameters
for param in model.lm_head.parameters():
    param.requires_grad = True
print(f"Total Number of Trainable Parameters (After freezing all layers except the lm_head (final, classification layer)): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
tokenizer.pad_token = tokenizer.eos_token

# Use 10% of the dataset for training, validation, and testing
train_size = 21000
train_subset = dataset['train'].select(range(train_size))
val_size = 6000
val_subset = dataset['validation'].select(range(val_size))
test_size = 3000
test_subset = dataset['test'].select(range(test_size))

class FineTuningDataset(Dataset):
    def __init__(self, data):
        input_ids=[]
        labels=[]
        for x in data:
            inputs = x['article']
            targets = x['highlights']
            inp_ids = tokenizer(inputs, max_length=max_length_global, padding="max_length", truncation=True, return_tensors="pt").input_ids
            lab = tokenizer(targets, max_length=max_length_global, padding="max_length", truncation=True, return_tensors="pt").input_ids
            lab[lab == tokenizer.eos_token_id] = -100
            labels.append(lab)
            input_ids.append(inp_ids)
        self.input_ids=input_ids
        self.labels=labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.input_ids[idx],self.labels[idx]

train_dataset = FineTuningDataset(train_subset)
val_dataset = FineTuningDataset(val_subset)

batch_size = 24
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
num_epochs = 3
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    num_samples = 0
    for inp,lab in train_dataloader:
        inputs = inp.to(device)
        labels = lab.to(device)
        outputs = model(input_ids=inputs.squeeze(1), labels=labels.squeeze(1))
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_samples += 1
    avg_train_loss = train_loss / num_samples
    train_losses.append(avg_train_loss)
    # Validation after each epoch
    model.eval()
    val_loss = 0
    num_val_samples = 0
    with torch.no_grad():
        for inp,lab in val_dataset:
            inputs=inp.to(device)
            labels=lab.to(device)
            outputs = model(input_ids=inputs.squeeze(1), labels=labels.squeeze(1))
            loss = outputs.loss
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

model.save_pretrained('./model_files')

# metrics for fine tuned model
rouge = evaluate.load("rouge")
model.eval()
generated_summaries = []
references = []
with torch.no_grad():
    for sample in test_subset:
        text = sample['article']
        reference_text=sample['highlights']
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length_global, truncation=True).to(device)
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length_global,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_summaries.append(generated_text)
        references.append(reference_text)

results = rouge.compute(predictions=generated_summaries, references=references)
print(results)

# metrics for pre-trained model
model_name = 'gpt2'
tokenizer2 = GPT2Tokenizer.from_pretrained(model_name)
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