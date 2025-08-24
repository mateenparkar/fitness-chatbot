from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler, DataCollatorForLanguageModeling
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


def tokenise():
    tokeniser = GPT2Tokenizer.from_pretrained("gpt2")

    special_tokens_dict = {
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|user|>', '<|bot|>', '<|endoftext|>']
    }
    tokeniser.add_special_tokens(special_tokens_dict)

    raw_data = Path("data/conversations.txt").read_text().strip().split("\n")

    tokenized_data = []
    for dialogue in raw_data:
        dialogue = dialogue.strip()
        if not dialogue:
            continue
        encoded = tokeniser(
            dialogue,
            truncation=True,
            max_length=512,
            padding=False,  
            return_tensors="pt"  
        )
        tokenized_data.append({
            "input_ids": encoded["input_ids"].squeeze(),  
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze().clone()  
        })

    return tokenized_data, tokeniser


class ChatDataset(Dataset):
    def __init__(self, tokenized_data, tokeniser):
        self.data = tokenized_data
        self.tokeniser = tokeniser

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"].clone()  
        attention_mask = item["attention_mask"].clone()
        labels = item["labels"].clone()

        bot_token_id = self.tokeniser.convert_tokens_to_ids('<|bot|>')
        
        bot_positions = (input_ids == bot_token_id).nonzero(as_tuple=True)[0]
        
        if len(bot_positions) > 0:
            bot_start = bot_positions[0]
            labels[:bot_start + 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train(dataset, tokeniser):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        max_length = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(batch)):
            pad_length = max_length - len(input_ids[i])
            padded_input_ids.append(
                torch.cat([input_ids[i], torch.full((pad_length,), tokeniser.pad_token_id, dtype=torch.long)])
            )
            
            padded_attention_masks.append(
                torch.cat([attention_masks[i], torch.zeros(pad_length, dtype=torch.long)])
            )
            
            padded_labels.append(
                torch.cat([labels[i], torch.full((pad_length,), -100, dtype=torch.long)])
            )
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels)
        }

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokeniser))
    model.config.pad_token_id = tokeniser.pad_token_id
    model.to(device)

    optimiser = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(train_dataloader) * 6  
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimiser,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    scaler = GradScaler() if torch.cuda.is_available() else None
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    gradient_accumulation_steps = 8

    print("Starting training...")
    model.train()
    
    for epoch in range(4):
        print(f"\nEpoch {epoch + 1}/6")
        total_loss = 0
        optimiser.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler is not None:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimiser)
                    
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if scaler is not None:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                    
                optimiser.zero_grad()
                lr_scheduler.step()

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("  New best validation loss! Saving model...")
            model.save_pretrained("fitness-chatbot-model-best")
            tokeniser.save_pretrained("fitness-chatbot-model-best")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        model.train()

    return model


if __name__ == "__main__":
    print("Loading data and tokenizing...")
    tokenized_data, tokeniser = tokenise()

    dataset = ChatDataset(tokenized_data, tokeniser)
    print(f"Created {len(dataset)} training examples")

    model = train(dataset, tokeniser)

    print("\nTraining completed!")
    model.save_pretrained("fitness-chatbot-model-final")
    tokeniser.save_pretrained("fitness-chatbot-model-final")
    print("Final model saved!")