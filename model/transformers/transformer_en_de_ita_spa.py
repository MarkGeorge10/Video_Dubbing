import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, MarianMTModel
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
import os
import logging
import warnings
warnings.filterwarnings("ignore")

# Configuration for each language pair
CONFIG = {
    "es": {
        "dataset": "/storage/work/mqf5675/Masters/NLP/project/data/spa.txt",
        "output_dir": "/storage/work/mqf5675/Masters/NLP/project/model_output/transformer_en_es",
        "model_name": "Helsinki-NLP/opus-mt-en-es"
    },
    "deu": {
        "dataset": "/storage/work/mqf5675/Masters/NLP/project/data/deu.txt",
        "output_dir": "/storage/work/mqf5675/Masters/NLP/project/model_output/transformer_en_deu",
        "model_name": "Helsinki-NLP/opus-mt-en-de"
    },
    "ita": {
        "dataset": "/storage/work/mqf5675/Masters/NLP/project/data/ita.txt",
        "output_dir": "/storage/work/mqf5675/Masters/NLP/project/model_output/transformer_en_ita",
        "model_name": "Helsinki-NLP/opus-mt-en-it"
    }
}

def setup_logging(output_dir):
    """Set up logging for the given output directory."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def preprocess_data(examples, tokenizer):
    inputs = examples["english"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(model, tokenizer, train_df, val_df, output_dir, batch_size=8, num_epochs=10, lr=2e-5):
    logger = setup_logging(output_dir)
    logger.info("Starting model training...")
    device = torch.device("cpu")
    model.to(device)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer), 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_data(x, tokenizer), 
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    logger.info(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if input_ids.shape[1] != 128:
                logger.warning(f"Adjusting input_ids shape from {input_ids.shape} to (batch_size, 128)")
                input_ids = torch.nn.functional.pad(input_ids, (0, 128 - input_ids.shape[1]))
                attention_mask = torch.nn.functional.pad(attention_mask, (0, 128 - input_ids.shape[1]))
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
        
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_loss}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                if input_ids.shape[1] != 128:
                    input_ids = torch.nn.functional.pad(input_ids, (0, 128 - input_ids.shape[1]))
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, 128 - input_ids.shape[1]))
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss}")
    
    # Save model and tokenizer
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Model and tokenizer saved to {model_dir}")
    
    # Plotting with handling for single epoch
    plt.figure(figsize=(12, 4))
    epochs_range = range(1, num_epochs + 1)
    
    if num_epochs > 1:
        plt.plot(epochs_range, train_losses, label='Training Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
    else:
        plt.scatter(epochs_range, train_losses, label='Training Loss', color='blue')
        plt.scatter(epochs_range, val_losses, label='Validation Loss', color='orange')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()
    logger.info("Training history plot saved.")
    
    return model, tokenizer

def compute_bleu(model, tokenizer, test_df, device):
    logger = logging.getLogger(__name__)
    logger.info("Computing BLEU score...")
    bleu = evaluate.load("bleu")
    predictions = []
    references = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_df), 8):
            batch = test_df[i:i+8]
            inputs = tokenizer(list(batch["english"]), return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = list(batch["target"])
            predictions.extend(preds)
            references.extend([ref] for ref in refs)
    score = bleu.compute(predictions=predictions, references=references)["bleu"]
    logger.info(f"BLEU Score: {score}")
    return score

def load_and_translate(model_path, text):
    """Load saved model and translate text"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path} for translation...")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    logger.info(f"Translated '{text}' to '{translation}'")
    return translation

def train_for_language(lang_code):
    """Train a model for a specific language."""
    config = CONFIG[lang_code]
    dataset_path = config["dataset"]
    output_dir = config["output_dir"]
    model_name = config["model_name"]
    
    # Set up logging for this language
    logger = setup_logging(output_dir)
    logger.info(f"Training model for language: {lang_code}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    # Read all columns, then select only 'english' and 'target'
    df = pd.read_csv(dataset_path, sep='\t', header=None)
    # Assuming the first column is 'english' and the second is 'target'
    df = df.iloc[:, [0, 1]]  # Select only the first two columns
    df.columns = ['english', 'target']  # Rename columns
    df = df.dropna()
    df = df.drop_duplicates()
   
    df['english'] = df['english'].str.lower().str.strip().str.replace(r'[^\w\s.]', '', regex=True)
    df['target'] = df['target'].str.lower().str.strip().str.replace(r'[^\w\s.]', '', regex=True)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_subset = train_df[:10000]
    
    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"First few rows:\n{df.head().to_string()}")
    
    # Initialize model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Train
    model, tokenizer = train_model(
        model, tokenizer, 
        train_subset, 
        val_df, 
        output_dir
    )
    
    # Compute BLEU
    bleu_score = compute_bleu(
        model, tokenizer, 
        test_df, 
        torch.device("cpu")
    )
    
    # Example translations
    logger.info("Performing example translations...")
    model_path = os.path.join(output_dir, "model")
    print(f"Translation of 'go.' ({lang_code}):", load_and_translate(model_path, "go."))
    print(f"Translation of 'hi.' ({lang_code}):", load_and_translate(model_path, "hi."))

def main():
    # Train for each language
    for lang_code in CONFIG.keys():
        train_for_language(lang_code)

if __name__ == "__main__":
    main()