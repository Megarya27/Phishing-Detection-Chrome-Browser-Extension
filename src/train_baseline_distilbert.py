#This program trains the DistilBERT model for binary classification of email text (malicious vs benign)
# It uses the Hugging Face Transformers library to load a pretrained DistilBERT model, fine-tune it on a labeled email dataset, 
# and evaluate its performance using metrics like F1 score, precision, and recall. 
# also saves the trained model and evaluation results.

#imports
import os
import json
import numpy as np
import torch

#used for creating custom data loadeers
from torch.utils.data import Dataset

# Hugging Face tools for NLP models, tokenization and model training
from transformers import ( DistilBertTokenizerFast,
    DistilBertForSequenceClassification, Trainer, TrainingArguments)

# metrics from scikit-learn to measure classification accuracy
from sklearn.metrics import f1_score, precision_score, recall_score

from global_config import (
    DISTILBERT_MODEL_NAME, MAX_SEQ_LEN, MODELS_DIR,
    RESULTS_DIR, RANDOM_SEED,
)
from data_text import build_text_dataset
from metrics import full_report


#tensorizes the text using the pretrained DistilBERT tokenizer, truncates/pads to a fixed length, and attaches the ground-truth labels.
class EmailTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_SEQ_LEN):
        #convert inputs to standard Python lists
        text_list = list(texts)
        self.labels = list(labels)

        #convert text into numerical tokens:
        #truncation=True: cut off emails longer than max_len
        #padding="max_length": add zeros to make all emails equal length
        self.encodings = tokenizer(text_list, truncation=True, padding="max_length",
        max_length=max_len,
        )
# returns total number of samples in this dataset
    def __len__(self):
        return len(self.labels)
#retrieves one training example at the given index (idx). Converts data into PyTorch tensors before inputting to model.
    def __getitem__(self, idx):
        item = {}
        #tokenizer outputs keys like 'input_ids' and 'attention_mask'
        for key in self.encodings.keys():
            feature_value = self.encodings[key][idx]
            item[key] = torch.tensor(feature_value)

        #attach truth label 
        target_label = self.labels[idx]
        item["labels"] = torch.tensor(target_label, dtype=torch.long)
        return item


#calculates F1, Precision, and Recall after each training epoch. Used by Hugging Face Trainer.
def compute_metrics(eval_pred):
    logits, true_labels = eval_pred #unpack the raw model outputs (logits) and actual labels
    logits_tensor = torch.tensor(logits) #convert logits to raw PyTorch tensor
    probabilities = torch.softmax(logits_tensor, dim=1) #apply softmax across output classes to get probabilities summing to 1
    class_1_probs = probabilities[:, 1].numpy()#extract the probability of positive class (Class 1)
    predicted_labels = (class_1_probs >= 0.5).astype(int) #convert probabilities to binary predictions (0 or 1) using 0.5 threshold

    #compute evaluation metrics
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    return {
        "f1": f1, "precision": precision, "recall": recall}


#loads pretrained DistilBERT, sets up training options, and runs training loop.
def train_distilbert(splits, output_dir, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    #load pretrained tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME,
        num_labels=2,  #binary classification head (2 classes)
    )

    #wrap train and validation sets into our custom Dataset format
    train_dataset = EmailTextDataset(
        texts=splits["train"]["text"],
        labels=splits["train"]["label"],
        tokenizer=tokenizer,
    )
    val_dataset = EmailTextDataset(
        texts=splits["val"]["text"],
        labels=splits["val"]["label"],
        tokenizer=tokenizer,
    )

    #configure hyperparameters and training settings
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,                  #number of passes over training set
        per_device_train_batch_size=16,      #samples per GPU/CPU during training
        per_device_eval_batch_size=32,       #samples per GPU/CPU during evaluation
        learning_rate=2e-5,                  #optimizer step size
        weight_decay=0.01,                   #L2 regularization to prevent overfitting
        eval_strategy="epoch",               #run evaluation at the end of each epoch
        save_strategy="epoch",               #save model checkpoint at end of each epoch
        load_best_model_at_end=True,         #keep best checkpoint after training finishes
        metric_for_best_model="f1",          #choose best model based on highest F1 score
        logging_steps=50,                    #prints training loss every 50 batches
        fp16=torch.cuda.is_available(),       #uses mixed precision if GPU is available as it is faster and uses less memory
        seed=seed,
        report_to=[])

    #Initialise Hugging Face trainer engine
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train() #start training
    return trainer, tokenizer


#generates predicted probabilities for class 1 given a list of raw email strings.
def predict_proba(trainer, tokenizer, texts):
    #create dummy labels (all zeros) as Dataset class needs a label input
    dummy_labels = [0] * len(texts)
    dataset = EmailTextDataset(texts, dummy_labels, tokenizer)

    #run predictions using trained model
    prediction_output = trainer.predict(dataset)
    raw_logits = torch.tensor(prediction_output.predictions)

    #Convert logits to class probabilities and extract class 1
    probabilities = torch.softmax(raw_logits, dim=1)
    class_1_probs = probabilities[:, 1].numpy()

    return class_1_probs


def main():
    #load pre-split email dataset (train, val, test)
    splits = build_text_dataset()
    output_dir = MODELS_DIR / "distilbert_semantic"

    #train model and get back trainer and tokenizer
    trainer, tokenizer = train_distilbert(splits, output_dir)

    #evaluate trained model on unseen testset
    test_texts = splits["test"]["text"].tolist()
    actual_labels = splits["test"]["label"]

    #generate probabilities and finalpredictions
    predicted_probabilities = predict_proba(trainer, tokenizer, test_texts)
    predicted_classes = (predicted_probabilities >= 0.5).astype(int)

    #generate detailed classification report
    report, f1_bootstrap_distribution = full_report(
        "distilbert_semantic",
        actual_labels,
        predicted_classes,
        predicted_probabilities,
    )

    #print summary metrics 
    print("Test Evaluation Report:")
    print(json.dumps(report, indent=2))

    #save model checkpoints and test results to disk
    final_model_path = str(output_dir / "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    #save bootstrap evaluation array and JSON performance report
    np.save(
        RESULTS_DIR / "baseline_distilbert_f1_bootstrap.npy",
        f1_bootstrap_distribution,
    )
    with open(RESULTS_DIR / "baseline_distilbert_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Model and results successfully saved to {final_model_path}")


if __name__ == "__main__":
    main()