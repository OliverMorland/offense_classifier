import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
from datasets import load_from_disk


def load_data(file_path):
    """
    Load data from a CSV file, converting labels to numeric codes.
    """
    df = pd.read_csv(file_path, encoding='cp1252')
    texts = df["text"].tolist()
    labels = df["label"].astype("category").cat.codes.tolist()  # Convert labels to numeric codes
    label_mapping = dict(enumerate(df["label"].astype("category").cat.categories))
    return texts, labels, label_mapping


class OffenseClassifier:
    def __init__(self, model_dir="offense_classifier_end_to_end/model"):
        """
        Initialize the Offense Classifier with end-to-end fine-tuning and GPU support.
        """
        print("Initializing Offense Classifier...")
        self.model = None
        self.tokenizer = None
        self.encoded_dataset = None
        self.label_mapping = None
        self.model_loaded = False

        # Check for GPU availability
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.load_or_train_model()

    def load_tokenizer(self):
        if self.tokenizer is None:
            model_name = "offense_classifier_end_to_end/distilbert_base_uncased/"
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        if self.encoded_dataset is not None:
            return
        # Load data
        file_path = "offense_classifier_end_to_end/datasets/samples.csv"
        texts, labels, self.label_mapping = load_data(file_path)

        # Split data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Create Hugging Face Dataset
        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Tokenization function
        def tokenize(batch):
            return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

        # Tokenize datasets
        cache_path = "offense_classifier_end_to_end/cache/encoded_dataset"
        if os.path.exists(cache_path):
            print("Loading tokenized dataset from cache...")
            encoded_dataset = load_from_disk(cache_path)
        else:
            print("Tokenizing dataset...")
            encoded_dataset = dataset.map(tokenize, batched=True, num_proc=4)
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            encoded_dataset.save_to_disk(cache_path)
        self.encoded_dataset = encoded_dataset

    def load_or_train_model(self):
        if self.model_loaded:
            return

        self.load_tokenizer()
        self.prepare_data()

        # Check if a saved model exists
        if os.path.exists(self.model_dir):
            print("Loading the pre-trained model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        else:
            print("No pre-trained model found. Starting training from scratch...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.tokenizer.name_or_path,
                num_labels=len(self.label_mapping)
            ).to(self.device)
            self.train_model(self.encoded_dataset)
            # Evaluate the model
            self.evaluate_model(self.encoded_dataset['test'])

    def train_model(self, encoded_dataset):
        """
        Train the model using Hugging Face's Trainer API with end-to-end fine-tuning.
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            metric_for_best_model="accuracy",
            fp16=True  # Enable mixed-precision training for faster computation
        )

        # Metrics function
        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return {
                'accuracy': (preds == p.label_ids).mean()
            }

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()
        trainer.evaluate()

        # Save the trained model
        print("Saving the trained model...")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def evaluate_model(self, test_dataset):
        """
        Evaluate model performance on the test set and display classification report.
        """
        print("\nEvaluating model on the test dataset...")
        trainer = Trainer(model=self.model)
        predictions, labels, _ = trainer.predict(test_dataset)

        # Ensure predictions and labels are on the same device
        predictions = torch.tensor(predictions).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        preds = torch.argmax(predictions, dim=1).cpu().numpy()  # Move to CPU before converting to NumPy

        # Generate and display classification report
        report = classification_report(labels.cpu().numpy(), preds, target_names=list(self.label_mapping.values()))
        print("\nClassification Report:\n", report)

    def funnel_category(self, input_category):
        dictionary = {
            'Public Order': ["Alcohol Violation", "Resisting Arrest", "Illicit Business", "Obstructing Justice", "Public Institution Violation"],
            'Traffic': ["Traffic Driving", "Traffic Vehicle", "Traffic Paperwork"]
        }
        for key, categories in dictionary.items():
            for category in categories:
                if input_category.lower() == category.lower():
                    return key
        return input_category

    def should_classify_to_all_other_offense(self, input_text):
        clues = ["parties", "habitual violator", "habitual offender"]
        for clue in clues:
            if clue in input_text.lower():
                return True

    def classify_text(self, text):
        """
        Classify a single text string.
        """
        if self.should_classify_to_all_other_offense(text):
            return "All Other Offenses", 0
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Move to CPU before converting to NumPy
        prediction_index = np.argmax(probabilities)
        predicted_label = self.label_mapping[prediction_index]
        confidence_score = probabilities[prediction_index]
        funneled_category = self.funnel_category(predicted_label)
        return funneled_category, round(float(confidence_score), 2)


if __name__ == "__main__":
    classifier = OffenseClassifier()
