# ===== SCRIPT OVERVIEW =====
# Task:   Apply fine-tuned mDeBERTa model to infer 8 emotion dimensions per tweet.
#         Emotions: anger_v2, fear_v2, disgust_v2, sadness_v2, joy_v2,
#                   enthusiasm_v2, pride_v2, hope_v2
# Input:  <DATA_DIR>/emotions/split_files/Austria-Panel-Text_*.csv.gz  (output of 2-split_files.py)
#         <script_dir>/pol_emo_mDeBERTa/model/pytorch_model.pt  (fine-tuned weights; not in repo)
# Output: <DATA_DIR>/emotions/emotion_inference.csv.gz  (columns: id + 8 emotion scores)
# Usage:  python 3-infer_emotion_split.py <DATA_DIR>
# Note:   GPU (CUDA) strongly recommended; batch size 32, max token length 280.
#         Model: https://github.com/tweedmann/pol_emo_mDeBERTa2

#from https://github.com/tweedmann/pol_emo_mDeBERTa2/releases/tag/v.1.0.0
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import F1Score
from torchmetrics.functional import accuracy, auroc #F1Score #f1
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, DebertaV2Model, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import os
import tqdm
import sys
import dask.dataframe as dd

# set data directory (absolute so it survives the chdir below)
src = os.path.abspath(sys.argv[1]) # e.g. /misinfo-panel-austria/data/emotions/

# set working directory to pol_emo_mDeBERTa (script-relative, not cwd-relative)
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Working directory set to:", os.getcwd())

# df = pd.read_csv(os.path.join(src, file), 
#                                 compression="gzip",
#                                 usecols=["id","text_cleaned"],
#                                 dtype={"id": int, "text_cleaned":str},
#                                 nrows=30_000_000)

# df = dd.read_csv(os.path.join(
#             src, file),
#                  compression="gzip",
#                  usecols=["id", "text_cleaned"],
#                  dtype={"id": int, "text_cleaned": str},
#                  blocksize="500MB")


#define function to apply mDeBERTa model
LABEL_COLUMNS = ['anger_v2', 'fear_v2', 
                 'disgust_v2', 'sadness_v2', 
                 'joy_v2', 'enthusiasm_v2', 
                 'pride_v2', 'hope_v2']
BASE_MODEL_NAME = "microsoft/mdeberta-v3-base"
BATCH_SIZE = 32
# CHUNK_SIZE = 30_000_000

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Process running on:", device)

class CrowdCodedTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = DebertaV2Model.from_pretrained(BASE_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.last_hidden_state[:, 0])
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):

    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(LABEL_COLUMNS):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-5) #DEFINING LEARNING RATE

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

# define function for inference
def predict_labels(df):
    input_text = df["text_cleaned"].astype(str).tolist()
    num_inputs = len(input_text)
    num_batches = (num_inputs - 1) // BATCH_SIZE + 1

    try:
        for i, batch in enumerate(tqdm.tqdm(range(num_batches))):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_inputs)
            batch_text = input_text[start_idx:end_idx]

            encoded_input = tokenizer(batch_text, 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=280, 
                                      return_tensors='pt')
            outputs = model(**encoded_input.to(device))

            tensor_values = outputs[1].tolist()
            decimal_numbers = [[num for num in sublist] for sublist in tensor_values]

            output_df = pd.DataFrame(decimal_numbers, columns=LABEL_COLUMNS)
            input_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            output_df = pd.concat([input_df, output_df], axis=1)

            # append to an existing CSV file or create a new one
            output_df.to_csv(out_file,
                             mode='a', index=False,
                             header=not os.path.exists(out_file),
                             compression="gzip")

    except KeyboardInterrupt:
        print("KeyboardInterrupt.")
        return

out_file = os.path.join(src, 'emotions', 'emotion_inference.csv.gz')
if os.path.exists(out_file):
    os.remove(out_file)
    print(f"Removed existing output file to avoid duplicates: {out_file}")

# put model into evaluation mode and load local fine-tuned model
model = CrowdCodedTagger(n_classes=8)
model.load_state_dict(torch.load("./model/pytorch_model.pt"), strict = False)
model.to(device)
model.eval()

# for every file in a given directory, apply the function
for file in os.listdir(os.path.join(src, "emotions", "split_files")):
    if file.endswith(".csv.gz"):
        print("Inferencing file:", file)
        df = pd.read_csv(os.path.join(src, "emotions", "split_files", file),
                         compression="gzip",
                         usecols=["id", "text_cleaned"],
                         dtype={"id": str, "text_cleaned": str})
        predict_labels(df)
    else:
        continue  
print("Inference completed.")

