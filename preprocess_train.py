import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statistics
import random
import transformers 
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import (
  AutoModel, 
  BertTokenizerFast
)

from torch.utils.data import (
  DataLoader, 
  TensorDataset, 
  RandomSampler, 
  SequentialSampler
)
from utils import (
  TurkishSentimentBERT,
  initialize_model,
  train,
  evaluate,
  set_seed,
  predict_test,
  evaluate_roc,
  device,
  predict
)

CSV_NAME = "turkish_movie_sentiment_dataset.csv"
PRE_TRAINED_MODEL = 'savasy/bert-base-turkish-sentiment-cased'
MAX_LEN = 25
BATCH_SIZE = 64
EPOCHS = 1


df = pd.read_csv(CSV_NAME)
df['point'] = df['point'].apply(lambda x:float(x[0:-2])) # point sütunundaki değerler 5,5 şeklindeydi bunları floata dönüştürdüm.
df['comment'] = df['comment'].apply(lambda x: x.strip()) # commentlerin başında /n ve fazladan boşluklar vardı, bunları kaldırdım.
df = df.drop(df[df.point==3].index,axis=0) # 3 puan verilen yorumlar nötr yorumlar olduğu için bunları verisetinden çıkarıyorum.

#%%

# Tahmin etmeye çalıştığımız şey, pozitif/negatif yorum şeklinde olduğu için 1 ve 2 puanı negatif, 4 ve 5'i ise pozitif kabul ediyoruz.

df['point'] = df['point'].replace(2.0 , 0.0) 
df['point'] = df['point'].replace(1.0 , 0.0)
df['point'] = df['point'].replace(4.0 , 1.0)
df['point'] = df['point'].replace(5.0 , 1.0)
df['point'] = df['point'].astype(int)
df.reset_index(inplace = True) # Bazı satırları çıkardığımız için indexler bozuldu, onları düzelttik.
df.drop(["index","film_name"], axis = 1, inplace = True) #film isimlerini de kullanmayacağımız için çıkarıyoruz.

#%%

train_text, temp_text, train_y, temp_y = train_test_split(df['comment'], 
                                                          df['point'],
                                                          random_state=42,
                                                          test_size = 0.3)


val_text, test_text, val_y, test_y = train_test_split(temp_text,
                                                      temp_y,
                                                      random_state=42,
                                                      test_size = 0.5)

#%%

# Modele verdiğimiz inputlar aynı uzunlukta olması gerektiği için veriseti içerisinden rastgele seçilmiş 100 örneğin uzunluklarına bakıp
# max_len belirliyoruz.

sentences_length = [len(i.split()) for i in train_text]
pd.Series(random.sample(sentences_length,100)).hist(bins=100)
print(statistics.median(sentences_length))

#%%

tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL)
model = AutoModel.from_pretrained(PRE_TRAINED_MODEL)


train_tokens = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LEN,
    pad_to_max_length = True,
    truncation = True,
    return_token_type_ids = False
)

val_tokens = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LEN,
    pad_to_max_length = True,
    truncation = True,
    return_token_type_ids = False
)

test_tokens = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LEN,
    pad_to_max_length = True,
    truncation = True,
    return_token_type_ids = False
)

#%%

train_text_tensor = torch.tensor(test_tokens['input_ids'])
train_mask_tensor = torch.tensor(test_tokens['attention_mask'])
train_y_tensor    = torch.tensor(test_y.tolist())

val_text_tensor = torch.tensor(val_tokens['input_ids'])
val_mask_tensor = torch.tensor(val_tokens['attention_mask'])
val_y_tensor    = torch.tensor(val_y.tolist())

test_text_tensor = torch.tensor(test_tokens['input_ids'])
test_mask_tensor = torch.tensor(test_tokens['attention_mask'])
test_y_tensor    = torch.tensor(test_y.tolist())

#%%

train_data = TensorDataset(train_text_tensor,
                           train_mask_tensor,
                           train_y_tensor)

train_sampler = RandomSampler(train_data)
train_dataloader= DataLoader(train_data,
                             sampler = train_sampler,
                             batch_size = BATCH_SIZE)

val_data = TensorDataset(val_text_tensor,
                         val_mask_tensor,
                         val_y_tensor)
val_sampler= SequentialSampler(val_data)
val_dataloader = DataLoader(val_data,
                            sampler=val_sampler,
                            batch_size= BATCH_SIZE)

#%%
set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS, model=model, train_dataloader = train_dataloader)

loss_fn = nn.CrossEntropyLoss()

train(bert_classifier,loss_fn,optimizer,scheduler, train_dataloader, val_dataloader, epochs=EPOCHS, evaluation=True)

#%%
probs = predict_test(bert_classifier,val_dataloader)
evaluate_roc(probs, val_y_tensor)

#%%
torch.save(bert_classifier,"bert_classifier.pth")
saved_model = torch.load("bert_classifier.pth")



#%%
test_deneme = "güzel olmadığını ve  beklentilerimi karşılamayacağını düşünsem de beni yanıltan bir film oldu."


predict(tokenizer = tokenizer,
        model = saved_model,
        input = test_deneme,
        device = device,
        MAX_LEN = MAX_LEN)


