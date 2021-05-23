import torch
import torch.nn as nn
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TurkishSentimentBERT(nn.Module):

  def __init__(self, model,freeze_bert=False):

    super(TurkishSentimentBERT,self).__init__()
    
    self.bert = model

    D_in, H, D_out = 768,50, 2

    

    self.classifier  = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(H,D_out)
    )

    if freeze_bert:
      for param in self.bert.parameters():

        param.requires_grad = False

  def forward(self, input_ids, attention_mask):

    outputs = self.bert(input_ids = input_ids,
                        attention_mask = attention_mask)
    
    last_hidden_state_cls = outputs[0][:, 0, :]

    # Feed input to classifier to compute logits
    logits = self.classifier(last_hidden_state_cls)

    return logits


def initialize_model(epochs,model,train_dataloader):
    
    
  bert_classifier = TurkishSentimentBERT(model,freeze_bert=False)
  bert_classifier.to(device)

  optimizer = AdamW(bert_classifier.parameters(),
                    lr = 5e-5,
                    eps=1e-8)

  total_steps = len(train_dataloader) * epochs

  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps = total_steps)
  
  return bert_classifier, optimizer, scheduler 



def set_seed(seed_value =42):

  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  
  
  
def train(model,loss_fn,optimizer,scheduler,train_dataloader, val_dataloader=None, epochs=4, evaluation=False):

  print("Start training...\n")

  for epoch_i in range(epochs):

    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*70)

    # Measure the elapsed time of each epoch
    t0_epoch, t0_batch = time.time(), time.time()

    # Reset tracking variables at the beginning of each epoch
    total_loss, batch_loss, batch_counts = 0, 0, 0


    model.train()

    for step,batch in enumerate(train_dataloader):

      batch_counts+=1

      b_inputs_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

      model.zero_grad()

      logits = model(b_inputs_ids, b_attn_mask)

      loss = loss_fn(logits,b_labels)
      batch_loss += loss.item()
      total_loss += loss.item()

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

      optimizer.step()
      scheduler.step()

      if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            # Calculate time elapsed for 20 batches
            time_elapsed = time.time() - t0_batch

            print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

            # Reset batch tracking variables
            batch_loss, batch_counts = 0, 0
            t0_batch = time.time()

    avg_train_loss = total_loss / len(train_dataloader)

    print("-"*70)

    if evaluation:

      val_loss, val_accuracy = evaluate(model, loss_fn,val_dataloader)

      time_elapsed = time.time() - t0_epoch
      print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
      print("-"*70)
  
  print("Training complete!")
  

def evaluate(model,loss_fn, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def predict_test(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def evaluate_roc(probs, y_true):
    
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def predict(tokenizer,model,input,device,MAX_LEN):
    
    tokenized_input = tokenizer.batch_encode_plus(
    [input],
    max_length = MAX_LEN,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)
      
    text_tensor      = torch.tensor(tokenized_input['input_ids'])
    attention_tensor = torch.tensor(tokenized_input['attention_mask']) 
    
    model.eval()
    
    with torch.no_grad():
        text_tensor,attention_tensor = text_tensor.to(device),attention_tensor.to(device)
        logits = model(text_tensor,attention_tensor)

    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = np.where(probs[:, 1] > 0.5, 1, 0)
    
    return preds
    # if preds[0] == 0:
    #     print("Bu bir negatif yorumdur...")
    
    # else:
    #     print("Bu bir pozitif yorumdur")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    