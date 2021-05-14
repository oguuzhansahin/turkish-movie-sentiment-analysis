from flask import Flask,render_template,url_for,request
import torch
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 25
tokenizer = BertTokenizerFast.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
model = torch.load("model/bert_classifier.pth")

app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      
        text = request.form['text']
           
        tokenized_input = tokenizer.batch_encode_plus(
        [text],
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
        
    return render_template('result.html', prediction = preds)
            
    


if __name__ == '__main__':
    app.run(debug=True)
    
#%%

'''
def home():
    
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      
        text = request.form['text']
           
        tokenized_input = tokenizer.batch_encode_plus(
        [text],
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
        
    return render_template('result.html', prediction = preds)

'''









