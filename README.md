# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Develop an LSTM-based model to recognize named entities from text using the ner_dataset.csv, with words and NER tags as features.

<img width="336" height="715" alt="image" src="https://github.com/user-attachments/assets/c4a462f7-17e1-407f-a4c3-59f89f2ab15b" />



## DESIGN STEPS

### STEP 1:
Import necessary libraries.

### STEP 2:
Load and preprocess the dataset.

### STEP 3:
Group words into sentences.

### STEP 4:
Encode sentences and tags.

### STEP 5:
Prepare data for model training.

### STEP 6:
Define the LSTM model.

### STEP 7:
Train the model on training data.

### STEP 8:
Evaluate model performance.

## STEP 9:
Visualize predictions.

## PROGRAM

### Name:KIRUTHIGA.B

### Register Number:212224040160

```
# Model definition
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,tagset_size)

    def forward(self,x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)


model=BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)



# Training and Evaluation Functions
def train_model(model,train_loader,test_loader,loss_fn,optimixer,epochs=3):
  train_losses,val_losses=[],[]
  for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
      input_ids=batch["input_ids"].to(device)
      labels=batch["labels"].to(device)
      optimizer.zero_grad()
      outputs=model(input_ids)
      loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
      for batch in test_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        val_loss+=loss.item()
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

  return train_losses,val_losses    

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="777" height="637" alt="image" src="https://github.com/user-attachments/assets/4b5d8344-134d-4f51-8bd3-7d1768f1933c" />


### Sample Text Prediction

<img width="408" height="492" alt="image" src="https://github.com/user-attachments/assets/db12ff55-23ce-49ed-951b-9a38d979c867" />



## RESULT

Thus, an LSTM-based model for recognizing the named entities in the text is developed successfully.
