# # IMPORTS

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random
import json
import re
from sklearn.manifold import TSNE
from scipy import spatial
import matplotlib.pyplot as plt
import pickle

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 
a = torch.zeros(4,3) 
a = a.to(device)

# # Tokenizer

def _tokenize(i):
  pattern = r"\w+"
  words = re.findall(pattern,i)
  return words

# # Hyperparameters

window_size = 8
negative_sample_count = 10
thresold_frequency_count = 5
embedding_size = 256
lr = 0.01
epochs = 3

# # Preprocessing dataset

data_words = []
with open('data.json') as dataset:
  i=0
  for line in tqdm(dataset):
        text = json.loads(line)['reviewText']
        text = text.lower()
        data_words.append(_tokenize(text))
        i+=1
        if i==200000:
          break
  dataset.close()

# for deviding into test and train dataset, suffeling the data
random.shuffle(data_words)

# data_words = data_words[:10]
word_count = {}
for sent in tqdm(data_words):
  for word in sent:
    try:
      word_count[word] += 1
    except:
      word_count[word] = 1

# preparing word 2 index and reverse dictionaries
word2index = {}
index2word = {}

word2index["UNK"] = 0
index2word[0] = "UNK"

ind = 1
for word in word_count.keys():
  if word_count[word] >= thresold_frequency_count:
    word2index[word] = ind
    index2word[ind] = word
    ind += 1 

# converting data words into indeces

data = []

for i,sent in enumerate(data_words):
  x = []
  for j,word in enumerate(sent):
    try:
      x.append(word2index[word])
    except:
      x.append(0)
  data.append(x)

# ### dataset info

vocab_size = len(word2index.keys())
sent_count = len(data)
train_size = int(sent_count*0.85)
valid_size = sent_count - train_size

# # MODEL ARCHITECTURE

class word2vec_cbow(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(word2vec_cbow, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, embedding_size+embedding_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, context_words, target_words, negative_words):
        context_embeddings = self.embedding(context_words)
        context_embeddings = torch.mean(context_embeddings, dim=1)
        context_embeddings = self.linear(context_embeddings)
        num_samples = context_embeddings.shape[0]
        num_dim = context_embeddings.shape[1]
        context_embeddings = context_embeddings.view(num_samples,1,num_dim)

        target_embeddings = self.embedding(target_words)
        target_embeddings = self.linear(target_embeddings)

        negative_embeddings = self.embedding(negative_words)
        negative_embeddings = self.linear(negative_embeddings)

        return context_embeddings, negative_embeddings, target_embeddings

# # Negative Sampling
# 
# ---
# 
# 

def get_negative_samples(word):
  
  negative_samples = []

  for i in range(negative_sample_count):
    while True:
      x = int(random.random()*(vocab_size))
      if x == word or x == 0:
        continue    
      negative_samples.append(x)
      break

  return negative_samples

# # Training Function

def train(model,input_data,criterion,optimizer):
  model.train()    
  epoc_loss = 0

  t_batch = []
  c_batch = []
  n_batch = []
  running_loss = []
  for sent in tqdm(input_data):
    for i in range(window_size,len(sent)-window_size):

      t_batch.append([sent[i]])
      c_batch.append(sent[i-window_size:i] + sent[i+1:i+window_size+1])
      n_batch.append(get_negative_samples(sent[i]))

      if len(t_batch)%256 == 255:
        t_batch = torch.tensor(t_batch, dtype=torch.long).to(device)
        c_batch = torch.tensor(c_batch, dtype=torch.long).to(device)
        n_batch = torch.tensor(n_batch, dtype=torch.long).to(device)

        optimizer.zero_grad()
        
        predictions = model(c_batch, t_batch, n_batch)
        
        cs = torch.nn.CosineSimilarity(dim=2)
#         similarity = torch.matmul(predictions[0], torch.permute(predictions[2], (0, 2, 1))).to(device)
        similarity = cs(predictions[0],predictions[2]).to(device)
#         difference = -torch.matmul(predictions[0], torch.permute(predictions[1], (0, 2, 1))).to(device)
        difference = -cs(predictions[0], predictions[1]).to(device)
    
#         print(similarity[0],"\n",difference[0])
    
        output = torch.cat([similarity,difference],dim=1).to(device)

#         y_pos = torch.ones((predictions[0].shape[0],1)).to(device)
#         y_neg = torch.zeros((predictions[0].shape[0],negative_sample_count)).to(device)
#         y = torch.cat([y_pos,y_neg],dim=1).to(device)
        
        loss = -criterion(output)
        loss = torch.sum(loss, 1)
        loss = torch.mean(loss)
#         print(loss.shape,type(loss))
        
        loss.backward()
        optimizer.step()
        
        

        running_loss.append(loss.item())
        
        del t_batch
        del c_batch
        del n_batch

        t_batch = []
        c_batch = []
        n_batch = []

  epoch_loss = np.mean(running_loss)
  print("training epoch_loss is", epoch_loss)
  return epoch_loss

# # Validation Function

def evaluate(model,input_data,criterion):
  model.eval()    
  epoc_loss = 0

  t_batch = []
  c_batch = []
  n_batch = []
  running_loss = []

  for sent in tqdm(input_data):
    for i in range(window_size,len(sent)-window_size):

      
      t_batch.append([sent[i]])
      c_batch.append(sent[i-window_size:i] + sent[i+1:i+window_size+1])
      n_batch.append(get_negative_samples(sent[i]))

      if len(t_batch)%256 == 255:
        t_batch = torch.tensor(t_batch, dtype=torch.long).to(device)
        c_batch = torch.tensor(c_batch, dtype=torch.long).to(device)
        n_batch = torch.tensor(n_batch, dtype=torch.long).to(device)
        
        predictions = model(c_batch, t_batch, n_batch)
  
        similarity = torch.matmul(predictions[0], torch.permute(predictions[2], (0, 2, 1))).to(device)
        difference = -torch.matmul(predictions[0], torch.permute(predictions[1], (0, 2, 1))).to(device)

        output = torch.cat([similarity,difference],dim=2).to(device)
        
        m = nn.LogSoftmax(dim=1)
        output = -m(output)

        y_pos = torch.ones((predictions[0].shape[0],1)).to(device)
        y_neg = torch.zeros((predictions[0].shape[0],negative_sample_count)).to(device)
        y = torch.cat([y_pos,y_neg],dim=2).to(device)



        loss = criterion(output, y)
  
        running_loss.append(loss.item())
        
        t_batch = []
        c_batch = []
        n_batch = []

  epoch_loss = np.mean(running_loss)
  print("validation epoch_loss is", epoch_loss)
  return epoch_loss

# # Model Criterion Optimizer

model = word2vec_cbow(vocab_size,embedding_size)
model.to(device)   

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.LogSoftmax(dim=1)
optimizer = optim.Adam(model.parameters(), lr=lr)

# # Save Embeddings

def save_embeddings():
  saved_embeddings = {}
  for i in tqdm(range(1,len(index2word.keys()))):#(len(index2word)):
      word = index2word[i]
#       saved_embeddings[word] = model.linear.weight[i].detach().cpu().numpy()
      saved_embeddings[word] = model.embedding.weight[i].detach().cpu().numpy()
  return saved_embeddings
      

# # Training with validation

# # to reduce the learning rate by a factor of 2 after every epoch associated with no improvement
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
# best_validation_loss = float('inf')

# best_embeddings = {}

# for epoch in range(epochs):
#   print("epoch number",epoch+1)
#   train_loss = train(model, train_sents,criterion,optimizer)
#   validation_loss = evaluate(model, valid_sents,criterion)

#   lr_scheduler.step(validation_loss)

#   if validation_loss < best_validation_loss:
#     best_validation_loss = validation_loss
#     best_embeddings = save_embeddings()
#     with open('best_embeddings.pkl','wb') as f:
#       pickle.dump(best_embeddings,f)

# only doing training no validation
best_embeddings = {}
for epoch in range(5):
  print("epoch number",epoch+1)
  train_loss = train(model, data,criterion,optimizer)



  

best_embeddings = save_embeddings()

with open('final_embeddings_2k.pkl','wb') as f:
  pickle.dump(best_embeddings,f)

torch.save(model.state_dict(),"Finalmodel.pt")

# For loading the pickle file

# file = open('final_embeddings_2k.pkl','rb')
# best_embeddings = pickle.load(file)
# file.close()

# For loading the model

# model = word2vec_cbow(vocab_size,embedding_size)
# model.load_state_dict(torch.load("Finalmodel.pt"))
# model.eval()

# # TSNE

# 5 selected words
words = ["glad","titanic","camera","lovely","points"]
# words = ["director","characters","camera","movie","entertain","shows","titanic","there","their","knife","war","great"]


top_embeddings = []

close_count = 10

for word in words:
  word_embedding = best_embeddings[word]
  similarities = []
  for w in best_embeddings:
    if w==word:
      continue
    similarities.append([1 - spatial.distance.cosine(word_embedding,best_embeddings[w]),w])
  similarities.sort(reverse=True)
  print(word)
  print("similar words - ",end = "")
  for w in similarities[:close_count]:
    print(w[1],end=" | ")
  print()
  top_embeddings.append([word,similarities[:close_count]])
    

X = []
Y = []
W = []
for x in top_embeddings:
  simi_ = x[1]
  X.append(best_embeddings[x[0]])
  Y.append(x[0])
  for y in simi_:
    X.append(best_embeddings[y[1]])
    Y.append(y[1])
    W.append(y[1])

X = np.array(X)
Y = np.array(Y)

n_components = 2

tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(X)

# now need to plot close words in different colours
for i in range(len(words)):
  xx = []
  yy = []
  zz = []
#   xx.append(tsne_result[i*(close_count+1)][0])
#   yy.append(tsne_result[i*(close_count+1)][1])
#   zz.append(Y[i*(close_count+1)])
  
  plt.scatter([tsne_result[i*(close_count+1)][0]], [tsne_result[i*(close_count+1)][1]])
  plt.annotate(Y[i*(close_count+1)], (tsne_result[i*(close_count+1)][0], tsne_result[i*(close_count+1)][1] + 0.2))
  
  for j in range(i*(close_count+1) + 1,i*(close_count+1)+(close_count+1)):
    xx.append(tsne_result[j][0])
    yy.append(tsne_result[j][1])
    zz.append(Y[j])
  for ii in range(len(xx)):
    plt.annotate(zz[ii], (xx[ii]-0.5, yy[ii] + 0.2))
  plt.scatter(xx, yy)
  print("word - ",Y[i*(close_count+1)])
  print("similar words - ")
  for w in zz:
    print(w,end=",")

  plt.show()
