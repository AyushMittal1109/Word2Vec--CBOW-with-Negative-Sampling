**Ayush Mittal**
**2021201030**
# iNLP Assignment 3
## For Training and preparing embeddings
### 1. SVD + Co-occurrence Matrix
keep the dataset with the name `data.json` with `SVD.py`, and run the SVD.py file as we run a normal python program, no argument needed
`python3 SVD.py`
### 2. CBOW
keep the dataset with the name `data.json` with `CBOW_NS.py`, and run the CBOW_NS.py file as we run a normal python program, no argument needed
`python3 CBOW_NS.py`

## **FOR USING PRETRAINED EMBEDDINGS**
### 1. **SVD + Co-occurrence Matrix**
 - Since weights(result after SVD) are stored and kept in a pickle file, we can directly use them
 - keep `svd_embeddings_200k_2_256.pkl` and `data.json` file in the same directory as 'SVD.py' file
 - Need to comment the training part and pre-processing part of the code
 - Uncomment the code for loading the dictionary, codes to uncomment - 
`file = open('svd_embeddings_200k_2_256.pkl','rb')`
`best_embeddings = pickle.load(file)`
`file.close()`
these code will load the embedding saved as dictionary in a pickle file
 - add all the words in `words` list, for which you want similar words
 - run python file as
`python3 SVD.py`

### 2. **CBOW**
 - Since model and embeddings weights(result after CBOw word2vec) are stored and kept in a pickle file, we can directly use them
 - keep `Finalmodel.pt`,`final_embeddings_2k.pkl` and `data.json` file in the same directory as `CBOW.py` file
 - Need to comment the training part and pre-processing part of the code
 - Uncomment the code for loading the dictionary, codes to uncomment - 
for loading embedding - 
`file = open('final_embeddings_2k.pkl','rb')`
`best_embeddings = pickle.load(file)`
`file.close()`
for loading model(not required for getting similar words) - 
`model = word2vec_cbow(vocab_size,embedding_size)`
`model.load_state_dict(torch.load("Finalmodel.pt"))`
`model.eval()`
these code will load the embedding saved as dictionary in a pickle file
 - add all the words in `words` list, for which you want similar words
 - run python file as
`python3 CBOW.py`


For downloading the full project(containing pretrained models, dictionaries and code), go to this link - https://drive.google.com/drive/folders/14efZefkMV0SdBCHlN-4ugn7Vk-b1Q6kZ?usp=share_link
