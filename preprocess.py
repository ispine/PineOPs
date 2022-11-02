import numpy as np
import pandas as pd
import json
import tensorflow as tf
import tqdm
import torch
from torchtext import data
from torch.utils.data import DataLoader, Dataset

from collections import Counter
from sklearn.model_selection import train_test_split

from konlpy.tag import Komoran

class sentiment_dataset(Dataset):
	def __init__(self, inputs, labels):
		self.inputs = inputs
		self.labels = labels

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, index):
		inputs = self.inputs[index]
		labels = self.labels[index]
		
		return inputs, labels

def load_data(train_path, test_path):
  train = pd.read_table(train_path)
  test = pd.read_table(test_path)

  return train, test

def make_vocab(words):
	counter = Counter(words)
	counter = counter.most_common(30000-4)
	vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]

	with open('./nsmc/vocab.txt', 'w') as w_file:
		for w in vocab:
			w_file.write(str(w)+'\n')
	
	return vocab
	

def preprocess(train_path, test_path, batch_size):
	train, test = load_data(train_path, test_path)
	tokenizer = Komoran()

	train.drop_duplicates(subset=['document'], inplace=True)
	test.drop_duplicates(subset=['document'], inplace=True)
	train, test = train.dropna(), test.dropna()

	print(f"train shape : {train.shape}\ntest shape : {test.shape}")

	# pos tagging
	train_tokenized = [[token+"/"+POS for token, POS in tokenizer.pos(doc_)] for doc_ in train['document']]
	test_tokenized = [[token+"/"+POS for token, POS in tokenizer.pos(doc_)] for doc_ in test['document']]

	# remove stopword(. , ! suffix etc)
	exclusion_tags = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'EF',
										'EP', 'EC', 'ETN', 'ETM', 'XSN', 'XSV', 'XSA']
	f = lambda x: x in exclusion_tags
	
	X_train = []
	for i in range(len(train_tokenized)):
		temp = []
		for j in range(len(train_tokenized[i])):
			if f(train_tokenized[i][j].split('/')[1]) is False:
				temp.append(train_tokenized[i][j].split('/')[0])
		X_train.append(temp)

	X_test = []
	for i in range(len(test_tokenized)):
		temp = []
		for j in range(len(test_tokenized[i])):
			if f(test_tokenized[i][j].split('/')[1]) is False:
				temp.append(test_tokenized[i][j].split('/')[0])
		X_test.append(temp)

	# make vocab
	is_vocab = False
	words = np.concatenate(X_train).tolist()
	if is_vocab:
		vocab = open('./nsmc/vocab.txt', 'r')
	else: vocab = make_vocab(words)
	word_to_index = {word:index for index, word in enumerate(vocab)}
	index_to_word = {index:word for word, index in word_to_index.items()}

	def wordlist_to_indexlist(wordlist):
		return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]
	
	X_train = list(map(wordlist_to_indexlist, X_train))
	X_test = list(map(wordlist_to_indexlist, X_test))
	
	# make the pad
	X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='pre', value=word_to_index["<PAD>"], maxlen=70)
	X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='pre', value=word_to_index["<PAD>"], maxlen=70)

	X_train, X_test = torch.tensor(X_train, dtype=torch.long), torch.tensor(X_test, dtype=torch.long)
	#train_iterator, test_iterator = data.BucketIterator.splits((X_train, X_test), batch_size=batch_size)
	train_data, test_data = sentiment_dataset(X_train, train['label']), sentiment_dataset(X_test, test['label'])
	train_iter, test_iter = DataLoader(train_data, batch_size=batch_size), DataLoader(test_data, batch_size=batch_size)
	return train_iter, test_iter, word_to_index, index_to_word, vocab

if __name__=="__main__":
	train_path = './nsmc/ratings_train.txt'
	test_path = './nsmc/ratings_test.txt'
	
	batch_size = 8
	train_iter, test_iter, wti, itw, vocab = preprocess(train_path, test_path, batch_size)
	for train_diter in train_iter:
		batch = train_diter[0]
		label = train_diter[1]
