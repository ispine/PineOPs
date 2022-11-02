import torch
import torch.nn as nn
import torch.optim  as optim
import time

from models.models import sentiment_model

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.required_grad)

def binary_accuracy(preds, y):
	rounded_preds = torch.rount(torch.sigmoid(preds))
	correct = (rounded_preds == y).float()
	acc = correct.sum() / len(correct)

	return acc

def train_model(model, iterator, optimizer, criterion):
	epoch_loss, epoch_acc = 0, 0
	
	model.train()
	for d_iter in iterator:
		batch = d_iter[0]
		label = d_iter[1]
		optimizer.zero_grad()

		predictions = model(batch).squeeze(1)

		try:
			loss = criterion(predictions, label)
		except: import pdb; pdb.set_trace()
		acc = binary_accuracy(predictions, label)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator), model

def train(iterator, epochs, input_dim=30000, embedding_dim=100, hidden_size=256, output_dim=1):
	
	model = sentiment_model(input_dim, embedding_dim, hidden_size, output_dim)
	optimizer = optim.SGD(model.parameters(), lr=1e-3)
	criterion = nn.BCEWithLogitsLoss()

	for epoch in range(epochs):
		start_time = time.time()
		
		train_loss, train_acc, model = train_model(model, iterator, optimizer, criterion)
		
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
	#torch.save(model.state_dict(), 
	
	return train_loss, train_acc

def evaluate(model, iterator, criterion):
	eopch_loss, epoch_acc = 0, 0

	model.eval()
	with torch.no_grad():
		for d_iter in iterator:
			batch = d_iter[0]
			label = d_iter[1]
			predictions = model(batch).squeeze(1)

			loss = criterion(predicitions, label)
			acc = binary_accuracy(predictions, label)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)
