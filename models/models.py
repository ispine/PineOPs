import torch
import torch.nn as nn

class sentiment_model(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
		super().__init__()
		
		try:
			self.embedding = nn.Embedding(input_dim, embedding_dim)
		except: import pdb; pdb.set_trace()
		self.rnn = nn.RNN(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, inputs):
		embedded = self.embedding(inputs)
		output, hidden = self.rnn(embedded)

		return self.fc(hidden.squeeze(0))
