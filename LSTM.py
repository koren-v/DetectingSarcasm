import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, num_layers, weights):
		super(LSTMClassifier, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.num_layers = num_layers
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers, batch_first = True, bidirectional=True)
		self.label = nn.Linear(hidden_size*2, output_size) #*2 for bidirect
		
	def forward(self, input_sentence, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		if batch_size is None:
		  h_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM, *2 for biderection
		  c_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM, *2 for biderection
		else:
			h_0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
		final_output = self.label(output[:, -1, :]) # Last output which is the same as final_hidden_state

		return final_output