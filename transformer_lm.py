# models.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")
        # produce prob distribution over all possible next chars in vocab given context


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")
        # assume next_chars is list of chars to find log prob of given context, but can also be list length 1


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")
    
    
class TransformerNextToken(torch.nn.Module):
    """
    Implement transformer using torch nn.TransformerEncoder and nn.TransformerEncoderLayer for next-token predicting task. Now given long continous sequence of characters and need to split into chunks of max length. Use causal mask and positional encoding from part 1.
    """
    def __init__(self, vocab_size: int=27, d_model: int=512, nhead: int=8, num_layers: int=6, dim_feedforward: int=2048, seq_len: int=20, vocab_index: Indexer=None, pad_token: str='PAD'):
        """
        :param vocab_size: size of character vocabulary (27)
        :param d_model: embedding dimension (512)
        :param nhead: number of attention heads (8)
        :param num_layers: number of transformer encoder layers (6)
        :param dim_feedforward: dimension of feedforward layer (2048)
        :param seq_len: maximum sequence length (20)
        :param vocab_index: an Indexer of the character vocabulary (27 characters)
        :param pad_token: the padding token string (PAD)
        """
        super().__init__()        
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=vocab_index.index_of(pad_token)) # last index is PAD token
        self.positional_encoding = PositionalEncoding(d_model, seq_len, batched=True)
        self.transformer_encoder = nn.TransformerEncoder(
            # important to set batch_first=True so input is (batch_dim, seq_len, embed_dim)
            # norm_first=True from deep learning course, showed to improve results
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, norm_first=True, batch_first=True), 
            num_layers
        )
        # map logits to vocab size for next token prediction
        self.linear_out = nn.Linear(d_model, vocab_size) 
        self.seq_len = seq_len
        self.vocab_index = vocab_index
        nn.init.xavier_uniform_(self.char_embedding.weight)
        # ensure padding token weights is zero after intialization to prevent it from being updated during training
        with torch.no_grad():
            self.char_embedding.weight[self.char_embedding.padding_idx].zero_()
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, indices):
        """
        :param indices: input token indices (batch_size, seq_len)
        :return: logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        # add batch dimension if not present
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(0) # (1, seq_len)
            
        # if input sequence is longer than seq_len, truncate to last seq_len tokens
        if indices.shape[1] > self.seq_len:
            indices = indices[:, -self.seq_len:] # (batch_size, seq_len)
        if indices.shape[1] < self.seq_len:
            # pad with PAD token index at the beginning if sequence is shorter than seq_len
            pad_length = self.seq_len - indices.shape[1]
            pad_idx = self.char_embedding.padding_idx
            pad_tensor = torch.full((indices.shape[0], pad_length), pad_idx, dtype=torch.long, device=indices.device) # (batch_size, pad_length)
            indices = torch.cat([pad_tensor, indices], dim=1) # (batch_size, seq_len)
            
        # shift tokens right by 1 position for next token prediction
        indices = torch.roll(indices, shifts=1, dims=1)
        
        # replace first position with BOS token (here is ' ' space) to each sequence in batch
        space_idx = self.vocab_index.index_of(' ')
        indices[:, 0] = space_idx
        # NOTE: we do above shifting and BOS token insertion before embedding bc the BOS token is part of the vocab and has a learned embedding
        x = self.char_embedding(indices)
        x = self.positional_encoding(x) # adds positional encoding to char embeddings
        mask=torch.nn.Transformer.generate_square_subsequent_mask(sz=self.seq_len).to(x.device) # (seq_len, seq_len) causal mask
        x = self.transformer_encoder(x, mask=mask, is_causal=True) # (batch_size, seq_len, d_model)
        x = self.linear_out(x) # (batch_size, seq_len, vocab_size)
        return x


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # add padding token to vocab
    pad_token = 'PAD'
    seq_len = 20
    vocab_index.add_and_get_index(pad_token)
    
    model = TransformerNextToken(vocab_size=len(vocab_index), vocab_index=vocab_index, pad_token=pad_token, seq_len=seq_len)
    model.zero_grad()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss() # applies softmax internally
    
    num_epochs = 10
    batch_size = 64
    for epoch in range(0, num_epochs):
        print("Starting epoch %i" % (epoch))
        loss_this_epoch = 0.0
        random_start = random.randint(0, model.seq_len - 1) # to get different chunks each epoch
        # chunk train text into sequences of length seq_len
        train_indices = [vocab_index.add_and_get_index(c) for c in train_text]
        train_chunks = [train_indices[i:i+model.seq_len] for i in range(random_start, len(train_indices)-model.seq_len, model.seq_len)]
        random.shuffle(train_chunks) # shuffle chunks each epoch
        # mini-batch training
        for b in range(0, len(train_chunks), batch_size):
            batch_chunks = train_chunks[b:b+batch_size]
            batch_tensor = torch.tensor(batch_chunks, dtype=torch.long).to(device) # (batch_size, seq_len)
            logits = model(batch_tensor) # (batch_size, seq_len, vocab_size)
            # reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len,) for cross entropy loss
            logits = logits.reshape(-1, logits.shape[-1]) 
            targets = batch_tensor.reshape(-1)
            loss = loss_fn(logits, targets, ignore_index=model.char_embedding.padding_idx)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_this_epoch += loss.item()
        print("Epoch %i training loss: %f" % (epoch, loss_this_epoch))
        
    # testing    
    model.eval() # also model.eval() to determinize inference (turns off dropout layers in TransformerEncoder)
    with torch.no_grad():
        loss_this_dev = 0.0
        # chunk dev text into sequences of length seq_len
        dev_indices = [vocab_index.add_and_get_index(c) for c in dev_text]
        dev_chunks = [dev_indices[i:i+model.seq_len] for i in range(0, len(dev_indices)-model.seq_len, model.seq_len)]
        for b in range(0, len(dev_chunks), batch_size):
            batch_chunks = dev_chunks[b:b+batch_size]
            batch_tensor = torch.tensor(batch_chunks, dtype=torch.long).to(device) # (batch_size, seq_len)
            logits = model(batch_tensor) # (batch_size, seq_len, vocab_size)
            logits = logits.reshape(-1, logits.shape[-1]) 
            targets = batch_tensor.reshape(-1)
            loss = loss_fn(logits, targets, ignore_index=model.char_embedding.padding_idx)
            loss_this_dev += loss.item()
        print("Epoch %i dev loss: %f" % (epoch, loss_this_dev))
    return NeuralLanguageModel(model, vocab_index)