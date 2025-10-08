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
        # make sure here to be in eval mode and detach from gpu (place back onto cpu) and convert to numpy array
        self.model.eval()
        device = next(self.model.parameters()).device
        context_indices = [self.vocab_index.index_of(c) for c in context]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).to(device) # (seq_len,)
        with torch.no_grad():
            # print("Context tensor:", context_tensor)
            logits = self.model(context_tensor) # (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :] # (vocab_size,)
            log_probs = torch.log_softmax(last_logits, dim=0) # (vocab_size,)
        return log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        # Compute log P(next_chars | context) sequentially by querying one-step
        # next-character probabilities. This avoids issues with truncation when
        # the combined length exceeds the model's seq_len and ensures consistency
        # with get_next_char_log_probs.
        self.model.eval()
        total_log_prob = 0.0
        cur_context = context
        # if next_chars is a string, 
        for c in next_chars:
            # print("Current c: ", c)
            # print("Current context: ", cur_context)
            logp = self.get_next_char_log_probs(cur_context)  # numpy array of log-probs
            char_idx = self.vocab_index.index_of(c)
            total_log_prob += float(logp[char_idx])
            cur_context = cur_context + c
        return total_log_prob
    
import math 
def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc

class TransformerNextToken(torch.nn.Module):
    """
    Implement transformer using torch nn.TransformerEncoder and nn.TransformerEncoderLayer for next-token predicting task. Now given long continous sequence of characters and need to split into chunks of max length. Use causal mask and positional encoding from part 1.
    """
    def __init__(self, vocab_size: int=27, d_model: int=512, nhead: int=8, num_layers: int=6, dim_feedforward: int=2048, seq_len: int=20, vocab_index: Indexer=None, pad_token: str='#'):
        """
        :param vocab_size: size of character vocabulary (27)
        :param d_model: embedding dimension (512)
        :param nhead: number of attention heads (8)
        :param num_layers: number of transformer encoder layers (6)
        :param dim_feedforward: dimension of feedforward layer (2048)
        :param seq_len: maximum sequence length (20)
        :param vocab_index: an Indexer of the character vocabulary (27 characters)
        :param pad_token: the padding token string
        """
        super().__init__()        
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=vocab_index.index_of(pad_token)) # last index is PAD token
        
        # IMPORTANT: we want to add the padding token to our embedding but not our vocabulary Indexer
        # self.char_embedding = nn.Embedding(vocab_size+1, d_model, padding_idx=vocab_size) 
        self.positional_encoding = PositionalEncoding(d_model, seq_len, batched=True)
        self.transformer_encoder = nn.TransformerEncoder(
            # important to set batch_first=True so input is (batch_dim, seq_len, embed_dim)
            # norm_first=True from deep learning course, showed to improve results
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, norm_first=False, batch_first=True), 
            num_layers
        )
        # map logits to vocab size for next token prediction
        # self.linear_out = nn.Linear(d_model, vocab_size+1) 
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
        
        """
        Below handles cases where input sequence length is not equal to model's expected seq_len (mainly for inferencing as training before calling forward will handle this)
        """
        # if input sequence is longer than seq_len, truncate to last seq_len tokens
        if indices.shape[1] > self.seq_len:
            indices = indices[:, -self.seq_len:] # (batch_size, seq_len)
        if indices.shape[1] < self.seq_len:
            # pad with PAD token index at the beginning if sequence is shorter than seq_len
            pad_length = self.seq_len - indices.shape[1]
            pad_idx = self.char_embedding.padding_idx
            pad_tensor = torch.full((indices.shape[0], pad_length), pad_idx, dtype=torch.long, device=indices.device) # (batch_size, pad_length)
            indices = torch.cat([pad_tensor, indices], dim=1) # (batch_size, seq_len)
            # print("\n\nPADDED INDICES:", indices)
        
        # print("\n\nBEFORE SHIFT INDICES:", indices)
        # shift tokens right by 1 position for next token prediction
        indices = torch.roll(indices, shifts=1, dims=1) 
        # print("\n\nAFTER SHIFT INDICES:", indices)
        
        # replace first position with BOS token (here is ' ' space) to each sequence in batch
        space_idx = self.vocab_index.index_of(' ')
        indices[:, 0] = space_idx
        # print("\n\nAFTER INSERT BOS INDICES:", indices)
        # NOTE: we do above shifting and BOS token insertion before embedding bc the BOS token is part of the vocab and has a learned embedding
        x = self.char_embedding(indices)
        # print("\n\nCHAR EMBEDDING OUTPUT:", x)
        # print(x.shape)
        x = self.positional_encoding(x) # adds positional encoding to char embeddings
        #x = x + get_positional_encoding(self.seq_len, x.shape[-1]).to(x.device) 
        mask=torch.nn.Transformer.generate_square_subsequent_mask(sz=self.seq_len).to(x.device) # (seq_len, seq_len) causal mask
        x = self.transformer_encoder(x, mask=mask, is_causal=True) # (batch_size, seq_len, d_model)
        x = self.linear_out(x) # (batch_size, seq_len, vocab_size)
        return x

def text_to_indices(text: str, vocab_index) -> List[int]:
    """Convert string to list of token indices using vocab_index.index_of(c)."""
    return [vocab_index.index_of(c) for c in text]

def chunk_non_overlapping(indices: List[int], seq_len: int,
                          drop_last: bool = True,
                          random_offset: bool = False) -> List[List[int]]:
    """
    Partition indices into non-overlapping chunks of length seq_len.
    If random_offset is True pick a start in [0, seq_len-1] each call (useful per-epoch).
    If drop_last is False, the final short chunk is left-padded (requires pad_idx provided externally).
    Returns list of chunks (each length <= seq_len). If you want guaranteed length seq_len,
    either drop_last=True or post-pad/pad-left externally.
    """
    if random_offset:
        start = random.randint(0, seq_len - 1)
    else:
        start = 0
    chunks = []
    for i in range(start, len(indices), seq_len):
        chunk = indices[i:i+seq_len]
        if len(chunk) < seq_len and drop_last:
            break
        chunks.append(chunk)
    return chunks

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # add padding token to vocab
    pad_token = '#'
    
    vocab_index.add_and_get_index(pad_token) # DO NOT ADD PAD TOKEN TO VOCAB INDEXER, in lm.py, the normalization_test for get_log_prob_sequence relies on vocab_index being only the 27 characters, by adding PAD token, it till attempt to predict using the pad token and got unsolvable error
    # perplexity 33
    
    # perplexity=31
    # seq_len = 100
    # nhead=4
    # dim_feedforward=512
    # d_model=256
    # num_layers=4
    
    seq_len = 20
    nhead=8
    dim_feedforward=2048
    d_model=512
    num_layers=6
    
    model = TransformerNextToken(nhead=nhead, dim_feedforward=dim_feedforward,
                                 d_model=d_model, num_layers=num_layers, vocab_size=len(vocab_index), vocab_index=vocab_index, pad_token=pad_token, seq_len=seq_len)
    model.zero_grad()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.char_embedding.padding_idx) # applies softmax internally
    
    
    train_indices = text_to_indices(train_text, vocab_index)
    train_chunks = chunk_non_overlapping(train_indices, seq_len=model.seq_len, drop_last=True, random_offset=False)
    
    num_epochs = 10
    batch_size = 32
    for epoch in range(0, num_epochs):
        print("Starting epoch %i" % (epoch))
        loss_this_epoch = 0.0
        #random_start = random.randint(0, model.seq_len - 1) # to get different chunks each epoch
        # chunk train text into sequences of length seq_len
        # train_indices = [vocab_index.index_of(c) for c in train_text]
        #train_chunks = [train_indices[i:i+model.seq_len] for i in range(random_start, len(train_indices)-model.seq_len, model.seq_len)]
        # train_chunks = [train_indices[i:i+model.seq_len] for i in range(0, len(train_indices), model.seq_len)        
       
        random.shuffle(train_chunks)
        for b in range(0, len(train_chunks), batch_size):
            batch_chunks = train_chunks[b:b+batch_size]
            batch_tensor = torch.tensor(np.array(batch_chunks), dtype=torch.long).to(device) # (batch_size, seq_len)
            logits = model(batch_tensor) # (batch_size, seq_len, vocab_size)
            # Collapse batch dim for cross entropy loss funct
            logits = logits.reshape(-1, logits.shape[-1]) # (batch_size * seq_len, vocab_size)
            targets = batch_tensor.reshape(-1) # (batch_size * seq_len,) 
            # Ground truth targets are the token index (from vocab) expected at each position
            # print("\nLOSS LOGITS argmax:", nn.functional.softmax(logits, dim=-1).argmax(dim=-1))
            # print("\nLOSS TARGETS:", targets)

            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_this_epoch += loss.item()
        print("Epoch %i training loss: %f" % (epoch, loss_this_epoch))
        
    # testing    
    model.eval() 
    # with torch.no_grad():
    #     loss_this_dev = 0.0
    #     # chunk dev text into sequences of length seq_len
    #     dev_indices = [vocab_index.add_and_get_index(c) for c in dev_text]
    #     dev_chunks = [dev_indices[i:i+model.seq_len] for i in range(0, len(dev_indices)-model.seq_len, model.seq_len)]
    #     for b in range(0, len(dev_chunks), batch_size):
    #         batch_chunks = dev_chunks[b:b+batch_size]
    #         batch_tensor = torch.tensor(batch_chunks, dtype=torch.long).to(device) # (batch_size, seq_len)
    #         logits = model(batch_tensor) # (batch_size, seq_len, vocab_size)
    #         logits = logits.reshape(-1, logits.shape[-1]) 
    #         targets = batch_tensor.reshape(-1)
    #         loss = loss_fn(logits, targets)
    #         loss_this_dev += loss.item()
    #     print("Epoch %i dev loss: %f" % (epoch, loss_this_dev))
    return NeuralLanguageModel(model, vocab_index)