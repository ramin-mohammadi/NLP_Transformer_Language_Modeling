# models.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
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
        # make sure here to be in eval mode, detach from gpu (place back onto cpu) and convert to numpy array
        self.model.eval()
        device = next(self.model.parameters()).device
        """
        Note edge cases of context (input to model)
        - If context is empty, we use a space as the initial context.
        - If len(context) > seq_len, we truncate to the last seq_len characters.
            - handled in preprocess_input
        - If len(context) < seq_len, no need to pad since we are using dynamic masking (mask is size of input sequence length)
        """
        if context == "":
            context = " "
        context_indices = [self.vocab_index.index_of(c) for c in context] # (seq_len,)
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device) # (1, seq_len)

        with torch.no_grad():
            # NOTE: do not shift or add BOS token during inferencing so specify training=False
            context_tensor = self.model.preprocess_input(context_tensor, training=False) # (1, seq_len)
            logits = self.model(context_tensor) # (1, seq_len, vocab_size), model outputs log-probabilities already (log softmax of logits)
            log_probs = logits[0, -1, :]  # (vocab_size,) , acquire log-probs for last position in sequence
        return log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        total_log_prob = 0.0
        cur_context = context
        
        if context == "":
            cur_context = " "
        
        for c in next_chars:
            logp = self.get_next_char_log_probs(cur_context) # (vocab_size,)
            char_idx = self.vocab_index.index_of(c)
            total_log_prob += float(logp[char_idx])
            cur_context = cur_context + c
        return total_log_prob


class TransformerNextToken(torch.nn.Module):
    """
    Transformer Encoder for next-token predicting task utilizing causal mask and PositionalEncoding.
    """
    def __init__(self, seq_len: int=20, d_model: int=512, nhead: int=8, num_layers: int=6, 
                 dim_feedforward: int=2048,  vocab_index: Indexer=None):
        """        
        :param seq_len: maximum sequence length
        :param d_model: embedding dimension 
        :param nhead: number of attention heads
        :param num_layers: number of transformer encoder layers
        :param dim_feedforward: dimension of feedforward layer
        :param vocab_index: Indexer for the character vocabulary (26 lowercase letters + space = 27)
        """
        super().__init__()   
        self.seq_len = seq_len
        self.vocab_index = vocab_index
        vocab_size = len(vocab_index) 
           
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, batched=True)           
        # important to set batch_first=True since input is (batch_dim, seq_len, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, norm_first=False, batch_first=True), 
            num_layers
        )
        self.linear_out = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1) # NLLLoss expects log-probabilities as input
        # if using CrossEntropyLoss, do not need perform log softmax at end of model's forward since the loss applies log softmax internally
        nn.init.xavier_uniform_(self.char_embedding.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
    
    def preprocess_input(self, indices: torch.Tensor, training: bool=False):
        # add batch dimension if not present
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(0) # (1, seq_len)
        # if input sequence is longer than seq_len, truncate to last seq_len tokens
        if indices.shape[1] > self.seq_len:
            indices = indices[:, -self.seq_len:] # (batch_size, seq_len)
            
        # SHIFT TOKEN OVER 1 LOCATION AND ADD BOS TOKEN ONLY DURING TRAINING, NOT DURING INFERENCING
        # NO LONGER NEED TO PAD BC using dynamic masking (mask is size of input sequence length)
        if training:
            # shift tokens right by 1 position for next token prediction
            indices = torch.roll(indices, shifts=1, dims=1) 
            # replace first position with BOS token (here is ' ' space) to each sequence in the batch
            indices[:, 0] = self.vocab_index.index_of(' ')
        return indices # (batch_size, seq_len)

    def forward(self, indices: torch.Tensor):
        """
        :param indices: input token indices that have been preprocessed (batch_size, seq_len)
        :return: log-probs for next token prediction (batch_size, seq_len, vocab_size)
        """
        # NOTE: we add the BOS token to indices (in preprocess_input method) before embedding layer 
        # bc the BOS token is part of the vocab so has a learned representation
        x = self.char_embedding(indices)
        x = self.positional_encoding(x) # adds positional encoding to char embeddings (see transformer.py)
        
        """
        make mask dynamic so can avoid having to pad by using the input sequence length rather than model's seq_len 
        - embedding layer simply converts given indices to their embedding vector representation
        - only layer seq_len matters is for PositionalEncoding
        - so for positional encoding's embedding layer, there isnt a problem if input sequence length is shorter than model seq_len 
           - seq_len is paramater of positional embedding layer but just specifies its lookup table size 
        - but input sequence length is a problem if larger than seq_len (outside of positional embedding's vocab) 
        where you would need to truncate input sequence to last seq_len tokens
        """
        input_seq_len = x.shape[1] 
        mask=torch.nn.Transformer.generate_square_subsequent_mask(sz=input_seq_len).to(x.device) # (input_seq_len, input_seq_len) causal mask
        x = self.transformer_encoder(x, mask=mask, is_causal=True) # (batch_size, seq_len, d_model)
        x = self.linear_out(x) # (batch_size, seq_len, vocab_size)
        x = self.log_softmax(x) # (batch_size, seq_len, vocab_size), log-probabilities for NLLLoss
        return x


def text_to_indices(text: str, vocab_index) -> List[int]:
    """Convert string to list of token indices using vocab_index.index_of(c)."""
    return [vocab_index.index_of(c) for c in text]

def sliding_window_chunks(indices: List[int], seq_len: int, stride: int = 1, drop_short: bool = True) -> List[List[int]]:
    """
    Create overlapping chunks using a sliding window.
    - stride < seq_len will create overlapping windows. stride = 1 is most extreme overlap (every sample is shifted by 1). 
    stride = seq_len is no overlap.
    - drop_short: if True drop the final partial window, otherwise include it as a shorter chunk.
    Returns a list of chunks (each of length seq_len, except possibly the final one when drop_short=False).
    """
    if len(indices) < seq_len:
        return [] if drop_short else [indices]
    chunks = [indices[i:i+seq_len] for i in range(0, len(indices) - seq_len + 1, stride)]
    if not drop_short:
        # include any trailing short tail
        tail_start = (len(indices) - seq_len + 1) + ((len(indices) - seq_len) % stride)
        tail = indices[tail_start + seq_len:]
        if tail:
            chunks.append(tail)
    return chunks

def print_text_chunks(chunks: List[List[int]], vocab_index):
    for i in range(len(chunks)):
        chunk = []
        for j in range(len(chunks[i])):
            chunk.append(vocab_index.get_object(chunks[i][j]))
        print(f"Chunk {i}: ", ''.join(chunk))


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args from lm.py
    :param train_text: train text as a sequence of characters (one list containing 100k chars)
    :param dev_text: dev text as a sequence of characters 
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    
    """
    Understanding regarding padding and using BOS token:
    - my implementation does not pad if input sequence to model is < model's seq_len
    - my model can handle sequence length from 1 up to seq_len 
    - (transformers in general are designed to be able to handle variable-length possible inputs in a certain range). 
    If your model can handle sequence length from 1 up to seq_len, then you don't need to pad at all.
    - We do not need to pad input sequences if we use dynamic masking (mask defined in forward is size of input sequence length)
      since the model will not attend to future tokens beyond the input sequence length. 
        - This is only a case we deal with during inferencing. We avoid this case during training by creating 
        our batches of chunks all with the model's seq_len.
        - if had to pad, make it right aligned and could just use space as the padding token
        - originally thought had to add a dedicated token for padding to vocab and be a part of our char 
        embedding and final linear layer logits of vocab_size -> ends up being noise so best to avoid
    - BOS token can be a space and does not have to be a different token than what is in our text corpus 
    (do not have to add and learn a unique BOS token). Space here is learned to be used as a BOS token and as a separator between words.
    # pad_token = '#'
    # vocab_index.add_and_get_index(pad_token)  
    # # BOS_token = '@'
    # BOS_token = ' '
    # vocab_index.add_and_get_index(BOS_token)
    """
    # TA approved hyperparameters: (smaller transformer here leads to better results, probably bc less overfitting)
    # seq_len=20
    # batch_size=100  # if batch size larger will need more epochs, with batch size 100, epochs=5 is good
    # nhead=2
    # d_model=64
    # dim_feedforward=d_model*4
    # num_layers=2
    # num_epochs=5
    
    seq_len=50
    batch_size=200
    nhead=2
    d_model=64
    dim_feedforward=128
    num_layers=2
    num_epochs=10

    model = TransformerNextToken(seq_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 dim_feedforward=dim_feedforward, vocab_index=vocab_index)
    model.zero_grad()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    loss_fn = nn.NLLLoss() # pass log-probabilities into NLL. With batching, make sure to collapse batch dim for log-probs and targets.
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # gamma is factor to multiply LR by
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    train_indices = text_to_indices(train_text, vocab_index)     
    #stride = max(1, seq_len // 2)
    #stride=seq_len # no overlap, basically no striding
    stride=1 # striding at extreme (every sample is just shifted by 1 (lots of overlapping)). Striding samples improved perplexity.
    # must drop_short for batching to work (cant have a sample with different seq_len when using batching)
    train_chunks = sliding_window_chunks(train_indices, seq_len=model.seq_len, stride=stride, drop_short=True) 

    for epoch in range(0, num_epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Starting epoch {epoch} - lr={cur_lr:.6e}")
        # track token-weighted losses so we can compute avg per-token loss and perplexity
        total_train_loss_tokens = 0.0
        total_train_tokens = 0

        random.shuffle(train_chunks)
        # NOTE: does not matter if chunk first then batch second or vice versa.
        # Actually easier if create chunks first then batch second similar to transformer.py and its given dataset already in chunks
        for b in range(0, len(train_chunks), batch_size):
            batch_chunks = train_chunks[b:b+batch_size]
            # make sure save different tensors for model input and target since preprocess_input here will shift tokens and add BOS token
            # dont want targets to be shifted too or model will be cheating -> will lead to perplexity of around 1 to 2
            # can see test in lm.py: error if perplexity < 3.5
            batch_tensor_targets = torch.tensor(np.array(batch_chunks), dtype=torch.long).to(device) # (batch_size, seq_len)
            batch_tensor_train = model.preprocess_input(batch_tensor_targets, training=True) # (batch_size, seq_len) with indices shifted and BOS token inserted at first position in each sequence
            logits = model(batch_tensor_train) # (batch_size, seq_len, vocab_size)
            # Collapse batch dim for NLL loss funct (would also have to do this if using cross entropy)
            logits = logits.reshape(-1, logits.shape[-1]) # (batch_size * seq_len, vocab_size)
            targets = batch_tensor_targets.reshape(-1) # (batch_size * seq_len,) 
            # Ground truth targets are the token index (from vocab indexer) expected at each position
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            # for tracking avg loss and perplexity each epoch
            n_tokens = len(targets)  # all tokens count since we are not using ignore_index in loss_fn
            if n_tokens > 0:
                total_train_loss_tokens += loss.item() * n_tokens
                total_train_tokens += n_tokens

        avg_train_loss = total_train_loss_tokens / max(1, total_train_tokens)
        train_ppl = float(np.exp(avg_train_loss))
        print(f"Epoch {epoch} training avg loss (nats/token): {avg_train_loss:.6f}, perplexity: {train_ppl:.4f}")

        # evaluate on dev set each epoch using same sliding window stride
        model.eval()
        with torch.no_grad():
            dev_indices = text_to_indices(dev_text, vocab_index)
            dev_chunks = sliding_window_chunks(dev_indices, seq_len=model.seq_len, stride=stride, drop_short=True)
            total_dev_loss_tokens = 0.0
            total_dev_tokens = 0
            for b in range(0, len(dev_chunks), batch_size):
                batch_chunks = dev_chunks[b:b+batch_size]
                batch_tensor_targets = torch.tensor(np.array(batch_chunks), dtype=torch.long).to(device)
                batch_tensor_train = model.preprocess_input(batch_tensor_targets, training=True) 
                logits = model(batch_tensor_train)
                logits = logits.reshape(-1, logits.shape[-1])
                targets = batch_tensor_targets.reshape(-1)
                loss = loss_fn(logits, targets)
                n_tokens = len(targets)
                if n_tokens > 0:
                    total_dev_loss_tokens += loss.item() * n_tokens
                    total_dev_tokens += n_tokens
        avg_dev_loss = total_dev_loss_tokens / max(1, total_dev_tokens)
        dev_ppl = float(np.exp(avg_dev_loss))
        print(f"Epoch {epoch} dev   avg loss (nats/token): {avg_dev_loss:.6f}, perplexity: {dev_ppl:.4f}")
        #scheduler.step() # update learning rate using schedueler each epoch (step per epoch)
        model.train()
    model.eval() 
    return NeuralLanguageModel(model, vocab_index)