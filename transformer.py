# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        """
        - vocab_index contains our vocab (lowercase letters and a space) where each char is mapped to an index
        - so input_indexed is just the input string mapped to their indices in vocab
        - output is what holds the result of the letter counting whether it be before-after or before counting (these are the ground truth counts)
        - we predict the counts using our transformer's forward and argmaxing the output probabilites
        """
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input]) 
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)        


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2) -> letter counting task.
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, task):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param task: either 'BEFORE' or 'BEFOREAFTER' depending on which letter counting task is being performed. BEFORE will add positional encoding to char embeddings. BEFOREAFTER does NOT add positional encoding.
        """
        super().__init__()
        #raise Exception("Implement me")  
        self.task = task      
        self.char_emb = nn.Embedding(vocab_size, d_model) # embedding layer for input chars
        self.pos_enc = PositionalEncoding(d_model, num_positions, batched=True)
        # make sure to create separate instances of TransformerLayer for each layer as each has its own weights that need to be updated
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes) # output layer to predict 0, 1, or 2
        self.log_softmax = nn.LogSoftmax(dim=-1) # log softmax to get log probabilities over classes
        nn.init.xavier_uniform_(self.char_emb.weight) # initialize embedding weights
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, indices):
        """

        :param indices: list of input indices (Torch Tensor)
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # see example of handling return of this forward in decode method
        # attention map is result of Q dot K^T / sqrt(d_k) after softmax
        # list of attention maps, is the result from each transformer layer
        # indices is the index of each char in the input sequence from our Indexer's vocab 
        # (data training on is already fixed length so dont have to do any padding)
        
        # add batch dimension if not present
        if indices.ndim == 1:
            indices = indices.unsqueeze(0) # (1, T) , T is seq_len

        # NOTE: this letter counting task doesnt need to add start of sequence token (BOS)

        attn_maps = []

        # embedding layer
        embedded = self.char_emb(indices)
        # print(self.task)
        if self.task == 'BEFORE':
            # add positional encodings to char embeddings
            pos_encoded = self.pos_enc(embedded) 
        else:
            # no positional encoding for BEFOREAFTER task
            pos_encoded = embedded 
        x = pos_encoded

        for layer in self.transformer_layers:
            x, attn_map = layer(x)            
            # attn_map are the attention maps from this layer transformer layer and for all samples in the batch (so shape (B, T, T))
            # decode method is only part of code that uses and plots attention maps, but method does not call transformer with batching (batch size 1)
            # and the plot imshow expects 2D array so need to remove batch dimension, when we know batch size is 1 (this is when decode calls forward)
            if attn_map.shape[0] == 1: # batch size 1
                attn_maps.append(attn_map.squeeze(0)) # remove batch dimension for plotting in decode
            else:
                attn_maps.append(attn_map) # keep batch dimension

        logits = self.output_layer(x) # (B, T, num_classes) -> (batch size, seq len, num classes)
        log_probs = self.log_softmax(logits)

        # batch size 1, remove batch dim in return -> bc decode method assumes no batching and has hard coded how it handles the output (cannot modify decode method)
        if indices.shape[0] == 1:
            return log_probs.squeeze(0), attn_maps
        return log_probs, attn_maps

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length. -> ignore, its up to us how to implement
        """
        super().__init__()
        
        # NOTE: 
        # BEFORE-AFTER counting task requires looking at previous locations and future locations, so first implement bidirectional attention (no mask)
        # BEFORE counting task requires looking at only previous locations, so implement causal mask (mask out future locations)
        # we're graded on before task so will need to implement causal mask but can implement bidirectional first to get things working
                
        """
        lecture 5.0.0 for architecture reference
            - in multi head attention, you want query and key dim d_k and value dim d_v after their linear layers to be d_model / num_heads, but here num_heads = 1 so d_k = d_v = d_model
            - HERE DOING SINGLE-HEAD ATTENTION
            - d_internal refers to the linear in the FFN after attention (like dim_feedforward in torch nn.Transformer)
            - when instantiating transformer, make sure d_internal > d_model
        """
        # self-attention
        self.d_model = d_model # embedding dim
        self.d_k = d_model # single head
        self.d_v = d_model
        self.linear_q = nn.Linear(d_model, self.d_k)
        self.linear_k = nn.Linear(d_model, self.d_k)
        self.linear_v = nn.Linear(d_model, self.d_v)
        self.layer_norm_attn = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        # feedforward network       
        self.d_internal = d_internal # FFN
        self.linear1_FFN = nn.Linear(d_model, d_internal)
        self.linear2_FFN = nn.Linear(d_internal, d_model)
        self.relu_FFN = nn.ReLU()        
        self.layer_norm_ffn = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.linear1_FFN.weight)
        nn.init.xavier_uniform_(self.linear2_FFN.weight)
        

    def forward(self, input_vecs):
        # input_vecs is [batch_dim, seq len, embedding dim] -> embeddings of each char in input sequence
        # if no batch dim, add batch size of 1
        if len(input_vecs.shape) == 2:
            input_vecs = input_vecs.unsqueeze(0) # (1, T, d_model)
            
        Q = self.linear_q(input_vecs) # (B, T, d_k) -> B is batch_size, T is seq_len
        K = self.linear_k(input_vecs) # (B, T, d_k)
        V = self.linear_v(input_vecs) # (B, T, d_v)
        # single-head attention
        # NOTE: use matmul bc we have batch dimension (dot prod of 3D tensors)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k) # (B, T, T)
        
        # Before thought had to use causal mask for BEFORE task but actually does not
        # if TASK == 'BEFORE':
        #     # causal mask to prevent attention to future positions
        #     T = Q.shape[1] # seq_len
        #     mask = torch.triu(torch.ones(T, T), diagonal=1).bool() # upper triangular matrix with 1s above diagonal
        #     """
        #     masked_fill(mask, value) returns a tensor where, for every position where mask is True, the corresponding element in the input tensor is replaced by value.
        #     mask must be boolean (torch.bool)
        #     unsqueeze(0) to make mask (1, T, T) so it broadcasts across batch dimension
        #     """
        #     scores = scores.masked_fill(mask.unsqueeze(0), float('-inf')) # (B, T, T) mask out future positions   
          
        attn_map = torch.softmax(scores, dim=-1) # (B, T, T)
        attn_output = torch.matmul(attn_map, V) # (B, T, d_v)
        # Possibly have a linear layer here but may only be if doing multi-head attention, here doing single-head attention so skip
        # add & norm
        attn_output = attn_output + input_vecs # residual connection
        attn_output = self.layer_norm_attn(attn_output)
        # feedforward network
        ffn_output = self.linear1_FFN(attn_output) # (B, T, d_internal)
        ffn_output = self.relu_FFN(ffn_output)
        ffn_output = self.linear2_FFN(ffn_output) # (B, T, d_model)
        # add & norm
        ffn_output = ffn_output + attn_output # residual connection
        ffn_output = self.layer_norm_ffn(ffn_output)
        return (ffn_output, attn_map) # return (B, T, d_model) and (B, T, T) attention map for this layer
    


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model) # create embeddings for positions
        self.batched = batched       
        nn.init.xavier_uniform_(self.emb.weight) # initialize positional embedding weights
        #nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)


    # call this class to add positional encodings to the char embeddings (x is the char embeddings)
    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor).to(x.device)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # can hardcode vocab_size bc know before hand that only lowercase letters and space (26+1) are present in data
    # and num_positions=20 bc training and testing data samples already formatted to be length 20 sequences
    # d_internal should be > d_model as d_internal corresponds to dim_feedforward
    # 3-class classification task (with labels 0, 1, or > 2 which we’ll just denote as 2) -> num_classes=3
    #d_model=512
    #d_internal=2048
    model = Transformer(vocab_size=27, num_positions=20, d_model=128, d_internal=256, num_classes=3, num_layers=2, task=args.task)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
    # we can optionally choose to train in batches (look at assignment 2 for batching example)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        # Batching
        batch_size = 32 # training set has 10k samples, test set has 1k samples
        batches = [ex_idxs[i:i + batch_size] for i in range(0, len(ex_idxs), batch_size)]
        for batch in batches:
            optimizer.zero_grad()
            batch_inputs = torch.stack([train[i].input_tensor for i in batch]) # (B, T)
            batch_targets = torch.stack([train[i].output_tensor for i in batch]) # (B, T)
            log_probs, _ = model(batch_inputs) # (B, T, num_classes)
            log_probs_reshaped = log_probs.view(-1, 3) # (B*T, num_classes)
            batch_targets_reshaped = batch_targets.view(-1) # (B*T,)
            loss_fcn = nn.NLLLoss()
            loss = loss_fcn(log_probs_reshaped, batch_targets_reshaped)
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("Epoch %i loss %f" % (t, loss_this_epoch))
        # loss_fcn = nn.NLLLoss()
        # for ex_idx in ex_idxs:
        #     loss = loss_fcn(...) # TODO: Run forward and compute loss
        #     # model.zero_grad()
        #     # loss.backward()
        #     # optimizer.step()
        #     loss_this_epoch += loss.item()
        
    # testing    
    model.eval()
    with torch.no_grad():
        loss_this_dev = 0.0
        for ex in dev:
            log_probs, _ = model(ex.input_tensor.unsqueeze(0)) # (1, T, num_classes)
            # no batching here so log_probs is (T, num_classes) after squeezing batch dim in forward
            # log_probs_reshaped = log_probs.view(-1, 3) # (T, num_classes)
            ex_output_reshaped = ex.output_tensor.view(-1) # (T,)
            loss_fcn = nn.NLLLoss()
            #loss = loss_fcn(log_probs_reshaped, ex_output_reshaped)
            loss = loss_fcn(log_probs, ex_output_reshaped)
            loss_this_dev += loss.item()
        print("Dev loss %f" % loss_this_dev)
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
