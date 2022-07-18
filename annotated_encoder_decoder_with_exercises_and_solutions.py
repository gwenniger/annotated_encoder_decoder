# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Annotated Encoder-Decoder with Attention
#
# Recently, Alexander Rush wrote a blog post called [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html), describing the Transformer model from the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). This post can be seen as a **prequel** to that: *we will implement an Encoder-Decoder with Attention* using (Gated) Recurrent Neural Networks, very closely following the original attention-based neural machine translation paper ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) of Bahdanau et al. (2015). 
#
# The idea is that going through both blog posts will make you familiar with two very influential sequence-to-sequence architectures. If you have any comments or suggestions, please let me know: [@BastingsJasmijn](https://twitter.com/BastingsJasmijn).

# %% [markdown]
# # Model Architecture
#
# We will model the probability $p(Y\mid X)$ of a target sequence $Y=(y_1, \dots, y_{N})$ given a source sequence $X=(x_1, \dots, x_M)$ directly with a neural network: an Encoder-Decoder.
#
# <img src="images/bahdanau.png" width="636">
#
# #### Encoder 
#
# The encoder reads in the source sentence (*at the bottom of the figure*) and produces a sequence of hidden states $\mathbf{h}_1, \dots, \mathbf{h}_M$, one for each source word. These states should capture the meaning of a word in its context of the given sentence.
#
# We will use a bi-directional recurrent neural network (Bi-RNN) as the encoder; a Bi-GRU in particular.
#
# First of all we **embed** the source words. 
# We simply look up the **word embedding** for each word in a (randomly initialized) lookup table.
# We will denote the word embedding for word $i$ in a given sentence with $\mathbf{x}_i$.
# By embedding words, our model may exploit the fact that certain words (e.g. *cat* and *dog*) are semantically similar, and can be processed in a similar way.
#
# Now, how do we get hidden states $\mathbf{h}_1, \dots, \mathbf{h}_M$? A forward GRU reads the source sentence left-to-right, while a backward GRU reads it right-to-left.
# Each of them follows a simple recursive formula: 
# $$\mathbf{h}_j = \text{GRU}( \mathbf{x}_j , \mathbf{h}_{j - 1} )$$
# i.e. we obtain the next state from the previous state and the current input word embedding.
#
# The hidden state of the forward GRU at time step $j$ will know what words **precede** the word at that time step, but it doesn't know what words will follow. In contrast, the backward GRU will only know what words **follow** the word at time step $j$. By **concatenating** those two hidden states (*shown in blue in the figure*), we get $\mathbf{h}_j$, which captures word $j$ in its full sentence context.
#
#
# #### Decoder 
#
# The decoder (*at the top of the figure*) is a GRU with hidden state $\mathbf{s_i}$. It follows a similar formula to the encoder, but takes one extra input $\mathbf{c}_{i}$ (*shown in yellow*).
#
# $$\mathbf{s}_{i} = f( \mathbf{s}_{i - 1}, \mathbf{y}_{i - 1}, \mathbf{c}_i )$$
#
# Here, $\mathbf{y}_{i - 1}$ is the previously generated target word (*not shown*).
#
# At each time step, an **attention mechanism** dynamically selects that part of the source sentence that is most relevant for predicting the current target word. It does so by comparing the last decoder state with each source hidden state. The result is a context vector $\mathbf{c_i}$ (*shown in yellow*).
# Later the attention mechanism is explained in more detail.
#
# After computing the decoder state $\mathbf{s}_i$, a non-linear function $g$ (which applies a [softmax](https://en.wikipedia.org/wiki/Softmax_function)) gives us the probability of the target word $y_i$ for this time step:
#
# $$ p(y_i \mid y_{<i}, x_1^M) = g(\mathbf{s}_i, \mathbf{c}_i, \mathbf{y}_{i - 1})$$
#
# Because $g$ applies a softmax, it provides a vector the size of the output vocabulary that sums to 1.0: it is a distribution over all target words. During test time, we would select the word with the highest probability for our translation.
#
# Now, for optimization, a [cross-entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) is used to maximize the probability of selecting the correct word at this time step. All parameters (including word embeddings) are then updated to maximize this probability.
#
#

# %% [markdown]
# ### Update model types
#
# >In this extended version of the original notebook we add exercises to vary the model type, in particular, 
# >using a BiLSTM in place of a GRU and removing the attention mechanism from the model. The aim is to get a feeling 
# >for how such model architecture changes work in practice and how they can influence the performance of the model.

# %% [markdown]
# # Prelims
#
# This tutorial requires **PyTorch >= 0.4.1** and was tested with **Python 3.6**.  
#
# Make sure you have those versions, and install the packages below if you don't have them yet.

# %%
# #!pip install torch numpy matplotlib sacrebleu

# %%
# %matplotlib inline
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython.core.debugger import set_trace

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
import warnings
# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Some convenience helper functions used throughout the notebook
# Taken from the annotated transformer project

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# %% [markdown]
# # Let's start coding!
#
# ## Model class
#
# Our base model class `EncoderDecoder` is very similar to the one in *The Annotated Transformer*.
#
# One difference is that our encoder also returns its final states (`encoder_final` below), which is used to initialize the decoder RNN. We also provide the sequence lengths as the RNNs require those.

# %%
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None, decoder_cell_state=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden, cell_state=decoder_cell_state)


# %% [markdown]
# To keep things easy we also keep the `Generator` class the same. 
# It simply projects the pre-output layer ($x$ in the `forward` function below) to obtain the output layer, so that the final dimension is the target vocabulary size.

# %%
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# %% [markdown]
# ## Encoder
#
# Our encoder is a bi-directional GRU. 
#
# Because we want to process multiple sentences at the same time for speed reasons (it is more effcient on GPU), we need to support **mini-batches**. Sentences in a mini-batch may have different lengths, which means that the RNN needs to unroll further for certain sentences while it might already have finished for others:
#
# ```
# Example: mini-batch with 3 source sentences of different lengths (7, 5, and 3).
# End-of-sequence is marked with a "3" here, and padding positions with "1".
#
# +---------------+
# | 4 5 9 8 7 8 3 |
# +---------------+
# | 5 4 8 7 3 1 1 |
# +---------------+
# | 5 8 3 1 1 1 1 |
# +---------------+
# ```
# You can see that, when computing hidden states for this mini-batch, for sentence #2 and #3 we will need to stop updating the hidden state after we have encountered "3". We don't want to incorporate the padding values (1s).
#
# Luckily, PyTorch has convenient helper functions called `pack_padded_sequence` and `pad_packed_sequence`.
# These functions take care of masking and padding, so that the resulting word representations are simply zeros after a sentence stops.
#
# The code below reads in a source sentence (a sequence of word embeddings) and produces the hidden states.
# It also returns a final vector, a summary of the complete sentence, by concatenating the first and the last hidden states (they have both seen the whole sentence, each in a different direction). We will use the final vector to initialize the decoder.

# %% [markdown]
# ### We add the model type in prepration for experiments with different model variants

# %%
from enum import Enum

class ModelType(Enum):
    BIGRU_WITH_ATTENTION = 1
    BILSTM_WITH_ATTENTION = 2
    BIGRU = 3 # A simple BiGRU, i.e. without attention
    BILSTM = 4 # A simple BiLSTM, i.e. without attention


# %%
class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., 
                 model_type:ModelType=ModelType.BIGRU_WITH_ATTENTION):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        
        if model_type == ModelType.BIGRU_WITH_ATTENTION or model_type == ModelType.BIGRU:
            # The encoder model is the same (GRU) both for BiGRU with attention and without attention
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)
        elif model_type == ModelType.BILSTM_WITH_ATTENTION or model_type == ModelType.BILSTM:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)   
        else:
            raise RuntimeError("Error: unknown model type: " + str(model_type))
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        if isinstance(self.rnn, nn.GRU):
            output, final = self.rnn(packed)
        else:
            #For LSTM type
            output, (final, _) = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


# %% [markdown]
# ### Decoder
#
# The decoder is a conditional GRU. Rather than starting with an empty state like the encoder, its initial hidden state results from a projection of the encoder final vector. 
#
# #### Training
# In `forward` you can find a for-loop that computes the decoder hidden states one time step at a time. 
# Note that, during training, we know exactly what the target words should be! (They are in `trg_embed`.) This means that we are not even checking here what the prediction is! We simply feed the correct previous target word embedding to the GRU at each time step. This is called teacher forcing.
#
# The `forward` function returns all decoder hidden states and pre-output vectors. Elsewhere these are used to compute the loss, after which the parameters are updated.
#
# #### Prediction
# For prediction time, for forward function is only used for a single time step. After predicting a word from the returned pre-output vector, we can call it again, supplying it the word embedding of the previously predicted word and the last state.

# %%
class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 model_type:ModelType=ModelType.BIGRU_WITH_ATTENTION,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        
        if model_type == ModelType.BIGRU_WITH_ATTENTION or model_type == ModelType.BIGRU:
            # Whether we use attention or not, the rnn is the same (of type GRU)
            self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
              batch_first=True, dropout=dropout)            
        elif model_type == ModelType.BILSTM_WITH_ATTENTION or model_type == ModelType.BILSTM:
            # Whether we use attention or not, the rnn is the same (of type LSTM)
            self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
              batch_first=True, dropout=dropout)            
        else:
            raise RuntimeError("Error: unknown model type: " + str(model_type))
        
        
        print("model_type: " + str(model_type))
        if model_type == ModelType.BIGRU or model_type == ModelType.BILSTM:
            # Attention must be set to none for these model types
            if self.attention is not None:
                raise RuntimeError("Error: self.attention must be none for the BIGRU and BILSTM model types")
        

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden, cell_state, encoder_final):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        
        if self.attention is not None:
            context, attn_probs = self.attention(
                query=query, proj_key=proj_key,
                value=encoder_hidden, mask=src_mask)
        else:
            #print("encoder_hidden.size(): " + str(encoder_hidden.size()))            
            #print("encoder_final.size(): " + str(encoder_final.size()))
            #print("prev_embed.size(): " + str(prev_embed.size()))
            
            # Simply use the final (hidden) state of the encoder as the context
            # Found out by looking at the sizes, that we have to swap the first two axes
            context = encoder_final.transpose(0,1)
            #raise RuntimeError("Error: forward step part not implemented")

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        
        if isinstance(self.rnn, nn.GRU):
            output, hidden = self.rnn(rnn_input, hidden)
        else:
            #For LSTM type
            output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))
        
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output, cell_state
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, cell_state=None, 
                max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        # initialize decoder cell state
        if cell_state is None:
            cell_state = torch.zeros_like(hidden)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = None
        if self.attention is not None:
            proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            ### Sizes for debugging
#             print("prev_embed.size(): " + str(prev_embed.size()))
#             print("encoder_hidden.size(): " + str(encoder_hidden.size()))
#             print("src_mask.size(): " + str(src_mask.size()))
#             print("proj_key.size(): " + str(proj_key.size()))
#             print("hidden.size(): " + str(hidden.size()))
            
            output, hidden, pre_output, cell_state = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden, cell_state, encoder_final)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors, cell_state  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))            



# %% [markdown]
# ### Attention                                                                                                                                                                               
#
# At every time step, the decoder has access to *all* source word representations $\mathbf{h}_1, \dots, \mathbf{h}_M$. 
# An attention mechanism allows the model to focus on the currently most relevant part of the source sentence.
# The state of the decoder is represented by GRU hidden state $\mathbf{s}_i$.
# So if we want to know which source word representation(s) $\mathbf{h}_j$ are most relevant, we will need to define a function that takes those two things as input.
#
# Here we use the MLP-based, additive attention that was used in Bahdanau et al.:
#
# <img src="images/attention.png" width="280">
#
#
# We apply an MLP with tanh-activation to both the current decoder state $\bf s_i$ (the *query*) and each encoder state $\bf h_j$ (the *key*), and then project this to a single value (i.e. a scalar) to get the *attention energy* $e_{ij}$. 
#
# Once all energies are computed, they are normalized by a softmax so that they sum to one: 
#
# $$ \alpha_{ij} = \text{softmax}(\mathbf{e}_i)[j] $$
#
# $$\sum_j \alpha_{ij} = 1.0$$ 
#
# The context vector for time step $i$ is then a weighted sum of the encoder hidden states (the *values*):
# $$\mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{h}_j$$

# %%
class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


# %% [markdown]
# ## Embeddings and Softmax                                                                                                                                                                                                                                                                                           
# We use learned embeddings to convert the input tokens and output tokens to vectors of dimension `emb_size`.
#
# We will simply use PyTorch's [nn.Embedding](https://pytorch.org/docs/stable/nn.html?highlight=embedding#torch.nn.Embedding) class.

# %% [markdown]
# ## Full Model
#
# Here we define a function from hyperparameters to a full model. 

# %%

def make_model(model_type:ModelType, src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    
    attention = None
    
    if model_type ==  ModelType.BIGRU_WITH_ATTENTION or  model_type == ModelType.BILSTM_WITH_ATTENTION:
        attention = BahdanauAttention(hidden_size)

    known_model_types = set([
        ModelType.BIGRU_WITH_ATTENTION,
        ModelType.BIGRU,
        ModelType.BILSTM_WITH_ATTENTION,
        ModelType.BILSTM])
        
    if model_type in known_model_types:
        model = EncoderDecoder(
            Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, model_type=model_type),
            Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout, model_type=model_type),
            nn.Embedding(src_vocab, emb_size),
            nn.Embedding(tgt_vocab, emb_size),
            Generator(hidden_size, tgt_vocab))
    else:
        raise RuntimeError("Error: unknown model type: " + str(model_type))

    return model.cuda() if USE_CUDA else model


# %% [markdown]
# # Training
#
# This section describes the training regime for our models.

# %% [markdown]
# We stop for a quick interlude to introduce some of the tools 
# needed to train a standard encoder decoder model. First we define a batch object that holds the src and target sentences for training, as well as their lengths and masks. 

# %% [markdown]
# ## Batches and Masking

# %%
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()
                


# %% [markdown]
# ## Training Loop
# The code below trains the model for 1 epoch (=1 pass through the training data).

# %%
def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        if batch == None:
            continue
            
#         if i > 10:
#             break
        
        out, _, pre_output, _ = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


# %% [markdown]
# ## Training Data and Batching
#
# We will use torch text for batching. This is discussed in more detail below. 

# %% [markdown]
# ## Optimizer
#
# We will use the [Adam optimizer](https://arxiv.org/abs/1412.6980) with default settings ($\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon=10^{-8}$).
#
# We will use $0.0003$ as the learning rate here, but for different problems another learning rate may be more appropriate. You will have to tune that.

# %% [markdown]
# # A First  Example
#
# We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols. 

# %% [markdown]
# ## Synthetic Data

# %%
def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
          np.random.randint(1, num_words, size=(batch_size, length)))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


# %% [markdown]
# ## Loss Computation

# %%
class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


# %% [markdown]
# ### Printing examples
#
# To monitor progress during training, we will translate a few examples.
#
# We use greedy decoding for simplicity; that is, at each time step, starting at the first token, we choose the one with that maximum probability, and we never revisit that choice. 

# %%
def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None
    cell_state = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output, cell_state = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden, cell_state)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        if model.decoder.attention is not None:
            attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    if len(attention_scores) > 0:
        return output, np.concatenate(attention_scores, axis=1)
    else:
        return output, None
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.get_itos()[i] for i in x]

    return [str(t) for t in x]


# %%
def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = SPECIALS.index("</s>") #src_vocab.get_stoi[EOS_TOKEN] 
        trg_sos_index = SPECIALS.index("<s>") #trg_vocab.get_stoi[SOS_TOKEN]
        trg_eos_index = SPECIALS.index("</s>") #trg_vocab.get_stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
        
        if batch == None:
            continue
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break


# %% [markdown]
# ## Training the copy task

# %%
def train_copy_task():
    """Train the simple copy task."""
    num_words = 11
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(ModelType.BIGRU_WITH_ATTENTION, num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=9)
        
    return dev_perplexities


# %%
# train the copy task
dev_perplexities = train_copy_task()

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    
plot_perplexity(dev_perplexities)

# %% [markdown]
# You can see that the model managed to correctly 'translate' the two examples in the end.
#
# Moreover, the perplexity of the development data nicely went down towards 1.

# %% [markdown]
# # A Real World Example
#
# Now we consider a real-world example using the IWSLT German-English Translation task. 
# This task is much smaller than usual, but it illustrates the whole system. 
#
# The cell below installs torch text and spacy. This might take a while.

# %%
# #!pip install git+git://github.com/pytorch/text spacy 
# #!python -m spacy download en
# #!python -m spacy download de

# %% [markdown]
# ## Data Loading
#
# We will load the dataset using torchtext and spacy for tokenization.
#
# This cell might take a while to run the first time, as it will download and tokenize the IWSLT data.
#
# For speed we only include short sentences, and we include a word in the vocabulary only if it occurs at least 5 times. In this case we also lowercase the data.
#
#
#
# If you have **issues** with torch text in the cell below (e.g. an `ascii` error), try running `export LC_ALL="en_US.UTF-8"` before you start `jupyter notebook`.

# %%
### THIIS IS THE OLD DATA LOADING CODE, BUT IT NO LONGER WORKS IN NEWER PYTORCH/TORCHTEXT VERSIONS 
### THEREFORE WE REPLACE IT WITH CODE ADAPTED FROM THE ANNOTATED TRANSFORMER NOTEBOOK
###

# # For data loading.
# from torchtext import data, datasets

# if True:
#     import spacy
#     spacy_de = spacy.load('de')
#     spacy_en = spacy.load('en')

#     def tokenize_de(text):
#         return [tok.text for tok in spacy_de.tokenizer(text)]

#     def tokenize_en(text):
#         return [tok.text for tok in spacy_en.tokenizer(text)]

#     UNK_TOKEN = "<unk>"
#     PAD_TOKEN = "<pad>"    
#     SOS_TOKEN = "<s>"
#     EOS_TOKEN = "</s>"
#     LOWER = True
    
#     # we include lengths to provide to the RNNs
#     SRC = data.Field(tokenize=tokenize_de, 
#                      batch_first=True, lower=LOWER, include_lengths=True,
#                      unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
#     TRG = data.Field(tokenize=tokenize_en, 
#                      batch_first=True, lower=LOWER, include_lengths=True,
#                      unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

#     MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
#     train_data, valid_data, test_data = datasets.IWSLT.splits(
#         exts=('.de', '.en'), fields=(SRC, TRG), 
#         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
#             len(vars(x)['trg']) <= MAX_LEN)
#     MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
#     SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
#     TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    
#     PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]



# %%
import spacy
from torchtext import data, datasets
# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


# %%
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])
        


# %%
from os.path import exists
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

PAD_TOKEN = "<blank>"   
EOS_TOKEN = "</s>"
SOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"
SPECIALS= [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN,UNK_TOKEN]
MIN_FREQ = 5 

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
#     train, val, test = datasets.IWSLT2016(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=MIN_FREQ,
        specials=SPECIALS,
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
#     train, val, test = datasets.IWSLT2016(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=MIN_FREQ,
        specials=SPECIALS,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt



if is_interactive_notebook():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])
    
#print(" vocab_src.get_stoi(): " + str( vocab_src.get_stoi()))    
#PAD_INDEX = vocab_src.get_stoi()[PAD_TOKEN]  ## Does not work somehow, perhaps specials are not
# returned in get_stoi
PAD_INDEX = SPECIALS.index(PAD_TOKEN)

# %% [markdown]
# ## Iterators
# In the new torch text data paradigm, we need data iterators. These have been adapted from the (new)
# annotated transformer project, with adaptations to this project.
#
# One of the main adaptations that needed to be made to the code is to adapt the collate_batch function so that it returns tuples with src and src_lengths paired, and tgt and tgt_lengths paired. Whereas the Transformer model code does not require these src_lengths, the Bahdenau model code does. Originally the Bahdenau model code by Bastings worked with BucketIterator, but this is no longer available in newer PyTorch versions. Hence things have to be fixed more manually in the collate function, as done below for the Transformer model and here adapted to also work for the Bahdenau model based on the code by Bastings.

# %%
from torch.nn.functional import pad
import numpy

def get_processed_examples_and_lengths(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab, device,
                                          max_length):
    """
    This function produces processed source and target examples, in torch tensor format, for further processing
    by the collate_batch function.
    By collecting src and tgt lengths while processing these examples, and returning them in a list
    we can use them when computing the necessary padding in the collate_batch function. 
    This prevents double work.
    """
   

    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
   

    processed_src_items_list, processed_tgt_items_list = [], []
    src_lengths_list, tgt_lengths_list = [], []
   
    for i, (_src, _tgt) in enumerate(batch):
        #print("example number in batch: " + str(i))
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
      
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        
        src_length =  len(processed_src)        
        tgt_length =  len(processed_tgt)
        
        # Filter out examples that are too long on the source side or target side
        if src_length <= max_length and tgt_length <= max_length:
            processed_src_items_list.append(processed_src)
            processed_tgt_items_list.append(processed_tgt)
            tgt_lengths_list.append(tgt_length)
            src_lengths_list.append(src_length)
#         else:
#             #print("src_length: " + str(src_length))
    return processed_src_items_list, processed_tgt_items_list, src_lengths_list, tgt_lengths_list


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
    max_length:int=25
):
    
    src_list, tgt_list = [], []
    processed_src_items_list, processed_tgt_items_list, src_lengths_list, tgt_lengths_list =\
        get_processed_examples_and_lengths(batch, src_pipeline,
        tgt_pipeline, src_vocab, tgt_vocab, device, max_length)
    
    if not processed_src_items_list:
        # list is empty
        return None
    
    #print("batch size after filtering items: " + str(len(processed_src_items_list)))
   
   
    max_padding_source = max(src_lengths_list)
    max_padding_target = max(tgt_lengths_list)

    for processed_src, src_length in zip(processed_src_items_list, src_lengths_list):
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    #max_padding - src_length,
                    max_padding_source - src_length,
                ),
                value=pad_id,
            )
        )
    for processed_tgt, tgt_length in zip(processed_tgt_items_list, tgt_lengths_list):
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding_target - tgt_length,
                ),
                value=pad_id,
            )
        )
        #print(" src_length:" + str(src_length))
        #print(" tgt_length:" + str(tgt_length))

    
    # Determine a sorting order so that elements are sorted by the src length, increasing 
    # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
    #sorting_order  = numpy.argsort(src_lengths_list).reverse() # supposedly faster than with sorte
    sorting_order = sorted(range(len(src_lengths_list)), key=lambda k: src_lengths_list[k], reverse=True)
    # Reorder all the lists according to sorting order
    src_lengths_list = [src_lengths_list[i] for i in sorting_order]
#     print("src_lengths_list after sorting: " + str(src_lengths_list))
    tgt_lengths_list = [tgt_lengths_list[i] for i in sorting_order]
    src_list = [src_list[i] for i in sorting_order]
    tgt_list = [tgt_list[i] for i in sorting_order]
        
    # Convert int lists to torch tensors
    src_lengths = torch.tensor(src_lengths_list,dtype=torch.int)
    tgt_lengths = torch.tensor(tgt_lengths_list,dtype=torch.int)
            
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    #return (src, tgt)
    return ((src,src_lengths), (tgt, tgt_lengths))

# %%
import torchtext

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_sizes=[12000,12000,12000],
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en"))
    # TODO: Filter sentences by length
    
#     # See this issue: https://github.com/pytorch/text/issues/1091
#     #url = 'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8'
#     url = "https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8"
#     #print("datasets.IWSLT2016.URL: " + str(url))
#     torchtext.utils.download_from_url(url)
    
    # See this issue: we have to manually download, and put in the right location to make this work for now
    # See: https://github.com/pytorch/text/issues/1676
#     train_iter, valid_iter, test_iter = datasets.IWSLT2016(
#         language_pair=("de", "en"))
    

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )
    test_iter_map = to_map_style_dataset(test_iter)
    test_sampler = (
        DistributedSampler(test_iter_map) if is_distributed else None
    )

    
    # Got rid of the iter maps for validation and test, use the normal iters instead,
    # because the order of iteration is not stable with the iter_map, which messes up the
    # evaluation otherwise (because you would expect a stable order there)
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_sizes[0],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
#         valid_iter_map,
        valid_iter,
        batch_size=batch_sizes[1],
#         shuffle=(valid_sampler is None),
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
#         test_iter_map,
        test_iter,
        batch_size=batch_sizes[2],
#         shuffle=(test_sampler is None),
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader, test_dataloader

# %% [markdown]
# ### Create the dataloaders

# %%
# Adapted to work with dataloaders instead of BucketIterator etc

train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_sizes=[128,1,1],
        is_distributed=False,
    )

# %% [markdown]
# ### Let's look at the data
#
# It never hurts to look at your data and some statistics.

# %%
from collections import Counter
import spacy

train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en"))
train_iter_map = to_map_style_dataset(train_iter)
valid_iter_map = to_map_style_dataset(valid_iter)
test_iter_map = to_map_style_dataset(test_iter)

def get_num_examples(dataloader):
    result = 0
    for b in dataloader:
        batch = rebatch(PAD_INDEX, b)
        if batch is not None:
            result += len(batch.src_lengths)
    return result
            
def get_examples_src_tgt(dataloader, as_string: bool = True):
    """
    This method essentially reconstructs the lists of words from the batches. This is necessary, because 
    it is not straightforward to loop over the data otherwise, in a correct way. 
    Esssentially we have the DataLoaders, 
    and can get the batches from these, which is what we use here. 
    """
    result = 0
    for b in dataloader:
        batch = rebatch(PAD_INDEX, b)
        if batch is not None:
            #print("batch.src.size(): " + str(batch.src.size()))
            #print("batch.tgt.size(): " + str(batch.trg.size()))
            num_batch_examples = batch.src.size(0)
            #print("num batch examples: " + str(num_batch_examples))
            for example_index in range(0, num_batch_examples):
                source_words = lookup_words(batch.src[example_index][:], vocab_src)
                target_words = lookup_words(batch.trg[example_index][:], vocab_tgt)
                if as_string:
                    src_string =  " ".join(source_words)
                    print("src_string:" + src_string)
                    tgt_string =  " ".join(target_words)
                    yield src_string, tgt_string
                else:
#                     print("source words: " + str(source_words))
#                     print("target words: " + str(target_words))
                    yield source_words, target_words
                
def get_examples_src(dataloader, as_string: bool = True):
    for src, tgt in  get_examples_src_tgt(dataloader, as_string):
        yield src
        
def get_examples_tgt(dataloader, as_string: bool = True):
    for src, tgt in  get_examples_src_tgt(dataloader, as_string):
        yield tgt
                                             
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def print_data_info(train_dataloader, valid_dataloader, test_dataloader):
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    # These statistics are more reliable, since they do not count sentences that are actually skipped
    print('train', get_num_examples(train_dataloader))
    print('valid', get_num_examples(valid_dataloader))
    print('test', get_num_examples(test_dataloader), "\n")
    # https://stackoverflow.com/questions/5384570/whats-the-shortest-way-to-count-the-number-of-items-in-a-generator-iterator
    # This we can still do with the basic dataset iterators
#     print('train', sum(1 for _ in train_iter))
#     print('valid', sum(1 for _ in valid_iter))
#     print('test', sum(1 for _ in test_iter), "\n")

    print("First training example:")
    print("(Retrieved from the batches of the dataloader)")
#     print("src:", " ".join(vars(train_dataloader[0])['src']))
#     print("trg:", " ".join(vars(train_dataloader[0])['trg']), "\n")
    #https://stackoverflow.com/questions/4741243/how-to-pick-just-one-item-from-a-generator
    src_string, tgt_string = next(get_examples_src_tgt(train_dataloader, True))
    print("src:",src_string , "\n")
    print("trg:", tgt_string, "\n")
    
    
    print("(directly from the train_iter)")
    for source_target in train_iter:
        print("src:",source_target[0] , "\n")
        print("trg:", source_target[1], "\n")
        break

    
    ### We use counters to get the frequencies, as frequencies are no longer available 
    ### in the vocabulary, in the newer PyTorch versions
    print("create counters...")
    source_counter = Counter()
    target_counter = Counter()
    print("(We collect statistics for the validation set, to be faster)")
    for _, (source_words, target_words) in enumerate(get_examples_src_tgt(valid_dataloader, False)):
        source_counter.update(source_words)
        target_counter.update(target_words)

      # This gives somehow unexpected results: the first example is repeated every time somehow    
#     for _, source_taget in enumerate(train_iter_map):
#         source = source_target[0]
#         target = source_target[1]
#         print("source: " + str(source))
#         print("target: " + str(target))
#         source_counter.update(tokenize_de(source))
#         target_counter.update(tokenize_en(target))
    
#     print("\n".join(["%10s %10d" % x for x in vocab_src.freqs.most_common(10)]), "\n")
#     print("Most common words (trg):")
#     print("\n".join(["%10s %10d" % x for x in vocab_tgt.freqs.most_common(10)]), "\n")
    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in source_counter.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in target_counter.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(vocab_src.get_itos()[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(vocab_tgt.get_itos()[:10])), "\n")

    print("Number of German words (types):", len(vocab_src))
    print("Number of English words (types):", len(vocab_tgt), "\n")
    
    
print_data_info(train_dataloader, valid_dataloader, test_dataloader)

# %% [markdown]
# ## Iterators
# Batching matters a ton for speed. We will use torch text's BucketIterator here to get batches containing sentences of (almost) the same length.
#
# #### Note on sorting batches for RNNs in PyTorch
#
# For effiency reasons, PyTorch RNNs require that batches have been sorted by length, with the longest sentence in the batch first. For training, we simply sort each batch. 
# For validation, we would run into trouble if we want to compare our translations with some external file that was not sorted. Therefore we simply set the validation batch size to 1, so that we can keep it in the original order.

# %%
# train_iter = data.BucketIterator(train_data, batch_size=64, train=True, 
#                                  sort_within_batch=True, 
#                                  sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
#                                  device=DEVICE)
# valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, 
#                            device=DEVICE)


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    
    if batch == None:
        return None
    
    src, tgt = batch 
    return Batch(src, tgt, pad_idx)

# %% [markdown]
# ## Training the System
#
# Now we train the model. 
#
# On a Titan X GPU, this runs at ~18,000 tokens per second with a batch size of 64.

# %%


def train(model, num_epochs=10, lr=0.0003, print_every=100):
    
    pad_idx = SPECIALS.index("<blank>")
        
    """Train a model on IWSLT"""
    
    if USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(pad_idx, b) for b in train_dataloader), 
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optim),
                                     print_every=print_every)
        
        
        model.eval()
        with torch.no_grad():
            print_examples((rebatch(pad_idx, x) for x in valid_dataloader), 
                           model, n=3, src_vocab=vocab_src, trg_vocab=vocab_tgt)        

            dev_perplexity = run_epoch((rebatch(pad_idx, b) for b in valid_dataloader), 
                                       model, 
                                       SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
        
    return dev_perplexities


# %%
import os
dir_path = os.path.dirname(os.path.realpath("."))
print("dir_path: " + str(dir_path))

# %%
bigru_with_atttention_model = make_model(ModelType.BIGRU_WITH_ATTENTION,
                   len(vocab_src), len(vocab_tgt),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)
dev_perplexities = train(bigru_with_atttention_model, print_every=100)

# %%
plot_perplexity(dev_perplexities)

# %% [markdown]
# ## Prediction and Evaluation
#
# Once trained we can use the model to produce a set of translations. 
#
# If we translate the whole validation set, we can use [SacreBLEU](https://github.com/mjpost/sacreBLEU) to get a [BLEU score](https://en.wikipedia.org/wiki/BLEU), which is the most common way to evaluate translations.
#
# #### Important sidenote
# Typically you would use SacreBLEU from the **command line** using the output file and original (possibly tokenized) development reference file. This will give you a nice version string that shows how the BLEU score was calculated; for example, if it was lowercased, if it was tokenized (and how), and what smoothing was used. If you want to learn more about how BLEU scores are (and should be) reported, check out [this paper](https://arxiv.org/abs/1804.08771).
#
# However, right now our pre-processed data is only in memory, so we'll calculate the BLEU score right from this notebook for demonstration purposes.
#
# We'll first test the raw BLEU function:

# %%
import sacrebleu

# %%
# this should result in a perfect BLEU of 100%
hypotheses = ["this is a test"]
references = ["this is a test"]
bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
print(bleu)

# %%
# here the BLEU score will be lower, because some n-grams won't match
hypotheses = ["this is a test"]
references = ["this is a fest"]
bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
print(bleu)

# %% [markdown]
# Since we did some filtering for speed, our validation set contains 690 sentences.
# The references are the tokenized versions, but they should not contain out-of-vocabulary UNKs that our network might have seen. So we'll take the references straight out of the `valid_data` object:

# %%
get_num_examples(valid_dataloader)

# %%
references_words = []
for b in valid_dataloader:
    #print("b " + str(b))
    if b is not None:
        batch = rebatch(PAD_INDEX, b)
        # print("batch.trg.size():" + str(batch.trg.size()))
        words = lookup_words(batch.trg[0][:], vocab_tgt)
        references_words.append(words)
    

                          
references = [" ".join(x) for x in references_words]                          

print(len(references))
print(references[0])

# https://stackoverflow.com/questions/3845423/remove-empty-strings-from-a-list-of-strings
# remove empty elements  (strings with length 0)  
references =  [x for x in references if x]

for reference in references:
    print("reference: " + str(reference))

# %%
references[-2]

# %% [markdown]
# **Now we translate the validation set!**
#
# This might take a little bit of time.
#
# Note that `greedy_decode` will cut-off the sentence when it encounters the end-of-sequence symbol, if we provide it the index of that symbol.

# %%
hypotheses_idx = []
alphas = []  # save the last attention scores
evaluation_inputs = []  # Collect the evaluation inputs for later use
for batch in valid_dataloader:
    if batch == None:
        continue
    
    batch = rebatch(PAD_INDEX, batch)
    evaluation_inputs.append(batch.src)
    pred, attention = greedy_decode(
    bigru_with_atttention_model, batch.src, batch.src_mask, batch.src_lengths, max_len=25,
    sos_index=SPECIALS.index(SOS_TOKEN),
    eos_index=SPECIALS.index(EOS_TOKEN))
    hypotheses_idx.append(pred)
    alphas.append(attention)

# %%
# we will still need to convert the indices to actual words!
hypotheses_idx[0]

# %%
hypotheses_words = [lookup_words(x, vocab_tgt) for x in hypotheses_idx]
print("len(hypotheses): " + str(len(hypotheses_words)))
hypotheses_words[0]



# %%
# finally, the SacreBLEU raw scorer requires string input, so we convert the lists to strings
hypotheses = [" ".join(x) for x in hypotheses_words]
        
print(len(hypotheses))
print(hypotheses[0])

for hypothesis in hypotheses:
    print("hypothesis: " + str(hypothesis))

# remove empty elements    
hypotheses =  [x for x in hypotheses if x]


# %%
# now we can compute the BLEU score!
bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
print(bleu)


# %% [markdown]
# ## Attention Visualization
#
# We can also visualize the attention scores of the decoder.

# %%
def plot_heatmap(src, trg, scores):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    ax.set_xticklabels(trg, minor=False, rotation='vertical')
    ax.set_yticklabels(src, minor=False)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()


# %%
# This plots a chosen sentence, for which we saved the attention scores above.
idx = 5  # Change this index to visualze attention for different examples

### Old code, no longer works because valid_data is not available
# src = valid_data[idx].src + ["</s>"]
# trg = valid_data[idx].trg + ["</s>"]
# pred = hypotheses[idx].split() + ["</s>"]
print("evaluation_inputs[5].size(): " + str(evaluation_inputs[idx].size()))
# Some tweaking is needed to get the right (parts of) these word lists again.
# We don't want the start symbol here, so we start from index 1 for src and trg
src = lookup_words((evaluation_inputs[idx])[0][:], vocab_src)[1:] #+ ["</s>"]
trg = references_words[idx][1:] + ["</s>"][1:]
pred = hypotheses_words[idx] + ["</s>"]

pred_att = alphas[idx][0].T[:, :len(pred)]
print("src", src)
print("ref", trg)
print("pred", pred)
plot_heatmap(src, pred, pred_att)

# %% [markdown]
# ## Exercise: Experiments with different model types
#
# >In this exercise you will test different variants of the BiGRU-with-attention model, namely:
# >>    1. A nearly identical model in which the the BiGRU is replaced with a BiLSTM
# >>    2. A similar BiGRU model, but without attention. Instead of using a weighted mean of the hidden states of the 
# >>       encoder, it just always uses the last state of the encoder.
# >>    3. A BiLSTM model. Similar to 2, but using again a BiLSTM in place of a BiGRU.
#
# > To do so, you will have to adapt the Encoder and Decoder classes in the right manner, 
# in order to implement these model variants.
# > To get you started and limit the coding work, a code scaffolding has already been provided, with 
# \#TODO and RuntimeErrors indicating places where the code should be adapted/augmented.
#
# Tips: 
# 1. The LSTM has a slightly different input and output as the GRU. In particular, it takes also a cell 
# state as input and produces ti as an output.
# See: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#         
# You can initialize it to zeros when providing it as first input, and you can create this zeros from the hidden 
# state tensor, using the torch.zeros_like method.
#
# https://pytorch.org/docs/stable/generated/torch.zeros_like.html
#     
# 2. When implementing the models without attention, you actually need the final hidden state ("encoder_final") produced by the rnn.
# Just incorporate the forward_step function to provide this tensor as an input. You can use the 
# "tensor.size()" methods to check size, and you need the "tensor.swap(0,1)" method call to swap the first to 
# axes of this tensor, when using it as a context vector.
#

# %%
bilstm_with_atttention_model = make_model(ModelType.BILSTM_WITH_ATTENTION,
                   len(vocab_src), len(vocab_tgt),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)
dev_perplexities = train(bilstm_with_atttention_model, print_every=100)

# %%
plot_perplexity(dev_perplexities)

# %%
# plot_perplexity(bilstm_with_atttention_model)
print(ModelType.BIGRU)

# %%
bigru_model = make_model(ModelType.BIGRU,
                   len(vocab_src), len(vocab_tgt),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)
dev_perplexities = train(bigru_model, print_every=100)

# %%
plot_perplexity(dev_perplexities)

# %%
bilstm_model = make_model(ModelType.BILSTM,
                   len(vocab_src), len(vocab_tgt),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)
dev_perplexities = train(bilstm_model, print_every=100)

# %%
plot_perplexity(dev_perplexities)


# %%
def evaluate_model(model):
    """
    The code from above was collected together into a method, parametrized by the model.
    This way, we can easilly computer performance for different models for comparison
    """
    hypotheses_idx = []
    alphas = []  # save the last attention scores
    evaluation_inputs = []  # Collect the evaluation inputs for later use
    for batch in valid_dataloader:
        if batch == None:
            continue

        batch = rebatch(PAD_INDEX, batch)
        evaluation_inputs.append(batch.src)
        pred, attention = greedy_decode(
        model, batch.src, batch.src_mask, batch.src_lengths, max_len=25,
        sos_index=SPECIALS.index(SOS_TOKEN),
        eos_index=SPECIALS.index(EOS_TOKEN))
        hypotheses_idx.append(pred)
        alphas.append(attention)
        
    # we will still need to convert the indices to actual words!
    hypotheses_idx[0]
    
    hypotheses_words = [lookup_words(x, vocab_tgt) for x in hypotheses_idx]
    print("len(hypotheses): " + str(len(hypotheses_words)))
    hypotheses_words[0]
    
    # finally, the SacreBLEU raw scorer requires string input, so we convert the lists to strings
    hypotheses = [" ".join(x) for x in hypotheses_words]

    print(len(hypotheses))
    print(hypotheses[0])

    
    #print("references: " + str(references))
#     for hypothesis in hypotheses:
#         print("hypothesis: " + str(hypothesis))

    # remove empty elements    
    hypotheses =  [x for x in hypotheses if x]
    
    # now we can compute the BLEU score!
    bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
    print("BLEU score: " + str(bleu))

   


# %% [markdown]
# ## Model comparison
# Having implemented the model code for the three different variants, you should be able to run the code below to 
# determine and compare the performance of the four different model variants

# %%
#print("Evaluate BiGRU with attention model")
#evaluate_model(bigru_with_atttention_model)
# The last three models will not work yet, you will have to augment the code to get them to work!
print("Evaluate BiLSTM with attention model...")
evaluate_model(bilstm_with_atttention_model)
print("Evaluate BiGRU model...")
#evaluate_model(bigru)
print("Evaluate BiLSTM model...")
evaluate_model(bilstm_model)

# %% [markdown]
# # Congratulations! You've finished this notebook.
#
# What didn't we cover?
#
# - Subwords / Byte Pair Encoding [[paper]](https://arxiv.org/abs/1508.07909) [[github]](https://github.com/rsennrich/subword-nmt) let you deal with unknown words. 
# - You can implement a [multiplicative/bilinear attention mechanism](https://arxiv.org/abs/1508.04025) instead of the additive one used here.
# - We used greedy decoding here to get translations, but you can get better results with beam search.
# - The original model only uses a single dropout layer (in the decoder), but you can experiment with adding more dropout layers, for example on the word embeddings and the source word representations.
# - You can experiment with multiple encoder/decoder layers.- Experiment with a benchmarked and improved codebase: [Joey NMT](https://github.com/joeynmt/joeynmt)

# %% [markdown]
# If this was useful to your research, please consider citing:
#
# > J Bastings. 2018. The Annotated Encoder-Decoder with Attention. https://bastings.github.io/annotated_encoder_decoder/
#
# Or use the following `Bibtex`:
# ```
# @misc{bastings2018annotated,
#   title={The Annotated Encoder-Decoder with Attention},
#   author={Bastings, J.},
#   journal={https://bastings.github.io/annotated\_encoder\_decoder/},
#   year={2018}
# }```
