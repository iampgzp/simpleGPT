""" GPT model for text generation """

import glob  # glob is used to read all the files in a folder
import torch  # Pytorch is used as the main library for ML
import torch.nn as nn  # nn is the neural network module
# F is the functional module, which contains loss functions
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 64  # number of sequences to process in parallel
BLOCK_SIZE = 256  # number of tokens in a sequence
MAX_ITERS = 5000  # number of training iterations
EVAL_INTERVAL = 500  # how often to evaluate loss and generate text samples
LEARNING_RATE = 3e-4  # learning rate for optimizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # use gpu if available
EVAL_ITERS = 200  # number of iterations to average loss over when evaluating
N_EMBEDDING = 384  # embedding dimension
N_HEAD = 6  # number of attention heads
N_LAYER = 6  # number of transformer blocks
DROPOUT_RATE = 0.2  # DROPOUT_RATE probability, this is to prevent overfitting

# Data Preparation Section

# read text files and concatenate them into a single string
# Put your data in the data folder as txt files
# Don't forget to remove the tiny-shakespeare.txt file
text_files = glob.glob("data/*.txt")
text = ""
for file_path in text_files:
    with open(file_path, mode="r", encoding="utf-8", errors='ignore') as f:
        text = text + f.read()

# get all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)  # number of unique characters
# encode and decode functions of mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(x): return [stoi[ch] for ch in x]
def decode(x): return ''.join([itos[i] for i in x])


# Split the data into training and validation sets
torch.manual_seed(1337)  # set random seed for reproducibility
# encode text into integers and then create tensors
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # 90% of data for training
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """ generate a batch of inputs and targets from the data"""
    data = train_data if split == 'train' else val_data
    # random starting indices for the sequences
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])  # input sequences
    # output sequences contains the target token to be predicted
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    # move the data to the DEVICE (cpu or gpu)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()  # we don't need gradients, this saves memory and speeds up computation
def estimate_loss():
    """ estimate loss on training and validation sets """
    out = {}
    model.eval()
    for split in ['train', 'val']:  # split and evaluate loss on both splits
        losses = torch.zeros(EVAL_ITERS, device=DEVICE)
        for k in range(EVAL_ITERS):  # average loss over multiple iterations
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ single attention head """

    def __init__(self, head_size):
        super().__init__()
        # key is what X has
        self.key = nn.Linear(N_EMBEDDING, head_size, bias=False)
        # query is what X wants to look for
        self.query = nn.Linear(N_EMBEDDING, head_size, bias=False)
        # value is, if X finds what it's looking for, what it will get
        self.value = nn.Linear(N_EMBEDDING, head_size, bias=False)
        self.DROPOUT_RATE = nn.Dropout(DROPOUT_RATE)
        # lower triangular matrix, used to mask out future tokens
        self.register_buffer('tril', torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        """ main attention calculation """
        B, T, C = x.shape  # B = batch size, T= sequence length, C = # of features

        k = self.key(x)
        q = self.query(x)

        # scaled dot product of K and Q,
        # note we scale by 1/sqrt(C) to make initialization more spread out
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # mask out the upper triangular part of the matrix to -infinity
        # When doing softmax, these values will become 0,
        # essentially ignoring them and not let attention look into the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # apply DROPOUT_RATE to the attention weights so that the model is less likely to overfit
        wei = self.DROPOUT_RATE(wei)

        # apply attention weights to the value vectors
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """ multi-head attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # this will create num_heads attention heads, each with head_size features
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # this helps to project the output of the attention heads back to the original dimension
        self.proj = nn.Linear(N_EMBEDDING, N_EMBEDDING)
        # again here we apply DROPOUT_RATE to prevent overfitting
        self.DROPOUT_RATE = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        # apply all the attention heads and concatenate them together
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # project the concatenated output back to the original dimension
        out = self.proj(out)
        out = self.DROPOUT_RATE(out)
        return out


class FeedFoward(nn.Module):
    """ feed forward module """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # we first expand the dimension to 4 times of the original dimension
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # relu is the activation function, you can change to others
            # then we project back to the original dimension, this is from Transformer paper
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ one transformer block """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # divide the embedding dimension into n_head parts
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(
            n_head, head_size)  # self attention head
        self.ffwd = FeedFoward(n_embd)  # feed forward module
        # layer normalization to reduce training time
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # note we apply layer normalization before self attention,
        # This is an improvement from the original paper
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """ GPT model """

    def __init__(self):
        super().__init__()
        # The first step is to convert the input tokens into embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDDING)

        # The second step is to add position embeddings to the input embeddings
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBEDDING)

        # here is the main part of the model, which is a stack of transformer blocks of size N_LAYER
        self.blocks = nn.Sequential(
            *[Block(N_EMBEDDING, n_head=N_HEAD) for _ in range(N_LAYER)])

        # after the transformer blocks, we project the output back to the original dimension
        self.lm_head = nn.Linear(N_EMBEDDING, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B = batch size, T = sequence length

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        # add the two embeddings together, so that we get position information
        x = tok_emb + pos_emb

        x = self.blocks(x)  # apply the transformer blocks
        # project the output back to the original dimension
        logits = self.lm_head(x)

        # now we have the logits which represent the probability of each token in the vocabulary

        if targets is None:
            # if targets is not provided, we don't need to calculate the loss
            loss = None
        else:
            B, T, C = logits.shape
            # reshape logits to be a 2D tensor, this is what cross entropy loss expects
            logits = logits.view(B*T, C)
            # reshape targets to be a 1D tensor, this is what cross entropy loss expects
            targets = targets.view(B*T)

            # cross entropy loss, you can change to others
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ generate new tokens given a context """
        for _ in range(max_new_tokens):
            # only use the last BLOCK_SIZE tokens as context
            idx_cond = idx[:, -BLOCK_SIZE:]

            # use the model to predict the next token
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # only use the last token's logits
            # convert logits to probabilities that add up to 1
            probs = F.softmax(logits, dim=-1)

            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the new token to the context
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


model = GPT()  # create the model
# model.load_state_dict(torch.load('model.pt')), # if you want to load a pretrained model
m = model.to(DEVICE)  # move the model to the DEVICE (cpu or gpu)

# here we use adam optimizer, you can change to others
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)

for i in range(MAX_ITERS):
    # training loop
    if i % EVAL_INTERVAL == 0:
        # evaluate loss and generate text samples
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')  # get a batch of inputs and targets
    logits, loss = model(xb, yb)  # train the model on the batch
    # reset the gradients to None, improve memory
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # calculate gradients
    optimizer.step()  # update parameters

# initialize context to be empty
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
# generate text samples
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# torch.save(model.state_dict(), 'model.pt') # save the model if you want
