import numpy as np

sentence = " Riinu is an awesome person, and my favorite human\nI like RNN they are pretty cool\n,Juneteenth is a cool holiday"


def Train():
    # Seq length is the Max-timestep value
    # If one has multiple examples a greedy approach would work, or whatever fancy way
    seq_length = len(sentence)
    p = 0
    i = 0
    
    # Also stolent this data prep from Andrej seem to be faster
    while True:
        # check if we've reached the end of the data; if so, reset the hidden state
        if p + seq_length + 1 >= len(sentence) or i == 0 or sentence[p] == "\n":
            a_prev = np.zeros((na, 1))
            p = 0

        # prepare inputs
        inputs = [word2Int[ch] for ch in sentence[p : p + seq_length]]
        targets = [word2Int[ch] for ch in sentence[p + 1 : p + seq_length + 1]]

        # prepare your inputs in one-hot representation
        x = np.zeros((len(inputs), nx, 1))
        for t, char in enumerate(inputs):
            x[t, char] = 1

        # similarly prepare your targets in one-hot representation
        y = np.zeros((len(targets), nx, 1))
        for t, char in enumerate(targets):
            y[t, char] = 1

        loss = RNN(x, y)
        if i % 100 == 0:
            print(f"  Iteration: {i}, Loss: {loss}")
            #print(sentence[p])
            sample(sentence[p], 90)

        p += seq_length
        i += 1


def RNN(x, y):
    a, yhats, loss = forwardCell(x, y, [wax, waa, wya, ba, by])
    dwaa, dwax, dwya, dba, dby = backwardCell(x, y, a, yhats)
    
    # Adagrad update (I stole this from Adrej Karpathy) this is basically the optimization step
    # I tried regular gradient descent it worked but it took forever.
    
    for param, dparam, mem in zip(
        [wax, waa, wya, ba, by],
        [dwax, dwaa, dwya, dba, dby],
        [mWax, mWaa, mWya, mBa, mBy],
    ):
        mem += dparam * dparam
        param += -stepsize * dparam / np.sqrt(mem + 1e-8)  # adagrad update
    return loss

# forward cell iterating through our timesteps
# A time-step is like the next index in a string, or frame in an image
def forwardCell(x, y, weights):
    a = {}
    xs = {}
    yhats = {}
    a[-1] = np.zeros((na, 1))
    loss = 0.0
    for t in range(maxTx - 1):
        # forward
        anext = np.tanh(np.dot(wax, x[t]) + np.dot(waa, a[t - 1]) + ba)
        yhat = softmax(np.dot(wya, anext) + by)
        a[t] = anext
        yhats[t] = yhat
        
        # note this is softmax crossEntropy loss.
        loss += -np.sum(y[t] * np.log(yhats[t]))
    return a, yhats, loss


# Softmax since we're doing multi-classification
def softmax(x):
    t = np.exp(x)
    s = np.sum(t)
    return t / s


# BPTT (backProp through time)
def backwardCell(x, y, a, yhats):
    dwaa = np.zeros_like(waa)
    dwax = np.zeros_like(wax)
    dwya = np.zeros_like(wya)
    dba = np.zeros_like(ba)
    dby = np.zeros_like(by)

    for t in reversed(range(maxTx - 1)):
        # backward
        dy = yhats[t] - y[t].reshape(ny, 1)
        dwya += np.dot(dy, a[t].T)
        danext = np.dot(wya.T, dy)

        dtanh = (1 - a[t] ** 2) * danext
        dwax += np.dot(dtanh, x[t].T)
        dba += dtanh
        dwaa += np.dot(dtanh, a[t - 1].T)
    return dwaa, dwax, dwya, dba, dby


def initWeights(nx, na, ny):
    wax = np.random.randn(na, nx) * np.sqrt(1.0 / nx)
    waa = np.random.randn(na, na) * np.sqrt(1.0 / na)
    wya = np.random.randn(ny, na) * np.sqrt(1.0 / na)
    ba = np.zeros((na, 1))
    by = np.zeros((ny, 1))
    return wax, waa, wya, ba, by


def sample(char, length):
    x = np.zeros((nx, 1))
    x[word2Int[char]] = 1
    anext = np.zeros((na, 1))
    for t in range(length):
        anext = np.tanh(np.dot(wax, x) + np.dot(waa, anext) + ba)
        yhat = softmax(np.dot(wya, anext) + by)
        ix = np.random.choice(range(nx), p=yhat.ravel())
        x = np.zeros((nx, 1))
        x[ix] = 1
        print(int2Word[ix], end="")


def getCharMappings(vocab):
    int2Word = {}
    word2Int = {}
    idx = 0
    for _char in sorted(vocab):
        int2Word[idx] = _char
        word2Int[_char] = idx
        idx += 1
    return int2Word, word2Int


def getVocab(sentence):
    vocab = []
    for _char in sorted(set(sentence)):
        vocab.append(_char)
    return vocab

# This is our vocab size (Dictionary of words)
vocab = getVocab(sentence)

# Numerical and Character mappings for oneHot-encodings
int2Word, word2Int = getCharMappings(vocab)

# some parameters
na = 100 # number of hidden units
nx = len(vocab) # number of features
maxTx = len(sentence) # Max time-step
ny = nx # number of output (same as input)
stepsize = 0.001 # learn-rate stepsize

wax, waa, wya, ba, by = initWeights(nx, na, ny)

# memory variables for Adagrad (Stolen from Adrej)
mWax, mWaa, mWya = np.zeros_like(wax), np.zeros_like(waa), np.zeros_like(wya)
mBa, mBy = np.zeros_like(ba), np.zeros_like(by)

Train()
