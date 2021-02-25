import numpy
import pickle
def load(file):
    with open(file, 'rb') as f:
        pkl = pickle.load(f)
    w1,w2,w3 = pkl['W1'].tolist(), pkl['W2'].tolist(), pkl['W3'].tolist()
    b1,b2,b3 = pkl['b1'].tolist(), pkl['b2'].tolist(), pkl['b3'].tolist()
    return w1,w2,w3,b1,b2,b3
