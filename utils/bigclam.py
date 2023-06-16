''' BigClaim '''
import numpy as np

def sigm(x):
    return np.exp(-x)/(1.-np.exp(-x))

def log_likelihood(F, A):
    """implements equation 2 of 
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    A_soft = F.dot(F.T)

    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A*np.log(1.-np.exp(-1.*A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1-A)*A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli

def gradient(F, A):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    scores = F @ F.T
    postive_gradients = sigm(scores)
    tmp_A = A * postive_gradients
    postive_gradients = tmp_A @ F # / np.sum(tmp_A, axis=1, keepdims=True)

    neg_A = (1-A)
    negative_gradients = neg_A @ F # / np.sum(neg_A, axis=1, keepdims=True)
    grad = postive_gradients - negative_gradients

    return grad

def train(A, C, batch_size=1000, iterations = 50, lr=0.005):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N,C)
    for n in range(iterations):  
        node_ids = np.random.permutation(N)
        ll = 0
        for i in range(0, N+batch_size, batch_size):
            batched_nodes = node_ids[i: i+batch_size]
            batched_F = F[batched_nodes]
            batched_A = A[batched_nodes, :][:, batched_nodes]
            grad = gradient(batched_F, batched_A)

            F[batched_nodes] += lr*grad

            F[batched_nodes] = np.maximum(0.001, F[batched_nodes]) # F should be nonnegative

            ll += log_likelihood(batched_F, batched_A)
        # print(F)
        print('At step %5i log-likelihood is %5.3f'%(n, ll))
    return F