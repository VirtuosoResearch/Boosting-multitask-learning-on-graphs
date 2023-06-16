import scipy
import numpy as np
from scipy.linalg import pinv
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, svds
import torch
from torch_geometric.utils import degree, to_undirected

def largest_eigenvalues(A, k=1):
    vals, _ = eigs(A, k=k)
    return vals.astype("float")

def largest_singular_values(A, k=1):
    '''
    Return the largest singular value of A
    '''
    _, D, _ = svds(A, k=k)
    return D[::-1]

def compute_laplacian(B, w):
    '''
    B is a numpy array
    W is a numpy array
    returns the laplacian matrix
    '''
    return B.T @ sp.diags(w) @ B

def to_edge_vertex_matrix(edge_index, num_nodes):
    '''
    return edge-vertex incidence matrix ExV
    '''
    rows, cols = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    num_edges = edge_index.size(1)
    B = sp.coo_matrix(
        ([1] * num_edges + [-1] * num_edges,
        (list(range(num_edges)) * 2, list(rows) + list(cols))), 
        shape=[num_edges, num_nodes]
    )    
    return B

def to_edge_index(edge_vertex_matrix, edge_weights):
    '''
    Transform B, w to edge_index and edge_weights as Tensors

    return edge_index, edge_weights (Both are tensors passed into GNNs)
    '''
    edge_index = []
    for i in edge_vertex_matrix: # get rid of this for loop
        head = np.where(i == 1)[0][0]
        tail = np.where(i == -1)[0][0]
        edge_index.append([head, tail])
    edge_index = np.array(edge_index).T
    edge_index = torch.from_numpy(edge_index)
    edge_weights = torch.from_numpy(edge_weights).float()
    return edge_index, edge_weights

def remove_redundant_edges(edge_index):
    new_edge_index = to_undirected(edge_index)
    row, col = new_edge_index
    new_edge_index = new_edge_index[:, row<col] # if (i,j) is an edge, then remove (j,i) if it exists
    return new_edge_index

'''
Uniform sampling
'''
def uniform_sampling(edge_index, w, ratio):
    '''
    edge_index: data.edge_index
    w: edge weights
    ratio: remaining ratio, i.e., (1-p) is the removal ratio
    '''
    ''' Sample 2'''
    # num_edges = edge_index.shape[1]
    # q = int(ratio*num_edges)+1
    # probs = np.ones(edge_index.shape[1])*w
    # probs = probs/probs.sum()
    # # sampled_edges = np.random.binomial(q, probs)
    # samples = np.random.choice(num_edges, q)
    # sampled_edges = np.zeros(num_edges)
    # for idx in samples:
    #     sampled_edges[idx] += 1
    # tilde_edge_index = edge_index[:, sampled_edges!=0]
    # tilde_w = (1/(probs*q) * w * sampled_edges)[sampled_edges!=0]
    # tilde_w = tilde_w*num_edges/tilde_w.sum()
    
    ''' Sample 1'''
    probs = np.ones(edge_index.shape[1])*ratio*w
    probs[probs>1.0] = 1.0

    sampled_edges = np.random.binomial(1, probs)

    tilde_edge_index = edge_index[:, sampled_edges==1]
    tilde_w = (1/probs * w)[sampled_edges==1]
    return tilde_edge_index, tilde_w

'''
GraphSAINT edge sampling
'''
def graphsaint_edge_sampling(edge_index, w, degrees, ratio):
    '''
    ratio: remaining ratio, i.e., (1-p) is the removal ratio
    '''
    scores = 1/degrees[edge_index[0]] + 1/degrees[edge_index[1]]
    num_edges = edge_index.shape[1]

    '''Sample: 3'''
    # q = int(ratio*num_edges)+1
    # probs = scores*w
    # probs = probs/probs.sum()

    # sampled_edges = np.random.binomial(q, probs) 
    # tilde_edge_index = edge_index[:, sampled_edges!=0]
    # tilde_w = (1/(probs*q) * w * sampled_edges)[sampled_edges!=0]

    ''' Sample 2 '''
    # probs = 1/degrees[edge_index[0]] + 1/degrees[edge_index[1]]
    # probs = probs*ratio
    # scale = int(probs.max())+1
    # probs = probs/scale
    # sampled_edges = np.random.binomial(scale, probs)

    ''' Sample 3 '''
    alpha = ratio*num_edges/scores.sum()
    probs = alpha*scores*w # probability of sampling each edge
    probs[probs>1.0] = 1.0
    sampled_edges = np.random.binomial(1, probs)
    tilde_edge_index = edge_index[:, sampled_edges==1]
    tilde_w = (1/probs * w)[sampled_edges==1]
    return tilde_edge_index, tilde_w

'''
Leverage score sampling
'''
def compute_Z(L, W_sqrt, B, k=100, eta=1e-3, max_iters=1000, convergence_after = 100,
                                    tolerance=1e-2, log_every=10, compute_exact_loss=False, edge_index=None, batch_size_Z = 128, batch_size_L = 1024):
    """ Computes the Z matrix using gradient descent
        from https://github.com/WodkaRHR/graph_sparsification_by_effective_resistances 
    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    eta : float
        Step size for the gradient descent.
    max_iters : int
        Maximum number of iterations.
    convergence_after : int
        If the loss did not decrease significantly for this amount of iterations, the gradient descent will abort.
    tolerance : float
        The minimum amount of energy decrease that is expected for iterations. If for a certain number of iterations
        no overall energy decrease is detected, the gradient descent will abort.
    log_every : int
        Log the loss after each log_every iterations.
    compute_exact_loss : bool
        Only for debugging. If set it computes the actual pseudo inverse without down-projection and checks if
        the pairwise distances in Z's columns are the same with respect to the forbenius norm.
        
    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    # Compute the random projection matrix
    # Theoretical value of k := int(np.ceil(np.log(B.shape[1] / epsilon**2))), However the constant could be large.
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = (W_sqrt @ B).tocsr()
    Y_red = Q @ Y

    if compute_exact_loss:
        # Use exact effective resistances to track actual similarity of the pairwise distances
        L_inv = np.linalg.pinv(L.todense())
        Z_gnd = sp.csr_matrix.dot(Y, L_inv)
        pairwise_dist_gnd = Z_gnd.T.dot(Z_gnd)
    
    # Use gradient descent to solve for Z
    Z = np.random.randn(k, L.shape[1])/(2*np.math.sqrt(k))
    best_loss = np.inf
    best_iter = np.inf

    best_Z = None

    for it in range(max_iters):
        batch_Z = np.random.choice(k, batch_size_Z, replace=False)
        batch_L = np.random.choice(L.shape[1], batch_size_L, replace=False)
        
        residual = Y_red[batch_Z][:, batch_L] - Z[batch_Z, :] @ L[:, batch_L]
        loss = np.linalg.norm(residual)
        if it % log_every == 0: 
            leverage_scores = compute_effective_resistances(Z, edge_index.numpy())
            print(f'Loss before iteration {it}: {loss}')
            print(f'Leverage score before iterations: {it}: {leverage_scores.sum()}')
            if compute_exact_loss:
                pairwise_dist = Z.T.dot(Z)
                exact_loss = np.linalg.norm(pairwise_dist - pairwise_dist_gnd)
                print(f'Loss w.r.t. exact pairwise distances {exact_loss}')
        
        if loss + tolerance < best_loss:
            best_loss = loss
            best_iter = it
            best_Z = Z.copy()
        elif it > best_iter + convergence_after:
            # No improvement for 100 iterations
            print(f'Convergence after {it - 1} iterations.')
            break
        
        Z[batch_Z] += eta * residual @ (L[:, batch_L]).T # L.dot(residual.T).T
    print(f"Best loss: {best_loss}")
    return best_Z

def spsolve_Z(L, W_sqrt, B, k=100):
    """ Computes the Z matrix using gradient descent.
    
    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    
    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    # Compute the random projection matrix
    # Theoretical value of k := int(np.ceil(np.log(B.shape[1] / epsilon**2))), However the constant could be large.
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = (W_sqrt @ B).tocsr()
    Y_red = Q @ Y
    L_T = L.T
    
    Z = []
    for i in range(k):
        Z_i = sp.linalg.minres(L_T, Y_red[i, :])[0]
        Z.append(Z_i)
    Z = np.stack(Z, axis=0)
    return Z

def compute_effective_resistances(Z, edges):
    """ Computes the effective resistance for each edge in the graph.
    
    Paramters:
    ----------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate effective resistances.
    edges : tuple
        A tuple of lists indicating the row and column indices of edges.
        
    Returns:
    --------
    R : ndarray, shape [e]
        Effective resistances for each edge.
    """
    rows, cols = edges
    assert(len(rows) == len(cols))
    R = []
    # Compute pairwise distances
    for i, j in zip(rows, cols):
        R.append(np.linalg.norm(Z[:, i] - Z[:, j]) ** 2)
    return np.array(R)

def leverage_score_sampling(edge_index, w, num_nodes, ratio, approx_scores=True, Z_dir = None, k=200):
    '''
    B: edge-vertex incidence matrix
    w: edge weights
    
    Return a sampled graph
    '''
    num_edges = edge_index.shape[1]
    if approx_scores: # solving linear systems
        if Z_dir is not None:
            Z = np.load(Z_dir)
        else:
            B = to_edge_vertex_matrix(edge_index, num_nodes)
            L = B.T @ sp.diags(w) @ B
            W_sqrt = sp.diags(np.sqrt(w))
            Z = spsolve_Z(L, W_sqrt, B, k=k)
        scores = compute_effective_resistances(Z, edge_index.numpy())
    else:
        B = to_edge_vertex_matrix(edge_index, num_nodes)
        L = B.T @ sp.diags(w) @ B
        L = L.todense()
        scores = B @ pinv(L) @ B.T # b^T (L)^{-1} b
        scores = np.diagonal(scores)

    # scores = scores*w
    # q = int(ratio*num_edges)+1
    # sampled_edges = np.argsort(scores)[-q:]
    # tilde_edge_index = edge_index[:, sampled_edges]
    # tilde_w = ( w*num_edges/q)[sampled_edges]
    '''Sample: 3'''
    # q = int(ratio*num_edges)+1
    # probs = scores*w
    # probs = probs/probs.sum()

    # samples = np.random.choice(num_edges, q, replace=True, p=probs)
    # sampled_edges = np.zeros_like(probs)
    # for idx in samples:
    #     sampled_edges[idx] += 1
    # # sampled_edges = np.random.binomial(q, probs) 
    # tilde_edge_index = edge_index[:, sampled_edges!=0]
    # tilde_w = (1/(probs*q) * w * sampled_edges)[sampled_edges!=0]
    # tilde_w = tilde_w*num_edges/tilde_w.sum()

    '''Sample: 2'''
    # alpha = ratio*num_edges/scores.sum()
    # probs = alpha*scores*w # probability of sampling each edge
    # scale = int(probs.max())+1
    # probs = probs/scale

    # sampled_edges = np.random.binomial(scale, probs)
    # print(scale)
    # tilde_edge_index = edge_index[:, sampled_edges!=0]
    # tilde_w = (1/(probs*scale) * w * sampled_edges)[sampled_edges!=0]
    # tilde_w = tilde_w*num_edges/tilde_w.sum()
    
    '''Sample 1: '''
    alpha = ratio*num_edges/scores.sum()
    probs = alpha*scores*w # probability of sampling each edge
    probs[probs>1.0] = 1.0

    sampled_edges = np.random.binomial(1, probs)

    tilde_edge_index = edge_index[:, sampled_edges==1]
    tilde_w = (1/probs * w)[sampled_edges==1]
    return tilde_edge_index, tilde_w

def graph_sparsification(data, method="leverage_score", 
        ratio=1.0, approx_scores=True, k=200, Z_dir = None, 
        **kwargs):
    '''
    Sampling method:
        Inputs: edge_index, **kwargs
        Outputs: new_edge_index, edge_weights
    '''
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if method == "leverage_score":
        edge_index = remove_redundant_edges(edge_index)
        w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()

        tilde_edge_index, tilde_w = leverage_score_sampling(edge_index, w, num_nodes, ratio, approx_scores=approx_scores, Z_dir = Z_dir, k=k)
    elif method == "graph_saint":
        # compute the degrees
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        degrees = degree(edge_index[0], num_nodes)
        degrees = degrees.cpu().numpy()

        # Sample on directed graph (a edge only apears once)
        edge_index = remove_redundant_edges(edge_index)
        w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()        
        tilde_edge_index, tilde_w = graphsaint_edge_sampling(edge_index, w, degrees, ratio)
    elif method == "uniform":
        edge_index = remove_redundant_edges(edge_index)
        w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()
        tilde_edge_index, tilde_w = uniform_sampling(edge_index, w, ratio)
    else:
        raise AssertionError(f"Sampling method {method} is not implemented!")
    
    data.edge_index = tilde_edge_index
    data.edge_weight = torch.Tensor(tilde_w, device=data.edge_index.device)

    # TODO: extend to data_loader in the future
    return data