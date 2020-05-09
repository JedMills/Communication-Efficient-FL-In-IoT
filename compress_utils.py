"""
Functions for sparsifying/quantizing weight updates and back.
"""
import numpy as np
from bitarray import bitarray
from model_utils import flatten_model, unflatten_model


def sparsify_flatten(w, s):
    """
    Sparsifies array by retrieving (s-1) fraction of values with the highest 
    magnitude, and returns these values (flat) and the indexes they held in the
    (flattened) array before sparsification. Line 36 of Algorithm 1.
    
    Parameters:
    w (array):  array to sparsify
    s (float):  sparsity (0-1), fraction of values to remove from array
    
    Returns:
    w_vals (array):     values after sparsification, shape [w.size * s]
    indexes (array):    indexs of w_vals in flattened w, shape [w.size * s]
    """
    # ce-fedavg occasionally produces nan's/inf's, so catch them here
    w_flat = np.nan_to_num(w.flatten(), nan=0, posinf=0, neginf=0)
    
    # find value below which values are discarded
    w_abs_flat = np.absolute(w_flat)
    w_sorted = np.sort(w_abs_flat)
    cutoff = w_sorted[int(w_sorted.shape[0] * s)]
    
    # retrieve values and (flat) indexes
    indexes = np.squeeze(np.argwhere(w_abs_flat >= cutoff))
    w_vals = w_flat[indexes]
    
    return w_vals, indexes

    
def vals_idxs_to_array(vals, idxs, shape):
    """
    Takes values produced after sparsification and their (flat) indexes, and 
    returns a sparse array with values in correct places. Used in lines 14 - 
    17 of Algorithm 1.
    
    Parameters:
    vals (array):       non-zero values after sparsifying
    idxs (array):       indexes of vals (after flattening - same shape as vals) 
    shape (iterable):   original shape of array 
    
    Returns:
    sparse (array):     sparse array with vals in (unflattened) idxs positions
    """
    sparse = np.zeros(shape, dtype=np.float32)
    # convert 1D idxs to dimensions of original array
    idxs_2d = np.unravel_index(idxs, shape)
    
    # if original array was 1D, np.unravel_index will give back a [shape, 1]
    # size array, to remove the extra dimension
    if len(shape) == 1:
        idxs_2d = np.squeeze(idxs_2d)
    
    sparse[idxs_2d] = vals
    return sparse


def sparsify_model(arrays, s):
    """
    Sparsifies and flattens each array in arrays with given sparsity s. 1D 
    arrays are not sparsified, but their values, indexs and shapes are still 
    returned. Line 36 of Algorithm 1, but for the whole model.
    
    Parameters:
    arrays (list):  lost of arrays to sparsify
    s (float):      sparsity (0-1)
    
    Returns:
    a_vals (list):  non-zero (array) values of arrays after sparsifying
    idxs (list):    indexes (array) of a_vals after flattening
    shapes (list):  original shapes of arrays
    """
    a_vals = []
    idxs = []
    shapes = []
    
    for a in arrays:
        # don't flatten 1D arrays - often leads to size 0 arrays with high s
        if len(a.shape) > 1:
            sparse_a, sparse_idxs = sparsify_flatten(a, s)
        else:
            sparse_a = np.copy(a)
            sparse_idxs = np.arange(a.size)
        
        a_vals.append(sparse_a)
        idxs.append(sparse_idxs)
        shapes.append(a.shape)
                
    return a_vals, idxs, shapes

    
def vals_to_sparse_weights(ws, idxs, shapes):
    """
    Applies vals_idxs_to_array for each array in ws.
    
    Parameters:
    ws (list):      list of non-zero (array) values after sparsifying 
    idxs (list):    list of corresponding indexes in ws (for flattened)
    shapes (list):  list of original shapes of ws
    
    Returns:
    list of sparse reconstructed arrays
    """
    return [vals_idxs_to_array(ws[i], idxs[i], shapes[i]) 
                for i in range(len(idxs))]


def golomb_encode(vals, s):
    """
    Golomb-encodes a given 1D array of integer values, estimating the optimal 
    number of bits per value using sparsity s, into a bitarray object. See 
    section 3 and Appendix A of reference [14] Sattler et al.
    
    Parameters:
    vals (array):   1D integer array of index values to encode
    s (float):      sparsity of array that the vals are for
    
    Returns:
    bits (bitarray):    golomb-encoded values
    bstar (float):      optimal bits per value, see equation 4
    """
    # differences between each value in vals, each number will be <= the actual
    # values, so are more efficiently encoded
    diffs = vals - np.roll(vals, 1)
    diffs[0] = vals[0]
    
    # equation 4
    bstar = 1 + int(np.log2(np.log(0.618) / np.log(s)))
    bstar2 = 2 ** bstar
    
    # encoding as per Appendix A of ref [14] Sattler et al.
    q = diffs // bstar2
    r = diffs % bstar2
    bits = ['1'*q[i]+'0'+format(r[i],'b').zfill(bstar) 
            for i in range(len(diffs))]
    bits = bitarray(''.join(bits))
    
    return bits, bstar
    
    
def golomb_decode(bits, bstar):
    """
    Decode golomb-encoded bitarray into 1D integer array. Implementation as per
    Appendix A of reference [14] Sattler et al.
    
    Parameters:
    bits (bitarray):    bitarray of golomb-encoded values
    bstar (float):      b* calculated during encoding as per equation 4
    
    Returns:
    1D integer array of decoded values
    """
    i = 0
    q = 0
    vals = []
    bstar2 = 2 ** bstar
    while i < len(bits):
        if bits[i] == False:
            # reaching a 0-bit indicates the end of a single encoded value
            vals.append(q*bstar2 + int(bits[i+1:i+bstar+1].to01(), 2))
            i += bstar + 1
            q = 0
        else:
            # otherwise increment current value
            i += 1
            q += 1
            
    return np.cumsum(vals)


def uni_q(a):
    """
    Uniform quantize 1D float array, as per Algorithm 2.
    
    Parameters:
    a (array):      1D float array to quantize
    
    Returns:
    q (array):      1D uint8 quantized array, of same shape as a
    lzmin (float):  minimum value < 0 in a, 0 if no values < 0
    lzmax (float):  maximum value < 0 in a, 0 if no values < 0
    gzmin (float):  minimum value > 0 in a, 0 if no values > 0
    gzmax (float):  maximum value > 0 in a, 0 if no values > 0
    """
    a_lz = a[a < 0]
    a_gz = a[a > 0]
    q = np.zeros(a.shape, dtype=np.uint8)   # to store quantized values
    lzmin = 0
    lzmax = 0
    gzmin = 0
    gzmax = 0
    
    if a_lz.size > 0:
        lzmin = np.amin(a_lz)       # line 2 of Alg 2
        lzmax = np.amax(a_lz)       # line 3 of Alg 2
        
        if lzmin == lzmax:
            # if only one value < 0 use 0, will be dealt with in uni_dq
            q[a < 0] = 0
        else:
            # line 7 of Alg 2
            q[a < 0] = np.floor((127 / (lzmax - lzmin)) * (a_lz - lzmin))
        
    if a_gz.size > 0:
        gzmin = np.amin(a_gz)       # line 4 of Alg 2
        gzmax = np.amax(a_gz)       # line 5 of Alg 2
        
        if gzmin == gzmax:
            # if only one value > 0 use 255, will be dealt with in uni_dq
            q[a > 0] = 255
        else:
            # line 8 of Alg 2
            q[a > 0] = 128 + np.floor((127 / (gzmax - gzmin)) * (a_gz - gzmin))
    
    return q, lzmin, lzmax, gzmin, gzmax


def uni_dq(q, lzmin, lzmax, gzmin, gzmax):
    """
    Uniform dequantize 1D uint8 array, as per Algorithm 2.
    
    Parameters:
    q (array):      1D uint8 array quantized using uni_q
    lzmin (float):  min value < 0, from uni_q
    lzmax (float):  max value < 0, from uni_q
    gzmin (float):  min value > 0, from uni_q
    gzmax (float):  max value > 0, from uni_q
    
    Returns:
    dq (array):     1D dequantized float32 array of same shape as q
    """
    dq = np.zeros(q.shape, dtype=np.float32)    # to store dequantized values
    
    if lzmin == lzmax:
        # lzmin == lzmax if there was only one value < 0 in original array
        dq[q < 128] = lzmin
    else:
        # line 14 of Alg 2
        dq[q < 128] = (((lzmax - lzmin) / 127) * q[q < 128]) + lzmin
        
    if gzmin == gzmax:
        # gzmin == gzmax if there was only one value > 0 in original array
        dq[q > 127] = gzmin
    else:
        # line 15 of Alg 2
        dq[q > 127] = (((gzmax - gzmin) / 127) * (q[q > 127] - 128)) + gzmin
        
    return dq
    
    
def exp_q(a):
    """
    Exponential Quantize float array a to uint8, as per Algorithm 3. Contains 
    checks to remove any potential NaN's, Inf's.
    
    Parameters:
    a (array):  float array to quantize 
    
    Returns:
    q (array):  uint8 quantized array, of same shape as a
    b (float):  base for exponentiation, needed for dequantization
    """
    # Adam very occasionally produces NaN's/Inf's during ce-fedavg, remove these
    a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)
    # calc smallest base for quantization, line 2 of Alg 3
    b = np.power(np.abs(a[a != 0]).min(), -1.0/127.0)
    
    q = np.zeros(a.shape, dtype=np.uint8)
    
    pows = np.log(np.abs(a), where=a!=0) / np.log(b)        # line 4 of Alg 3
    # very small values in b sometimes produce overflows in np.log, resulting
    # in NaN's/Inf's, so remove them here 
    pows = np.nan_to_num(pows, nan=1, posinf=1, neginf=1)   
    
    pows[pows > -1] = -1
    pows[pows < -127] = -127
    gz = a > 0
    lz = a < 0
    q[lz] = np.abs(pows[lz])        # line 5 of Alg 3
    q[gz] = 128 + np.abs(pows[gz])  # line 6 of Alg 3
    
    return q, b
    
    
def exp_dq(q, b):
    """
    Exponentially dequantize uint8 array to float32, as per Algorithm 3.
    
    Parameters:
    q (array):  uint8 from uni_q array to dequantize
    b (float):  base from uni_q
    
    Returns:
    d (array):  float32 dequantized array, of same shape as q
    """
    d = np.zeros(q.shape, dtype=np.float32)
    lz = (q < 128) * (q != 0)
    gz = q >= 128
    
    # cast to signed int to allow negative powers
    qf = q.astype(np.int16)
    d[lz] = -np.power(b, -qf[lz])       # line 12 of Alg 3
    d[gz] = np.power(b, 128.0 - qf[gz]) # line 13 of Alg 3

    return d


def compress_fed_avg_deltas(deltas, s):
    """
    Compressing then decompressing model deltas from worker to server. Each 
    matrix in the model is sparsified and flattened, the non-zero values 
    are quantized using Uniform Quantization (see Algorithm 2), and the sparse 
    indexes are encoded using golomb encoding. The indexes are then decoded, 
    values dequantized and restructured into sparse arrays. A lot of the steps 
    could be removed and matrices simply sparsified, but this part of the 
    simulation represents a tiny fraction of the total running time (it is 
    mostly spent in training).
    
    Parameters:
    deltas (list):  model deltas (list of layers of arrays) to sparsify
    s (float):      sparsity

    Returns:
    model after compressing and decompressing
    """
    # turn model (2d list of layers) to 1d to make subsequent steps simpler
    flat_deltas = flatten_model(deltas)
    # sparsify and extract non-zero weights in deltas, their indexes, and 
    # the original shapes of the arrays
    big_deltas, idxs, shapes = sparsify_model(flat_deltas, s)    
    # golomb encode indexes of sparse arrays
    gs_bstars = [golomb_encode(i, s) for i in idxs]
    # uniform-quantize sparsified values
    qs_lzs_gzs = [uni_q(d) for d in big_deltas]
    
    # decode golomb indexes
    idxs_decompr = [golomb_decode(g, b_star) for (g, b_star) in gs_bstars]
    # dequantize values
    deltas_dq = [uni_dq(*q_lz_gz) for q_lz_gz in qs_lzs_gzs]
    # return non-zero dequantized weight values to 1d list of sparse arrays
    flat_deltas_afr_compr = vals_to_sparse_weights(   deltas_dq, 
                                              idxs_decompr, 
                                              shapes)
    # turn 1d list of decompressed weight deltas to 2d list of layers
    return unflatten_model(flat_deltas_afr_compr, deltas)


def compress_ce_fed_avg_deltas(corr_w_idxs, w_deltas, o_deltas, s):
    """
    Performs compression followed by decompression of CE-FedAvg values (model 
    weights and optimizer parameters) according to Algorithm 1. Requires 1D 
    array matching optimizer parameters to model weights, see the docstring of
    get_corr_optims for details.
    
    Parameters:
    corr_w_idxs (array):    1d array of model weights matching optimizer params
    w_deltas (list):        2d list of model layer weight deltas
    o_deltas (list):        1d list of optimizer param deltas
    s (float):              sparsity 0 <= s < 1
    
    Returns:
    decompr_model (list):   model deltas (2d list of layers) after decompression
    decompr_optim (list):   optim deltas (1d list of params) after decompression
    """
    # flatten model deltas (2d list of layers) to 1d list of weights
    flat_w_deltas = flatten_model(w_deltas)
    # extract non-zero weights after sparsification, their (flat) indexes and 
    # original shapes, line 36 of Algorithm 1
    nez_deltas, idxs, shapes = sparsify_model(flat_w_deltas, s)
    
    # golomb encode (flat) indexes of non-zero sparse delts, line 37 of Alg 1
    gs_bstars = [golomb_encode(i, s) for i in idxs]
    
    # uniform encode model weight deltas, line 38 of Alg 1
    qs_lzs_gzs = [uni_q(d) for d in nez_deltas]
    
    # Extract optim deltas using same indexes as the sparse weights: we find 
    # sparsifying these independently causes many exploding gradients as 
    # described in section IV - lines 39 and 40 of Alg 1
    qs_bs = []
    for (i, c_idx) in enumerate(corr_w_idxs):
        if c_idx == -1:
            qs_bs.append(exp_q(o_deltas[i].flatten()))
        else:
            qs_bs.append(exp_q(np.take(o_deltas[i].flatten(), idxs[c_idx])))
    
    # de-encode golomb-encoded indexes, line 13 of Alg 1
    idxs_decompr = [golomb_decode(g, b_star) for (g, b_star) in gs_bstars]
    # dequantize model deltas, line 14 of Alg 1
    deltas_dq = [uni_dq(*q_lz_gz) for q_lz_gz in qs_lzs_gzs]
    # turn dequantized values to sparse arrays
    deltas_afr_compr = vals_to_sparse_weights(deltas_dq, idxs_decompr, shapes)
    # dequantize optimizer values, lines 16 and 17 of Alg 1
    o_deltas_dq = [exp_dq(q, b) for (q, b) in qs_bs]
    
    # turn dequantized optimizer values into sparse arrays using model shapes
    decompr_optim = []
    for (i, c_idx) in enumerate(corr_w_idxs):
        if c_idx == -1:
            decompr_optim.append(o_deltas_dq[i].reshape(o_deltas[i].shape))
        else:
            decompr_optim.append(vals_idxs_to_array(o_deltas_dq[i],
                                                    idxs[c_idx],
                                                    shapes[c_idx]))
    # turn 1d list of model arrays to 2d layer list 
    decompr_model = unflatten_model(deltas_afr_compr, w_deltas)
         
    return decompr_model, decompr_optim