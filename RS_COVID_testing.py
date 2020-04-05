from sklearn.linear_model import Lasso
from scipy import sparse as sp
from math import log, log2, ceil
from pyfinite import ffield
import random
import png
from functools import reduce
import numpy as np
import time
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from statistics import mean

@ignore_warnings(category=ConvergenceWarning)
def simulate_SR_code(
        N,              # Pop size
        d,              # Num defectives
        est_d = 1,      # Estimated prevalence (%)
        m = None,       # Number of layers
        b = None,       # Inverse of size of batches (1 = max size, 2 = half max)
        verbose = False,
        alpha=0.001,
        pt=0.1):

    # Arithmetic tables for the finite field of order 16
    FFsum =[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14],
            [2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13],
            [3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12],
            [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11],
            [5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10],
            [6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9],
            [7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8],
            [8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7],
            [9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6],
            [10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5],
            [11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4],
            [12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3],
            [13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2],
            [14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1],
            [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
            ]
    
    FFmul =[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            [0,2,4,6,8,10,12,14,3,1,7,5,11,9,15,13],
            [0,3,6,5,12,15,10,9,11,8,13,14,7,4,1,2],
            [0,4,8,12,3,7,11,15,6,2,14,10,5,1,13,9],
            [0,5,10,15,7,2,13,8,14,11,4,1,9,12,3,6],
            [0,6,12,10,11,13,7,1,5,3,9,15,14,8,2,4],
            [0,7,14,9,15,8,1,6,13,10,3,4,2,5,12,11],
            [0,8,3,11,6,14,5,13,12,4,15,7,10,2,9,1],
            [0,9,1,8,2,11,3,10,4,13,5,12,6,15,7,14],
            [0,10,7,13,14,4,9,3,15,5,8,2,1,11,6,12],
            [0,11,5,14,10,1,15,4,7,12,2,9,13,6,8,3],
            [0,12,11,7,5,9,14,2,10,6,1,13,15,3,4,8],
            [0,13,9,4,1,12,8,5,2,15,11,6,3,14,10,7],
            [0,14,15,1,13,3,2,12,9,7,6,8,4,10,11,5],
            [0,15,13,2,9,6,4,11,1,14,12,3,8,7,5,10]
            ]
    
    def power(base,exp):
        res = 1
        for i in range(exp):
            res = FFmul[res][base]
        return res
    
    def evaluate(poly, x):
        # Evaluate polynomial with coeficients in poly at x
        res = 0
        for exp,coef in enumerate(poly):
            res = FFsum[res][FFmul[coef][power(x,exp)]]
    
        return res
    
    def printv(x):
        if verbose:
            print(x)
    
    # Empirically found good parameters
    good_params = {
        0.1: {'m':4, 'b':1},
        1: {'m':5, 'b':1},
        5: {'m':6,'b':2}
    }
    
    #===== Start ========
    
    start = time.time()
    
    true_pos = 0
    false_pos = 0
    false_neg = 0
    num_tests = 0
    pool_size = []

    if m is None:
        m = good_params[est_d]['m']
    if b is None:
        b = good_params[est_d]['b']
    
    # Randomize the infected and store them as an N-dimensional binary vector x
    x = [0 for i in range(N)]
    D = 0
    while D < d:
        num = random.randint(0,N-1)
        if x[num] == 0:
            x[num] = 1
            D += 1
    
    # We divide the population in batches of size at most 512. This is necessary so
    # that pool sizes won't exceed 32, our estimated upper bound
    batch_size = 16*32
    iter = 0
    Ncount = 0
    while(Ncount < N):
    
        printv(f"Beginning iteration {iter}")
    
        n = ceil(batch_size/b if N-Ncount > batch_size/b else N-Ncount)
        Ncount += n
        # n = size of current batch
    
        printv(f"n = {n}")
            
        # TODO test different values of q (Note that this requires changing
        # the finite field
        q = 16
        
        # Choose minimum k that verifies the constraint
        # n < q^k => log_q(n) < k
        k = ceil(log(n)/log(q))
        
        # Assign polynomials to x. Count carries the current polynomial
        count = [0 for i in range(k)]
        # polys[i] is the polynomial of individual i
        polys = []
        
        for i in range(n):
            polys.append(count.copy())
            acc = 1
            for j in range(k):
                count[j] = count[j] + acc
                acc = 0
                if count[j] >= q:
                    count[j] = count[j] % q
                    acc = 1
        
            #print(polys[i])
        
        # Select the current batch
        x_act = x[Ncount-n:Ncount]
        
        # Layers
        
        M = []
        
        for j in range(m):
            M.append([[0 for i in range(n)] for i in range(q)])
            # Layer j
            for i in range(n):
                # Column i
                l = evaluate(polys[i],j)
                M[j][l][i] = 1
        
        M_flat = [row for layer in M for row in layer]
        image = [[255 if item == 0 else 1 for item in row] for row in M_flat]
        png.from_array(image, 'L').save(f"designs/design{iter}.png")
        
        printv(f"Design saved to 'design{iter}.png'. Coding...")
        
        y = [reduce(lambda a,b: min(a + b[0]*b[1],1), zip(row,x_act), 0) for row in M_flat]
        #print(y)
        
        printv("Decoding...")

        # TODO better decoder?
        model = Lasso(alpha=alpha, positive=True)
        model.fit(sp.coo_matrix(M_flat),y)

        # Prediction vector. If a value in x_pred is > pt, we take it as positive
        x_pred = model.coef_
    
        true_positives = sum([(1 if real==1 and pred > pt else 0) for real,pred in zip(x_act,x_pred)])
        true_pos += true_positives
        false_positives = sum([(1 if real==0 and pred > pt else 0) for real,pred in zip(x_act,x_pred)])
        false_pos += false_positives
        false_negatives = sum([1 if real==1 and pred < pt else 0 for real,pred in zip(x_act,x_pred)])
        false_neg += false_negatives
        num_tests += q*m
        pool_size.append(ceil(n/q))
        iter += 1
        printv(f"Done in {model.n_iter_} iterations")
    
    ttime = time.time() - start
    printv(f"Done in {ttime} seconds")
    printv(f"Tamaño de la población: {N}, número de infectados: {d}")
    printv(f"Número de positivos: {true_pos + false_pos}")
    printv(f"True positives: {true_pos}")
    printv(f"False positives: {false_pos}")
    printv(f"False negatives: {false_neg}")
    printv(f"Número de tests realizados: {num_tests}")
    printv(f"Mean pool size: {mean(pool_size)}")
    printv(f"Max pool size: {max(pool_size)}")                                               
    return {'tp':true_pos, 'fp':false_pos, 'fn':false_neg, 'nt':num_tests, 'ps':max(pool_size), 'st':true_pos/(true_pos+false_neg)}



def test_params():
    for i in range(10):
        print(simulate_SR_code(16*32, 0.01*16*32, 0.1, m=6, b=2))

def grid_test_params():

    for prev in [0.001, 0.01, 0.05]:
        for i in range(1,16):
            for j in [1,1.2,1.4,1.6,2,3]:
                print(f"{prev*100} - {i} - {j}")
                print(simulate_SR_code(16*32, prev*16*32, max(ceil(prev*100),1), m=i, b=j))
                print(simulate_SR_code(16*32, prev*16*32, max(ceil(prev*100),1), m=i, b=j))



def simulate():
    results = dict()

    N = 100*2*16*32 # 100*2^10 ~= 100k

    for prev in [0.001, 0.01, 0.05]:
        results[prev] = []
        for i in range(33):
            print(f"{prev*100} - {i}")
            results[prev].append(simulate_SR_code(N, prev*N, max(ceil(prev*100),1)))

        for fun in [mean, max, min]:
            tp = fun([point['tp'] for point in results[prev]])
            fp = fun([point['fp'] for point in results[prev]])
            fn = fun([point['fn'] for point in results[prev]])
            nt = fun([point['nt'] for point in results[prev]])
            ps = fun([point['ps'] for point in results[prev]])
            sens = fun([point['tp']/(point['tp']+point['fn']) for point in results[prev]])
            print(f"{prev*100}%, {fun.__name__}: tp: {tp}, fp: {fp}, fn: {fn}\n" \
                    f"sensibilidad: {sens}, ntests:{nt}, pool size: {ps}")


if __name__== "__main__":
    simulate()
