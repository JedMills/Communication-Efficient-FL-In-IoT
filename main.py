"""
Contains FedAvg and CE-FedAvg functions to run experiments.
"""
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model_utils import *
from models import *
import pickle
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy as CatCrossEnt 
from tensorflow.keras.optimizers import SGD, Adam
from data_utils import *
from compress_utils import *
    

def get_out_fname(exp_type, model_type, C, E, W, iid, lr, sparsity, seed):
    """ Turn parameters into a formatted string ending with '.pkl'. """
    return '{}-{}-C-{}-E-{}-W-{}-iid-{}-lr-{}-S-{}-seed-{}.pkl'.format(
            exp_type, model_type, C, E, W, iid, lr, sparsity, seed)


def save_data(fname, data):
    """ Saves data in file with name fname using pickle. """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def run_ce_fed_avg(dataset, model_fn, C, E, B, W, iid, R, s, seed):
    """
    Load dataset and perform R rounds of CE-FedAvg using FedAvg parameters. 
    Saves round errors and accuracies at server in file with exp details.
    
    Parameters:
    dataset (str):          'mnist' or 'cifar'
    model_fn (FedAvgModel): callable *not* instance of model class to use 
    C (float):              fraction of workers used per round
    E (int):                number of local worker epochs of training
    B (int):                worker batch size 
    W (int):                number of workers 
    iid (bool):             iid or non-iid data partition
    R (int):                total number of rounds to run
    s (float):              sparsity 0 <= s < 1
    seed (int):             random seed for trial
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    train, test = load_dataset(dataset, W, iid)

    master_model = model_fn()
    worker_model = model_fn()
    
    # get which model weights correspond to optim params - see get_corr_optims
    corr_optim_idxs = get_corr_optims(worker_model)
    
    central_errs = []
    central_accs = []
    worker_ids = np.arange(W)
    workers_per_round = max(int(C * W), 1)
    
    for r in range(R):
        round_master_weights = master_model.get_weights()
        round_master_optims = master_model.get_optim_weights()
        
        # to store aggregate updates
        agg_model = zeros_like_model(round_master_weights)
        agg_optim = zeros_like_optim(round_master_optims)
        
        round_total_samples = 0
        
        err, acc = master_model.test(test[0], test[1], B)
        print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(r, R, err, acc))
        central_errs.append(err)
        central_accs.append(acc)
        
        # indexes of workers participating in this round
        choices = np.random.choice(worker_ids, workers_per_round, replace=False)
        
        for w in choices:
            # "download" global model
            worker_model.set_weights(round_master_weights)
            worker_model.set_optim_weights(round_master_optims)
            
            w_samples = train[0][w].shape[0]
            round_total_samples += w_samples
            
            # train worker model for given num epochs
            for e in range(E):
                w_x, w_y = shuffle_client_data((train[0][w], train[1][w]))
                worker_model.train(w_x, w_y, B)
            
            model_deltas = minus_model_ws( worker_model.get_weights(),
                                            round_master_weights)
            optim_deltas = minus_optim_ws(  worker_model.get_optim_weights(),
                                            round_master_optims)
            
            # compress and decompress deltas as per Algorithm 1
            if s > 0:
                model_deltas, optim_deltas = compress_ce_fed_avg_deltas(
                                                        corr_optim_idxs,
                                                        model_deltas,
                                                        optim_deltas,
                                                        s)
            
            # add to agg model, weighted by num local samples
            p_deltas = multiply_model_ws(model_deltas, w_samples)
            p_optims = multiply_optim_ws(optim_deltas, w_samples)
            agg_model = add_model_ws(agg_model, p_deltas)
            agg_optim = add_optim_ws(agg_optim, p_optims)
        
        # global model is weighted average of client models
        agg_model = divide_model_ws(agg_model, round_total_samples)
        agg_optim = divide_optim_ws(agg_optim, round_total_samples)
        
        master_model.set_weights(add_model_ws(round_master_weights, agg_model))
        master_model.set_optim_weights(add_optim_ws(round_master_optims,
                                                    agg_optim))
        
    err, acc = master_model.test(test[0], test[1], B)
    print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(R, R, err, acc))
    central_errs.append(err)
    central_accs.append(acc)
        
    # save stats
    fname = get_out_fname(  'ce_fedavg', master_model.name, C, E, 
                            W, iid, s, None, seed)
    save_data(fname, [central_errs, central_accs])


def run_fed_avg(dataset, model_fn, C, E, B, W, iid, R, s, lr, seed):
    """
    Load dataset and perform R rounds of FedAvg using given FedAvg parameters. 
    Saves round errors and accuracies at server in file with exp details.
    
    Parameters:
    dataset (str):          'mnist' or 'cifar'
    model_fn (FedAvgModel): callable *not* instance of model class to use 
    C (float):              fraction of workers used per round
    E (int):                number of local worker epochs of training
    B (int):                worker batch size 
    W (int):                number of workers 
    iid (bool):             iid or non-iid data partition
    R (int):                total number of rounds to run
    s (float):              sparsity 0 <= s < 1
    lr (float):             SGD learning rate used
    seed (int):             random seed for trial
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    train, test = load_dataset(dataset, W, iid)
    
    master_model = model_fn()
    worker_model = model_fn()
    
    central_errs = []
    central_accs = []
    worker_ids = np.arange(W)
    workers_per_round = max(int(C * W), 1)
    
    for r in range(R):
        round_master_weights = master_model.get_weights()

        # to store aggregate updates
        agg_model = zeros_like_model(round_master_weights)
        
        round_total_samples = 0
        
        err, acc = master_model.test(test[0], test[1], B)
        print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(r, R, err, acc))
        central_errs.append(err)
        central_accs.append(acc)
        
        # indexes of workers participating in round
        choices = np.random.choice(worker_ids, workers_per_round, replace=False)
        
        for w in choices:
            # "download" global model
            worker_model.set_weights(round_master_weights)

            w_samples = train[0][w].shape[0]
            round_total_samples += w_samples
            
            # train worker model for given num epochs
            for e in range(E):
                w_x, w_y = shuffle_client_data((train[0][w], train[1][w]))
                worker_model.train(w_x, w_y, B)
            
            worker_deltas = minus_model_ws( worker_model.get_weights(),
                                            round_master_weights)
            
            # compress and decompress deltas as per (part of) Algorithm 1
            if s > 0:
                worker_deltas = compress_fed_avg_deltas(worker_deltas, s)
            
            # add to aggregate model, weighted by local samples
            p_deltas = multiply_model_ws(worker_deltas, w_samples)
            agg_model = add_model_ws(agg_model, p_deltas)
        
        # global model is weighted average of client models
        round_deltas = divide_model_ws(agg_model, round_total_samples)
        master_model.set_weights(add_model_ws(round_master_weights, round_deltas))
        
    err, acc = master_model.test(test[0], test[1], B)
    print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(R, R, err, acc))
    central_errs.append(err)
    central_accs.append(acc)
        
    # save stats
    fname = get_out_fname('fedavg', master_model.name, C, E, W, iid, s, lr, seed)
    save_data(fname, [central_errs, central_accs])


def main():
    # FedAvg / CE-FedAvg hyperparameters
    E           = 1              
    B           = 20            
    W           = 10            # number of workers 
    IID         = False         
    R           = 100           # number of rounds 
    C           = 0.5           
    SPARSITY    = 0.6           
    LR          = 0.2           
    DATASET     = 'mnist' # 'cifar'
    SEEDS       = [0]

    # Use for FedAvg
    # optim = lambda: SGD(LR)
    
    # Use for CE-FedAvg
    optim = lambda: Adam(0.001, 0.9, 0.999)
    
    # Use with MNIST
    model_fn = lambda: MNIST2NNModel(optim, CatCrossEnt, 784, 10)
    # model_fn = lambda: MNISTCNNModel(optim, CatCrossEnt, 28, 1, 10)
    
    # Use with CIFAR
    # model_fn = lambda: CIFARCNNModel(optim, CatCrossEnt, 32, 3, 10)

    for seed in SEEDS:
        # run FedAvg
        # run_fed_avg(DATASET, model_fn, C, E, B, W, IID, R, SPARSITY, LR, seed)
    
        # Run CE-FedAvg
        run_ce_fed_avg(DATASET, model_fn, C, E, B, W, IID, R, SPARSITY, seed)

if __name__ == '__main__':
    main()
