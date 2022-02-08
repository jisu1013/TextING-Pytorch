import argparse

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default='naver', help='type of dataset.')
    parser.add_argument('--model', type=str, default='gnn', help='model string')    
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate') #0.005
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4096, help='size of batches per epoch') #4096 2048 8192
    parser.add_argument('--test_batch_size', type=int, default=4096, help='size of batches per epoch') #4096
    parser.add_argument('--input_dim', type=int, default=100, help='dimension of input')
    parser.add_argument('--hidden', type=int, default=64, help='Number of units in hidden layer') # 32, 64, 96, 128
    parser.add_argument('--steps', type=int, default=2, help='Number of graph layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout') # 0.5
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight for L2 loss on embedding matrix') #5e-4
    parser.add_argument('--early_stopping', type=int, default=-1, help='Tolerance for early stopping (# of epochs)')
    #parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree') # not used
    parser.add_argument('--gpu', type=bool, default=True, help='Using GPU')
    parser.add_argument('--gpu_id', type=int, default=7, help='GPU ID number')

    return parser.parse_args()

###
# Test Result
# Best ACC : 0.7999
# Best Epoch :  20
# Training Time : 10009.5194
# 2048 / 128 / learning=0.01 / dropout 0.5

# Test Result
#  Best ACC : 0.8066
#  Best Epoch :  99
#  Training Time : 2031.6235
# 5000 / 64 / 0.005 / dropout 0.5

#  Test Result
#  Best ACC : 0.8290
#  Best Epoch :  23
#  Training Time : 4040.8412
#  5000 / 64 / 0.001 / dropout 0.0