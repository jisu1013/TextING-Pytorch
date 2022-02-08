import argparse

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default='naver', help='type of dataset.')
    parser.add_argument('--model', type=str, default='gnn', help='model string')    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate') #0.005
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=5000, help='size of batches per epoch') #4096 2048 8192
    parser.add_argument('--test_batch_size', type=int, default=5000, help='size of batches per epoch') #4096 2048 8192
    parser.add_argument('--input_dim', type=int, default=100, help='dimension of input')
    parser.add_argument('--hidden', type=int, default=64, help='Number of units in hidden layer') # 32, 64, 96, 128
    parser.add_argument('--steps', type=int, default=2, help='Number of graph layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout') # 0.5
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight for L2 loss on embedding matrix') #5e-4
    parser.add_argument('--early_stopping', type=int, default=-1, help='Tolerance for early stopping (# of epochs)')
    parser.add_argument('--gpu', type=bool, default=True, help='Using GPU')
    parser.add_argument('--gpu_id', type=int, default=7, help='GPU ID number')

    return parser.parse_args()