import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import parse_args
import random
from data_loader.build import build_loader
from utils import *
from models import MLP, GNN
import warnings
import numpy as np
from time import time
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)


def train(train_loader, test_loader, val_loader, args_config, device, output_dim):

    n_epochs = args_config.epochs
    gnn = GNN(args_config.input_dim, output_dim) 
    gnn_optimizer = torch.optim.Adam(gnn.parameters(), args_config.learning_rate)
    
    gnn = gnn.cuda()    

    best_val = 0.
    best_epoch = 0
    best_acc = 0. 
    batch_size = args_config.batch_size  
    n_batch = len(train_loader)  

    # Train
    print('Start train')
    for epochs in tqdm(range(n_epochs)):
        train_loss = 0.
        for batch in tqdm(train_loader):            
            gnn_optimizer.zero_grad()     
            adj, mask, emb, labels = batch   
            outputs = gnn(emb.cuda(), adj.cuda(), mask.cuda())    
            bce_loss_batch = gnn.bce_loss(outputs, labels.float().cuda()) 
            l2_loss_batch = gnn._l2_loss() 
            loss_batch = (l2_loss_batch + bce_loss_batch) 
            train_loss += loss_batch
            #print(' loss_batch: {:.4f}'.format(loss_batch.item()))
            loss_batch.backward()        
            gnn_optimizer.step()        
        train_loss = train_loss / n_batch
        print(' loss_batch: {:.4f}'.format(train_loss.item()))
        f = open('./train_loss.txt','a')
        f.write(str(train_loss.item())+'\n')
        f.close()

        # Validation
        with torch.no_grad():
            val_acc = 0.
            for batch in tqdm(val_loader):
                val_adj, val_mask, val_emb, val_labels = batch     
                outputs = gnn(val_emb.cuda(), val_adj.cuda(), val_mask.cuda())            
                preds = gnn.predict(outputs)
                val_acc += accuracy(preds, val_labels.float().cuda())
            val_acc = val_acc / len(val_loader)
            print(' validation acc : {:.4f}'.format(val_acc.item()))
            if best_val <= val_acc:
                best_val = val_acc
        
        # Test
        #_tbar = tqdm(test_loader, ascii=True)
        with torch.no_grad():
            test_acc = 0.
            for batch in tqdm(test_loader):
                test_adj, test_mask, test_emb, test_labels = batch 
                outputs = gnn(test_emb.cuda(), test_adj.cuda(), test_mask.cuda())
                preds = gnn.predict(outputs)
                test_acc += accuracy(preds, test_labels.float().cuda())
            test_acc = test_acc / len(test_loader)
            print(' test acc : {:.4f}'.format(test_acc.item()))
            if best_acc <= test_acc:
                best_acc = test_acc
                best_epoch = epochs
                best_preds = preds
                best_emb = outputs
    
    return best_acc, best_epoch, best_preds, best_emb


if __name__ == '__main__':
    
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)

    """initialize args and dataset"""
    args_config = parse_args()
    
    """set gpu id"""
    args_config.gpu = args_config.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if args_config.gpu else "cpu")

    if args_config.gpu:
        print('Using GPU')
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(args_config.gpu_id)
    
    print("Load Data")
    train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y = load_data(args_config.dataset)
    
    # preprocessing
    train_adj, train_mask = preprocess_adj(train_adj)
    train_embed = preprocess_features(train_embed)
    val_adj, val_mask = preprocess_adj(val_adj)
    val_embed = preprocess_features(val_embed)
    test_adj, test_mask = preprocess_adj(test_adj)
    test_embed = preprocess_features(test_embed)

    print("Build Data Loader")
    train_loader, test_loader, val_loader = build_loader(args_config=args_config, train_adj=train_adj, train_mask=train_mask, train_emb=train_embed, train_y=train_y,
                                             test_adj=test_adj, test_mask=test_mask, test_emb=test_embed, test_y=test_y,
                                             val_adj=val_adj, val_mask=val_mask, val_emb=val_embed, val_y=val_y)

    t1 = time()

    best_acc, best_epoch, best_preds, best_emb = train(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        args_config=args_config,
        device=device,
        output_dim = train_y.shape[1]
    )

    print(' Test Result')
    print(' Best ACC : {:.4f}'.format(best_acc.item()))
    print(' Best Epoch : ', best_epoch)
    print(' Training Time : {:.4f}'.format(time()-t1))


