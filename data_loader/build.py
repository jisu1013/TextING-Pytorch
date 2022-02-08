from torch.utils.data import DataLoader
from data_loader.data_loader import TrainGenerator, TestGenerator, ValGenerator

def build_loader(args_config, train_adj, train_mask, train_emb, train_y, 
                test_adj, test_mask, test_emb, test_y, val_adj, val_mask, val_emb, val_y,):

    print('batch size: ', args_config.batch_size)
    
    train_generator = TrainGenerator(args_config=args_config, adj=train_adj, mask=train_mask, emb=train_emb, y=train_y)
    train_loader = DataLoader(
        train_generator,
        batch_size=args_config.batch_size,
        num_workers=4,
    )

    test_generator = TestGenerator(args_config=args_config, adj=test_adj, mask=test_mask, emb=test_emb, y=test_y)
    test_loader = DataLoader(
        test_generator,
        batch_size=args_config.test_batch_size,
        num_workers=4,
    )

    val_generator = ValGenerator(args_config=args_config, adj=val_adj, mask=val_mask, emb=val_emb, y=val_y)
    val_loader = DataLoader(
        val_generator,
        batch_size=args_config.test_batch_size,
        num_workers=4,
    )

    return train_loader, test_loader, val_loader