"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
# from utils.evaluate_utils import contrastive_evaluate
from utils.memory_wo_target import MemoryBank
# from utils.memory import MemoryBank
from utils.train_utils import simclr_train, simclr_pe_train
from utils.utils_wo_target import fill_memory_bank, fill_memory_bank_pe, plot_metrics
# from utils.utils import fill_memory_bank
from termcolor import colored
from time import perf_counter
from byol_pytorch import BYOL
import torch.nn.functional as F
def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--use_batch_sampler', default=False, action="store_true")
parser.add_argument("--gpu_id", default=1, type=int)

args = parser.parse_args()
# fold_finder_re = re.compile(".+fold(\\d).*")
def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.seed, args.use_batch_sampler)
    print(colored(p, 'red'))
    # m = fold_finder_re.match(p["train_db_name"])
    # fold_in_name = m.group(1)
    # assert str(p["fold_idx"]) == fold_in_name
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    
    is_byol = isinstance(model, BYOL)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    # print(model)
    # model = model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    # if 'blink' in p['train_db_name']:
    #     train_dataset, batch_sampler = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
    #                                     split='train+unlabeled') # Split is for stl-10
    #     train_dataloader = get_train_dataloader(p, train_dataset, batch_sampler if p["use_batch_sampler"] else None)
    # else:
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                    split='train+unlabeled') # Split is for stl-10
    train_dataloader = get_train_dataloader(p, train_dataset)
    # val_dataset = get_val_dataset(p, val_transforms)
     
    # val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} train samples'.format(len(train_dataset)))
    # print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    # if 'blink' in p['train_db_name']:
    #     base_dataset, _ = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    # else:
    base_dataset = get_val_dataset(p, val_transforms) # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset)
    # next(iter(base_dataloader))
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'], 
                                p['model_kwargs']['pre_lasts_dim'])
    memory_bank_base.cuda()
    # memory_bank_val = MemoryBank(len(val_dataset),
    #                             p['model_kwargs']['features_dim'],
    #                             p['num_classes'], p['criterion_kwargs']['temperature'],
    #                             p['model_kwargs']['pre_lasts_dim'])
    # memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)
    device = torch.device("cuda", args.gpu_id)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location="cpu")
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_to(optimizer, device)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']
        if "training_losses" in checkpoint:
            training_losses = checkpoint["training_losses"]
        else:
            training_losses = []

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
        training_losses = []
    # if not p["batch_size"] == "npatient" and not is_byol:
    model = torch.nn.parallel.DataParallel(model, [device])
    
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.6f}'.format(lr))
        
        # Train
        print('Train ...')
        start_training = perf_counter()
        if p['setup'] == 'simclr_pe':
            l = simclr_pe_train(train_dataloader, model, criterion, optimizer, epoch)
        else:
            l = simclr_train(train_dataloader, model, criterion, optimizer, epoch, is_byol)
        training_losses.append(l)
        print("training elapsed: %.2f" % (perf_counter() - start_training))
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.module.state_dict(), 
                    'epoch': epoch + 1, "training_losses": training_losses}, p['pretext_checkpoint'])

        # Fill memory bank
        if epoch % 10 == 0:
            print('Fill memory bank for kNN...')
            if p['setup'] == 'simclr_pe':
                assert False, "doesnot support pe with target"
                fill_memory_bank_pe(base_dataloader, model, memory_bank_base)
            else:
                fill_memory_bank(base_dataloader, model, memory_bank_base)
            np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy())
            # plot_metrics({"training losses": training_losses}, "loss", p['loss_plot_path'])
            if len(training_losses) > 0:
                plt.figure()
                # plt.yscale("log")
                plot_metrics({"training losses": training_losses}, "loss", show=False, save_path=p['loss_plot_path'])
                test_images = torch.stack([base_dataset[0]["image"], base_dataset[500]["image"]]).cuda()
                img1 = test_images[0].cpu().numpy()
                img2 = test_images[1].cpu().numpy()
                if is_byol:
                    outputs = model(test_images, return_embedding=True, return_projection=False)
                else:
                    outputs = model(test_images)
                    img1 = img1 * 0.22 + 0.45
                    img2 = img2 * 0.22 + 0.45
                sim = F.cosine_similarity(outputs[0].unsqueeze(0), outputs[1].unsqueeze(0))
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(np.transpose(img1, (1, 2, 0)))
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(np.transpose(img2, (1, 2, 0)))
                plt.axis("off")
                plt.title(f"similarily: {sim.item():.2f} (should be 0)")
                plt.savefig("test_similarity.png")
            else:
                print("no loss data, no loss plot")
            
        # Evaluate (To monitor progress - Not for validation)
        # print('Evaluate ...')
        # top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        # print('Result of kNN evaluation is %.2f' %(top1))  

        # topk = 20
        # print('Mine the nearest neighbors (Top-%d)' % (topk))
        # indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        # np.save(p['topk_neighbors_train_path'], indices)
        # fill_memory_bank(val_dataloader, model, memory_bank_val)
        # np.save(p['pretext_features'].replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())

    # Save final model
    torch.save(model.module.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    if p['setup'] == 'simclr_pe':
        fill_memory_bank_pe(base_dataloader, model, memory_bank_base)
    else:
        fill_memory_bank(base_dataloader, model, memory_bank_base)
    # topk = 20
    # print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    # indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    # np.save(p['topk_neighbors_train_path'], indices)
    # save features
    np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy()) 
    if len(training_losses) > 0:
        plt.figure()
        plt.yscale("log")
        plot_metrics({"training losses": training_losses}, "loss", show=False, save_path=p['loss_plot_path'])
    else:
        print("no loss data, no loss plot")
    # np.save(p['pretext_features'].replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    # print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    # fill_memory_bank(val_dataloader, model, memory_bank_val)
    # topk = 5
    # print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    # indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    # np.save(p['topk_neighbors_val_path'], indices)   

if __name__ == '__main__':
    assert args.gpu_id in [0, 1]
    if args.use_batch_sampler:
        main()
    else:
        with torch.cuda.device(args.gpu_id):
            main()
