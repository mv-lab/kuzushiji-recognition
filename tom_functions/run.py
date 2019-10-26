import pandas as pd
import numpy as np
import gc
from tqdm import tqdm_notebook as tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tom_functions.scheduler import *
from tom_functions.dataset import *
from tom_functions.models import *
from tom_functions.losses import *


def run_MSE(train_fnames, valid_fnames, device, transform_train, transform_valid,
        fold, load_local_weight_path, output_path, config):
    
    log_cols = ['fold', 'epoch', 'lr', 'loss_trn', 'loss_val',
                'trn_score', 'val_score', 
                'elapsed_time']
    log_df   = pd.DataFrame(columns=log_cols)
    log_counter = 0
    
    criterion = nn.MSELoss().to(device)
   
    for fold in [fold]: #dummy list
        print('fold = ', fold)

        #data loader
        data_trn = train_fnames 
        data_val = valid_fnames 
        
        train_dataset = KuzushijiDetectionDatasetTrain(data=data_trn,
                                                       transform=transform_train)
        valid_dataset = KuzushijiDetectionDatasetTrain(data=data_val,
                                                       transform=transform_valid)
        train_loader  = DataLoader(train_dataset, batch_size=config['trn_batch_size'],
                                   shuffle=True, num_workers=4, pin_memory=True)
        valid_loader  = DataLoader(valid_dataset, batch_size=config['test_batch_size'],
                                   shuffle=False, num_workers=4, pin_memory=True)
        
        #model
        model = UNET_RESNET34(load_weights=True).to(device, torch.float32)
        if load_local_weight_path is not None:
            model.load_state_dict(torch.load(load_local_weight_path+f'model_fold{fold}_bestscore.pth'))
            
        for p in model.parameters():
            p.required_grad = True
        optimizer = optim.Adam(model.parameters(), **config['Adam'])
        
        if config['lr_scheduler_name']=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
        elif config['lr_scheduler_name']=='CosineAnnealingLR':
            scheduler = CosineLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        
        #training
        val_score_best  = -1e+99
        val_score_best2 = -1e+99
        loss_val_best   = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        trn_score_each = 0
        start_time = time.time()
        for epoch in range(1, config['num_epochs']+1):
#             if elapsed_time(start_time) > config['time_limit']:
#                 print('elapsed_time go beyond {} sec'.format(config['time_limit']))
#                 break
            
            if config['lr_scheduler_name']=='CosineAnnealingLR':
                scheduler.step()
                
            print('lr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            model.train()
            running_loss_trn = 0
            counter = 0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            for i,data in enumerate(tk0):
                batch  = data['img'].size(0)
                logits = model(data['img'].to(device, torch.float32, non_blocking=True))
                y      = data['targetmap'][:,0,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y1     = data['targetmap'][:,1,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y2     = data['targetmap'][:,2,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                
                optimizer.zero_grad()
                loss  =      criterion(torch.sigmoid(logits[:,0,:,:]), y)
                loss += 0.01*criterion(logits[:,1,:,:], y1)
                loss += 0.01*criterion(logits[:,2,:,:], y2)
                
                trn_score += dice_sum((torch.sigmoid(logits[:,0,:,:])).cpu().detach().numpy(), 
                                      y.cpu().detach().numpy(), config)
                
                loss.backward()
                optimizer.step()
                running_loss_trn += loss.item() * batch
                counter  += 1
                tk0.set_postfix(loss=(running_loss_trn / (counter * train_loader.batch_size) ))
            epoch_loss_trn = running_loss_trn / len(train_dataset)
            trn_score /= len(train_dataset)
            #release GPU memory cache
            del data, loss,logits,y, y1,y2
            torch.cuda.empty_cache()
            gc.collect()

            #eval
            model.eval()
            loss_val  = 0
            val_score = 0
            for i, data in enumerate(valid_loader):
                batch  = data['img'].size(0)
                logits = model(data['img'].to(device, torch.float32, non_blocking=True))
                y      = data['targetmap'][:,0,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y1     = data['targetmap'][:,1,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y2     = data['targetmap'][:,2,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                loss_val +=      criterion(torch.sigmoid(logits[:,0,:,:]), y).item() * batch
                loss_val += 0.01*criterion(logits[:,1,:,:], y1).item() * batch
                loss_val += 0.01*criterion(logits[:,2,:,:], y2).item() * batch
                
                val_score += dice_sum((torch.sigmoid(logits[:,0,:,:])).cpu().detach().numpy(), 
                                      y.cpu().detach().numpy(), config)
                
                #release GPU memory cache
                del data,logits,y, y1,y2
                torch.cuda.empty_cache()
                gc.collect()
            loss_val  /= len(valid_dataset)
            val_score /= len(valid_dataset)
            
            #logging
            log_df.loc[log_counter,log_cols] = np.array([fold, epoch,
                                                         [ group['lr'] for group in optimizer.param_groups ],
                                                         epoch_loss_trn, loss_val, 
                                                         trn_score, val_score,
                                                         elapsed_time(start_time)])
            log_counter += 1
            
            #monitering
            print('epoch {:.0f} loss_trn = {:.5f}, loss_val = {:.5f}, trn_score = {:.4f}, val_score = {:.4f}'.format(epoch, epoch_loss_trn, loss_val, trn_score, val_score))
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if config['early_stopping']:
                if loss_val < loss_val_best: #val_score > val_score_best:
                    val_score_best = val_score #update
                    loss_val_best  = loss_val #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
                    torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestloss.pth') #save
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > config['patience']:
                    print('early stopping, epoch_best {:.0f}, loss_val_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
                    break
            else:
                torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestloss.pth') #save the latest update
               
            if val_score > val_score_best2:
                val_score_best2 = val_score #update
                torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestscore.pth') #save
                print('model (best score) saved')
            
            if config['lr_scheduler_name']=='ReduceLROnPlateau':
                scheduler.step(loss_val)
                
            #for snapshot ensemble
            if config['lr_scheduler_name']=='CosineAnnealingLR':
                t0 = config['lr_scheduler']['CosineAnnealingLR']['t0']
                if (epoch%t0==0) or (epoch%(t0-1)==0) or (epoch%(t0-2)==0):
                    torch.save(model.state_dict(), output_path+f'model_fold{fold}_epoch{epoch}.pth') #save
                    print(f'model saved epoch{epoch} for snapshot ensemble')
            
            print('')
            
        #best model
        if config['early_stopping']&(counter_ES<=config['patience']):
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
                    
        #save result
        log_df.to_csv(output_path+f'log_{fold}.csv')
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')
    
    res = {
        'log_df':log_df,
    }
    
    return res



def run_HGNet(train_fnames, valid_fnames, device, transform_train, transform_valid,
        fold, load_local_weight_path, output_path, config):
    
    log_cols = ['fold', 'epoch', 'lr', 'loss_trn', 'loss_val',
                'trn_score', 'val_score', 
                'elapsed_time']
    log_df   = pd.DataFrame(columns=log_cols)
    log_counter = 0
    
    criterion = nn.MSELoss().to(device)
   
    for fold in [fold]: #dummy list
        print('fold = ', fold)

        #data loader
        data_trn = train_fnames 
        data_val = valid_fnames 
        
        train_dataset = KuzushijiDetectionDatasetTrain(data=data_trn,
                                                       transform=transform_train)
        valid_dataset = KuzushijiDetectionDatasetTrain(data=data_val,
                                                       transform=transform_valid)
        train_loader  = DataLoader(train_dataset, batch_size=config['trn_batch_size'],
                                   shuffle=True, num_workers=4, pin_memory=True)
        valid_loader  = DataLoader(valid_dataset, batch_size=config['test_batch_size'],
                                   shuffle=False, num_workers=4, pin_memory=True)
        
        #model
        model = HGNET_RESNET34(load_weights=True).to(device, torch.float32)
        if load_local_weight_path is not None:
            model.load_state_dict(torch.load(load_local_weight_path+f'model_fold{fold}_bestscore.pth'))
            
        for p in model.parameters():
            p.required_grad = True
        optimizer = optim.Adam(model.parameters(), **config['Adam'])
        
        if config['lr_scheduler_name']=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
        elif config['lr_scheduler_name']=='CosineAnnealingLR':
            scheduler = CosineLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        
        #training
        val_score_best  = -1e+99
        val_score_best2 = -1e+99
        loss_val_best   = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        trn_score_each = 0
        start_time = time.time()
        for epoch in range(1, config['num_epochs']+1):
#             if elapsed_time(start_time) > config['time_limit']:
#                 print('elapsed_time go beyond {} sec'.format(config['time_limit']))
#                 break
            
            if config['lr_scheduler_name']=='CosineAnnealingLR':
                scheduler.step()
                
            print('lr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            model.train()
            running_loss_trn = 0
            counter = 0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            for i,data in enumerate(tk0):
                batch  = data['img'].size(0)
                logits,logits2 = model(data['img'].to(device, torch.float32, non_blocking=True))
                y      = data['targetmap'][:,0,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y1     = data['targetmap'][:,1,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y2     = data['targetmap'][:,2,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                
                optimizer.zero_grad()
                loss  =      criterion(torch.sigmoid(logits[:,0,:,:]), y)
                loss +=      criterion(torch.sigmoid(logits2[:,0,:,:]), y)
                loss += 0.01*criterion(logits2[:,1,:,:], y1)
                loss += 0.01*criterion(logits2[:,2,:,:], y2)
                
                trn_score += dice_sum((torch.sigmoid(logits2[:,0,:,:])).cpu().detach().numpy(), 
                                      y.cpu().detach().numpy(), config)
                
                loss.backward()
                optimizer.step()
                running_loss_trn += loss.item() * batch
                counter  += 1
                tk0.set_postfix(loss=(running_loss_trn / (counter * train_loader.batch_size) ))
            epoch_loss_trn = running_loss_trn / len(train_dataset)
            trn_score /= len(train_dataset)
            #release GPU memory cache
            del data, loss,logits,y, y1,y2, logits2
            torch.cuda.empty_cache()
            gc.collect()

            #eval
            model.eval()
            loss_val  = 0
            val_score = 0
            for i, data in enumerate(valid_loader):
                batch  = data['img'].size(0)
                logits,logits2 = model(data['img'].to(device, torch.float32, non_blocking=True))
                y      = data['targetmap'][:,0,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y1     = data['targetmap'][:,1,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                y2     = data['targetmap'][:,2,:,:].contiguous().to(device, torch.float32, non_blocking=True)
                loss_val +=      criterion(torch.sigmoid(logits[:,0,:,:]), y).item() * batch
                loss_val +=      criterion(torch.sigmoid(logits2[:,0,:,:]), y).item() * batch
                loss_val += 0.01*criterion(logits2[:,1,:,:], y1).item() * batch
                loss_val += 0.01*criterion(logits2[:,2,:,:], y2).item() * batch
                
                val_score += dice_sum((torch.sigmoid(logits2[:,0,:,:])).cpu().detach().numpy(), 
                                      y.cpu().detach().numpy(), config)
                
                #release GPU memory cache
                del data,logits,y, y1,y2, logits2
                torch.cuda.empty_cache()
                gc.collect()
            loss_val  /= len(valid_dataset)
            val_score /= len(valid_dataset)
            
            #logging
            log_df.loc[log_counter,log_cols] = np.array([fold, epoch,
                                                         [ group['lr'] for group in optimizer.param_groups ],
                                                         epoch_loss_trn, loss_val, 
                                                         trn_score, val_score,
                                                         elapsed_time(start_time)])
            log_counter += 1
            
            #monitering
            print('epoch {:.0f} loss_trn = {:.5f}, loss_val = {:.5f}, trn_score = {:.4f}, val_score = {:.4f}'.format(epoch, epoch_loss_trn, loss_val, trn_score, val_score))
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if config['early_stopping']:
                if loss_val < loss_val_best: #val_score > val_score_best:
                    val_score_best = val_score #update
                    loss_val_best  = loss_val #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
                    torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestloss.pth') #save
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > config['patience']:
                    print('early stopping, epoch_best {:.0f}, loss_val_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
                    break
            else:
                torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestloss.pth') #save the latest update
               
            if val_score > val_score_best2:
                val_score_best2 = val_score #update
                torch.save(model.state_dict(), output_path+f'model_fold{fold}_bestscore.pth') #save
                print('model (best score) saved')
            
            if config['lr_scheduler_name']=='ReduceLROnPlateau':
                scheduler.step(loss_val)
                
            #for snapshot ensemble
            if config['lr_scheduler_name']=='CosineAnnealingLR':
                t0 = config['lr_scheduler']['CosineAnnealingLR']['t0']
                if (epoch%t0==0) or (epoch%(t0-1)==0) or (epoch%(t0-2)==0):
                    torch.save(model.state_dict(), output_path+f'model_fold{fold}_epoch{epoch}.pth') #save
                    print(f'model saved epoch{epoch} for snapshot ensemble')
            
            print('')
            
        #best model
        if config['early_stopping']&(counter_ES<=config['patience']):
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
                    
        #save result
        log_df.to_csv(output_path+f'log_{fold}.csv')
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')
    
    res = {
        'log_df':log_df,
    }
    
    return res