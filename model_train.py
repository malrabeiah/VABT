'''
Author: Muhammad Alrabeiah
Jan. 2020
'''
import torch
import torch.nn as nn
import torch.optim as optimizer
# from torch.utils.data import DataLoader
import numpy as np
import time

def modelTrain(net,trn_loader,val_loader,options_dict):
    """

    :param net:
    :param data_samples:
    :param options_dict:
    :return:
    """
    # Optimizer:
    # ----------
    if options_dict['solver'] == 'Adam':
        opt = optimizer.Adam(net.parameters(),
                             lr=options_dict['lr'],
                             weight_decay=options_dict['wd'],
                             amsgrad=True)
    else:
        ValueError('Not recognized solver')

    scheduler = optimizer.lr_scheduler.MultiStepLR(opt,
                                                   milestones=options_dict['lr_sch'],
                                                   gamma=options_dict['lr_drop_factor'])

    # Define training loss:
    # ---------------------
    criterion = nn.CrossEntropyLoss()

    # Initialize training hyper-parameters:
    # -------------------------------------
    itr = 0
    embed = nn.Embedding(options_dict['cb_size'], options_dict['embed_dim'])
    running_train_loss = []
    running_trn_top_1 = []
    running_val_top_1 = []
    train_loss_ind = []
    val_acc_ind = []


    print('------------------------------- Commence Training ---------------------------------')
    t_start = time.clock()
    for epoch in range(options_dict['num_epochs']):

        net.train()
        h = net.initHidden(options_dict['batch_size'])
        h = h.cuda()

        # Training:
        # ---------
        for batch, y in enumerate(trn_loader):

            itr += 1
            init_beams = y[:, :options_dict['inp_seq']].type(torch.LongTensor)
            inp_beams = embed(init_beams)
            inp_beams = inp_beams.cuda()
            targ = y[:, options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']]\
                   .type(torch.LongTensor)
            targ = targ.view(-1)
            targ = targ.cuda()
            batch_size = y.shape[0]
            # if epoch == 1:
            #     print('pause')
            h = h.data[:,:batch_size,:].contiguous().cuda()

            opt.zero_grad()
            out, h = net.forward(inp_beams, h)
            out = out.view(-1,out.shape[-1])
            train_loss = criterion(out, targ)  # (pred, target)
            train_loss.backward()
            opt.step()
            out = out.view(batch_size,options_dict['out_seq'],options_dict['cb_size'])
            pred_beams = torch.argmax(out,dim=2)
            targ = targ.view(batch_size,options_dict['out_seq'])
            top_1_acc = torch.sum( torch.prod(pred_beams == targ, dim=1, dtype=torch.float) ) / targ.shape[0]
            if np.mod(itr, options_dict['coll_cycle']) == 0:  # Data collection cycle
                running_train_loss.append(train_loss.item())
                running_trn_top_1.append(top_1_acc.item())
                train_loss_ind.append(itr)
            if np.mod(itr, options_dict['display_freq']) == 0:  # Display frequency
                print(
                    'Epoch No. {0}--Iteration No. {1}-- Mini-batch loss = {2:10.9f} and Top-1 accuracy = {3:5.4f}'.format(
                    epoch + 1,
                    itr,
                    train_loss.item(),
                    top_1_acc.item())
                    )

            # Validation:
            # -----------
            if np.mod(itr, options_dict['val_freq']) == 0:  # or epoch + 1 == options_dict['num_epochs']:
                net.eval()
                batch_score = 0
                with torch.no_grad():
                    for v_batch, beam in enumerate(val_loader):
                        init_beams = beam[:, :options_dict['inp_seq']].type(torch.LongTensor)
                        inp_beams = embed(init_beams)
                        inp_beams = inp_beams.cuda()
                        batch_size = beam.shape[0]

                        targ = beam[:,options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']]\
                               .type(torch.LongTensor)
                        # if options_dict['out_seq'] == 1:
                        #     targ = targ.view(-1)
                        # else:
                        targ = targ.view(batch_size,options_dict['out_seq'])
                        targ = targ.cuda()
                        h_val = net.initHidden(beam.shape[0]).cuda()
                        out, h_val = net.forward(inp_beams, h_val)
                        pred_beams = torch.argmax(out, dim=2)
                        batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
                    running_val_top_1.append(batch_score.cpu().numpy() / options_dict['test_size'])
                    val_acc_ind.append(itr)
                    print('Validation-- Top-1 accuracy = {0:5.4f}'.format(
                        running_val_top_1[-1])
                    )
                net.train()

        current_lr = scheduler.get_lr()[-1]
        scheduler.step()
        new_lr = scheduler.get_lr()[-1]
        if new_lr != current_lr:
            print('Learning rate reduced to {}'.format(new_lr))

    t_end = time.time()
    train_time = (t_end - t_start)/60
    print('Training lasted {0:6.3f} minutes'.format(train_time))
    print('------------------------ Training Done ------------------------')
    train_info = {'train_loss': running_train_loss,
                  'train_top_1': running_trn_top_1,
                  'val_top_1':running_val_top_1,
                  'train_itr':train_loss_ind,
                  'val_itr':val_acc_ind,
                  'train_time':train_time}

    return [net, options_dict,train_info]
