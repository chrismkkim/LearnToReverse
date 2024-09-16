import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import argparse 
import random
import numpy as np
from models import ALIF
from tools_data import GaussianNoise
from tools_target import genTarget, genIdealChoice
from tools_stat import stat_allkinds
import tools_data
from param import load_param
import torch.optim as optim
import pdb
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
import sys
import os
import math

gamma_kk = 0.9
lr_update_step = 3

# trialType_is_random = 'mixed'
trialType_is_random = 'fixed'

# choice_delay is random
choice_delay_is_random = False

# taus_list    = [10, 15, 20, 25, 30, 50, 100]
# noise_list   = [0.5]
# taus_list    = [20.0, 30.0, 50.0, 70.0, 100.0]
taus_list    = [10.0, 20.0, 30.0, 50.0, 70.0]
noise_list   = [0.5, 1.0, 2.0]
fbnoise_list = [0.3, 0.4, 0.5]

ll = int(sys.argv[1])
mm = 0
pp = 0 
qq = int(sys.argv[2]) 
args = load_param(taus_list, noise_list, fbnoise_list, ll, mm, pp, trialType_is_random)

# if choice_delay_is_random:
dirdata = '/data/kimchm/data/spiketorch/data/rate_network/fixedTrialType/tauslong/taus' + str(ll) + '/ntwk' + str(qq) + '/'
# dirdata = '/data/kimchm/data/spiketorch/data/rate_network/fixedTrialType/tauslearn/taus' + str(ll) + '/ntwk' + str(qq) + '/'

if not os.path.isdir(dirdata):
    os.makedirs(dirdata)

# save training results
path_to_model = dirdata + 'spiking_model.pt'
file_loss = open(dirdata + 'file_loss.txt', 'w') 
file_correct = open(dirdata + 'file_correct.txt', 'w') 

# load mnist data
transform = transforms.Compose([transforms.ToTensor(), GaussianNoise(0.0, 0.3, args.SimTime)]) ##################### override MNIST with gaussian noise ####################
mnist_train = datasets.MNIST('/home/kimchm/spiketorch/data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('/home/kimchm/spiketorch/data', train=False, download=False, transform=transform)
train_kwargs = {'batch_size': args.batch_size}
trainloader = torch.utils.data.DataLoader(mnist_train,**train_kwargs, shuffle=True, drop_last=True)

# device
device = torch.device(args.device)
# device = torch.device('cuda')

# generate training variables
phasePre, phasePeri, phasePost, stim0, stim1, A_correct, A_wrong, B_correct, B_wrong = tools_data.gen_variables(dirdata, args, device)

# define model
spiking_model = ALIF(args)
spiking_model.to(device)

# set up an optimizer
optimizer = optim.Adam(spiking_model.parameters(), lr=args.lr)

# scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_update_step, gamma=gamma_kk)

# loss function
cross_entropy = torch.nn.CrossEntropyLoss()

# softmax
softmax_ = torch.nn.Softmax(dim=2)

# train the spiking network
ntrain = 9 # 9
ntest = 1 # 1

def blockloss(outsym, choice_ideal):
    #----- reshape out, target -----#
    # outsym:    batchsize x ntrials x nout --> batchsize * ntrials x nout (first scan 1-axis, i.e. trial)
    # target:    batchsize x ntrials        --> batchsize * ntrials        (first scan 1-axis, i.e. trial)
    # trialType: batchsize x ntrials

    # learn the ideal choice (8/9/2021)
    loss = cross_entropy(outsym.view(args.batch_size * args.ntrials, -1), choice_ideal.view(-1))

    return loss


quit_now = False
start_time = time.time()
for epoch in range(args.Nepoch):
    loss = 0
    correct = 0
    for idx, (img, target_mnist) in enumerate(trainloader):
        
        print(idx)

        target, reversal, hro, trialType = genTarget(args, device, 'twotab_sigmoid')

        if idx % (ntrain+ntest) < ntrain:
            #------ DETECT AUTOGRAD ANOMALY -------#
            # with torch.autograd.detect_anomaly():
            optimizer.zero_grad() 
            out, feedback = spiking_model.forward(stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, args, False)        
            outsym = tools_data.symmetric_output(out)    
            ireverse = stat_allkinds.runPosterior_ideal(feedback, args)
            choice_ideal, choice_reversal = genIdealChoice(args, device, ireverse, hro, trialType, choice_delay_is_random)
            loss = blockloss(outsym, choice_ideal) # target: batch_size x ntrials
            loss.backward()           
            optimizer.step()

            # learning rate scheduler
            scheduler.step()

        if idx % (ntrain+ntest) >= ntrain:
            with torch.no_grad():
                out, feedback = spiking_model.forward(stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, args, False) # out: batch_size x ntrials x Nout
                outsym = tools_data.symmetric_output(out)    
                ireverse = stat_allkinds.runPosterior_ideal(feedback, args)
                choice_ideal, choice_reversal = genIdealChoice(args, device, ireverse, hro, trialType, choice_delay_is_random)
                loss = blockloss(outsym, choice_ideal) # target: batch_size x ntrials
                pred = torch.argmax(outsym, axis=2) # pred: batch_size x ntrials
                correct = pred.eq(target).sum().item() / (args.batch_size * args.ntrials)
                
            if idx % (ntrain+ntest) == ntrain+ntest-1:
                # save test loss / accuracy
                print('epoch: ' + str(epoch) + ',  batch: ' + str(idx))
                print('correct: ', correct)
                file_loss.write(str(loss.item())+'\n')
                file_correct.write(str(correct)+'\n')

                # print elapsed time
                elapsed_time = time.time() - start_time
                print('elapsed: ', elapsed_time, '\n')
                start_time = time.time()

                # save test accuracy (current test only)
                file_correct_tmp = open(dirdata + 'file_correct_' + str(epoch) + '_' + str(idx) + '.txt', 'w') 
                file_correct_tmp.write(str(correct) + '\n')
                file_correct_tmp.close()

                # if epoch > 0:
                path_to_model_tmp = dirdata + 'spiking_model_epoch' + str(epoch) + '_idx' + str(idx) + '.pt'
                torch.save(spiking_model.state_dict(), path_to_model_tmp)

                # quit if
                if correct > args.accuracy:
                    print('Reached idx > ' + str(idx) + '. Save and quit.')
                    file_loss.close()
                    file_correct.close()
                    torch.save(spiking_model.state_dict(), path_to_model)
                    quit_now = True
                    break

                loss = 0
                correct = 0

    if quit_now:
        break

    if epoch == args.Nepoch-1:
        print('Reached the end')
        file_loss.close()
        file_correct.close()
        torch.save(spiking_model.state_dict(), path_to_model)

