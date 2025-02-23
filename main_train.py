import torch
import torch.autograd
import torch.optim as optim
import numpy as np
from models import RNN
import tools_data
from param import load_param
import time
import os

dirdata = '/data/kimchm/data/spiketorch/data/rate_network/github/'
mode = 'train'
args = load_param(mode)


if not os.path.isdir(dirdata):
    os.makedirs(dirdata)

# save training results
path_to_model = dirdata + 'rnn_model.pt'
file_loss = open(dirdata + 'file_loss.txt', 'w') 
file_correct = open(dirdata + 'file_correct.txt', 'w') 

# device
device = torch.device(args.device)

# generate training variables
stim0, stim1, A_correct, A_wrong, B_correct, B_wrong = tools_data.gen_variables(dirdata, args, device)

# define model
rnn_model = RNN(args)
rnn_model.to(device)

# set up an optimizer
optimizer = optim.Adam(rnn_model.parameters(), lr=args.lr)

# scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

# loss function
cross_entropy = torch.nn.CrossEntropyLoss()

# train the spiking network
ntrain = 9 # 9
ntest = 1 # 1

def blockloss(outsym, choice_ideal, fixation, fixation_lambda):

    # fixate output to 0 (start to feedback_end)
    fixation_mse = torch.sqrt(torch.mean(fixation))
    # learn the ideal choice (8/9/2021)
    choice_xent = cross_entropy(outsym.view(args.batch_size * args.ntrials, -1), choice_ideal.view(-1))
    # total loss
    loss = choice_xent + fixation_lambda * fixation_mse

    return loss


quit_now = False
start_time = time.time()
Niter = 240
for epoch in range(args.Nepoch):
    loss = 0
    correct = 0
    for idx in range(Niter):
        
        print(idx)

        target, reversal, hro, trialType = tools_data.genTarget(args, device)

        if idx % (ntrain+ntest) < ntrain:
            optimizer.zero_grad() 
            out, feedback, fixation = rnn_model.forward(stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, args, False)        
            outsym = tools_data.symmetric_output(out)    
            ireverse = tools_data.runPosterior_ideal(feedback, args)
            choice_ideal, choice_reversal = tools_data.genIdealChoice(args, device, ireverse, hro, trialType)
            loss = blockloss(outsym, choice_ideal, fixation, args.fixation_penalty) # target: batch_size x ntrials
            loss.backward()           
            optimizer.step()

            # learning rate scheduler
            scheduler.step()

        if idx % (ntrain+ntest) >= ntrain:
            with torch.no_grad():
                out, feedback, fixation = rnn_model.forward(stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, args, False) # out: batch_size x ntrials x Nout
                outsym = tools_data.symmetric_output(out)    
                ireverse = tools_data.runPosterior_ideal(feedback, args)
                choice_ideal, choice_reversal = tools_data.genIdealChoice(args, device, ireverse, hro, trialType)
                loss = blockloss(outsym, choice_ideal, fixation, args.fixation_penalty) # target: batch_size x ntrials
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
                torch.save(rnn_model.state_dict(), path_to_model_tmp)

                # quit if
                if correct > args.accuracy:
                    print('Reached idx > ' + str(idx) + '. Save and quit.')
                    file_loss.close()
                    file_correct.close()
                    torch.save(rnn_model.state_dict(), path_to_model)
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
        torch.save(rnn_model.state_dict(), path_to_model)

