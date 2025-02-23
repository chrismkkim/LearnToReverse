import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import pdb

def genTarget(args, device):
    target = torch.zeros(args.batch_size, args.ntrials).long().to(device)
    # randomize hro
    #   - random if training
    #   - not random if testing
    if args.hro_random:
        hro = (torch.rand(args.batch_size) < 0.5).long().to(device)
    else:
        hro = torch.zeros(args.batch_size).long().to(device)
    # trialType
    #   - always 'fixed' in this code
    if args.trialType_random == 'fixed':
        trialType = torch.zeros(args.batch_size, args.ntrials).float().to(device)

    # if args.trialType_random == 'random':
    #     trialType = (torch.rand(args.batch_size, args.ntrials) < 0.5).float().to(device)
    # elif args.trialType_random == 'mixed':
    #     half_batch = int(args.batch_size/2)
    #     trialType = torch.zeros(args.batch_size, args.ntrials).float().to(device)
    #     trialType[:half_batch,:] = (torch.rand(half_batch, args.ntrials) < 0.5).float().to(device)
    #     trialType[half_batch:,:] = torch.zeros(half_batch, args.ntrials).float().to(device)

    start_trial = args.reversal_start 
    end_trial = args.reversal_end 
    reversal = torch.randint(start_trial, end_trial, (args.batch_size,)) # reversal trial randomized

    # target probability: smooth transition from target_pr to 1-target_pr
    target_pr_sigmoid = torch.zeros(args.ntrials)

    # target ordering: AB and BA
    for bi in range(args.batch_size):
        target_pr_sigmoid[0:reversal[bi]] = args.target_pr
        target_pr_sigmoid[reversal[bi]:None] = 1 - args.target_pr
        if hro[bi] == 0:
            targetType0 = (torch.rand(args.ntrials) > target_pr_sigmoid).long().to(device)
            targetType1 = (torch.rand(args.ntrials) < target_pr_sigmoid).long().to(device)
            trialType_bi = trialType[bi,:]
            target[bi,:] = ((1 - trialType_bi)*targetType0 + trialType_bi*targetType1).long()
        else:
            targetType0 = (torch.rand(args.ntrials) < target_pr_sigmoid).long().to(device)
            targetType1 = (torch.rand(args.ntrials) > target_pr_sigmoid).long().to(device)
            trialType_bi = trialType[bi,:]
            target[bi,:] = ((1 - trialType_bi)*targetType0 + trialType_bi*targetType1).long()
    
    return target, reversal, hro, trialType


def genIdealChoice(args, device, ireverse, hro, trialType):
    # ideal choice
    choice_ideal = torch.zeros(args.batch_size, args.ntrials).long().to(device)
    choice_delay = args.choice_delay

    # when to reverse choice
    choice_reversal = torch.argmax(ireverse, dim=1) + choice_delay
    choice_reversal[choice_reversal >= args.ntrials] = args.ntrials - 1

    # choice probability
    choice_pr = 1
    choice_pr_sigmoid = torch.zeros(args.ntrials)

    # ideal choice
    for bi in range(args.batch_size):
        choice_pr_sigmoid[0:choice_reversal[bi]] = choice_pr
        choice_pr_sigmoid[choice_reversal[bi]:None] = 1 - choice_pr
        if hro[bi] == 0:
            choiceType0 = (torch.rand(args.ntrials) > choice_pr_sigmoid).long().to(device)
            choiceType1 = (torch.rand(args.ntrials) < choice_pr_sigmoid).long().to(device)
            trialType_bi = trialType[bi,:]
            choice_ideal[bi,:] = ((1 - trialType_bi)*choiceType0 + trialType_bi*choiceType1).long()
        else:
            choiceType0 = (torch.rand(args.ntrials) < choice_pr_sigmoid).long().to(device)
            choiceType1 = (torch.rand(args.ntrials) > choice_pr_sigmoid).long().to(device)
            trialType_bi = trialType[bi,:]
            choice_ideal[bi,:] = ((1 - trialType_bi)*choiceType0 + trialType_bi*choiceType1).long()    

    return choice_ideal, choice_reversal


def replaceParam(param_trained, spiking_model):

    param_trained['SimTime'] = spiking_model.SimTime
    param_trained['trial_duration'] = spiking_model.trial_duration
    param_trained['ntrials'] = spiking_model.ntrials
    param_trained['trial_min'] = spiking_model.trial_min
    param_trained['trial_max'] = spiking_model.trial_max
    param_trained['Nsteps'] = spiking_model.Nsteps
    param_trained['batch_size'] = spiking_model.batch_size
    param_trained['phase_wgt'] = spiking_model.phase_wgt

    return param_trained

def symmetric_output(z):
    # make the output choice symmetric: z -> (z,-z)
    #   outchoice: Nbatch x Ntrials x Nout
    #   concatenate along Nout    
    zsym = torch.cat((z,-z), dim=2)
    return zsym


def symmetric_output_onetrial(z):
    # make the output choice symmetric: z -> (z,-z)
    #   outchoice: Nbatch x Nout
    #   concatenate along Nout    
    zsym = torch.cat((z,-z), dim=1)
    return zsym

def gen_variables(dirdata, args, device):

    path_to_stim0 = dirdata + 'stim0.pt'
    path_to_stim1 = dirdata + 'stim1.pt'
    path_to_A_correct = dirdata + 'A_correct.pt'
    path_to_A_wrong = dirdata + 'A_wrong.pt'
    path_to_B_correct = dirdata + 'B_correct.pt'
    path_to_B_wrong = dirdata + 'B_wrong.pt'

    # cue
    stim0 = args.scale_extinp * torch.randn(args.Ncells).to(device)
    stim1 = args.scale_extinp * torch.randn(args.Ncells).to(device)
    torch.save(stim0, path_to_stim0)
    torch.save(stim1, path_to_stim1)

    # feedback signal (correct / wrong)
    A_correct = args.scale_extinp * torch.randn(args.Ncells).to(device)
    A_wrong = args.scale_extinp * torch.randn(args.Ncells).to(device)
    B_correct = args.scale_extinp * torch.randn(args.Ncells).to(device)
    B_wrong = args.scale_extinp * torch.randn(args.Ncells).to(device)
    
    torch.save(A_correct, path_to_A_correct)
    torch.save(A_wrong, path_to_A_wrong)
    torch.save(B_correct, path_to_B_correct)
    torch.save(B_wrong, path_to_B_wrong)

    return stim0, stim1, A_correct, A_wrong, B_correct, B_wrong



def runPosterior_ideal(feedback, args):

    try:
        feedback = feedback.to('cpu')
    except:
        None

    # feedback: batch_size x ntrial
    batch_size, ntrials = feedback.shape

    ilikelihood = np.ones((batch_size, ntrials, 2))

    for reverse in range(ntrials):

        # target 1 high prob
        for t1 in [0, 1]:

            sequencep = np.zeros_like(feedback)

            # trials 1 to 80
            for ti in range(ntrials):

                if t1 == 0:
                    if ti < reverse:
                        q = args.target_pr
                    elif ti >= reverse:
                        q = 1 - args.target_pr
                elif t1 == 1:
                    if ti < reverse:
                        q = 1 - args.target_pr 
                    elif ti >= reverse:
                        q = args.target_pr

                sequencep[feedback[:,ti] == 0, ti] = 1 - q  # A_wrong (A0)
                sequencep[feedback[:,ti] == 1, ti] = q      # A_correct (A1) 
                sequencep[feedback[:,ti] == 2, ti] = q      # B_wrong (B0)
                sequencep[feedback[:,ti] == 3, ti] = 1- q   # B_correct (B1)

            ilikelihood[:,reverse, t1] = np.prod(sequencep, axis=1) # product along trials

    iZ = np.sum(np.sum(ilikelihood, axis=2), axis=1)
    iZ_unsqueeze = np.expand_dims(iZ, axis=(1,2))
    ilikelihood = ilikelihood / iZ_unsqueeze
    ireverse    = torch.from_numpy(np.sum(ilikelihood, axis=2))

    return ireverse


def runPosterior_behavior(feedback):

    # feedback: batch_size x ntrial
    batch_size, ntrial = feedback.shape

    # feedback = 0 if choice A, not rewarded
    #          = 1 if choice A, rewarded
    #          = 2 if choice B, not rewarded
    #          = 3 if choice B, rewarded
    choice = np.zeros_like(feedback)
    idx_feedbackA = feedback <= 1
    idx_feedbackB = feedback >= 2
    choice[idx_feedbackA] = 0
    choice[idx_feedbackB] = 1

    # parameters
    p = 0.70

    mlikelihood = np.ones((batch_size, ntrial, 2))

    for reverse in range(ntrial):

        # target 1 high prob
        for t1 in [0, 1]:

            msequencep = np.zeros_like(feedback)

            # trials 1 to 80
            for ti in range(ntrial):

                if t1 == 0: 
                    if ti < reverse:
                        q = p
                    else:
                        q = 1 - p
                elif t1 == 1:
                    if ti < reverse:
                        q = 1 - p
                    else:
                        q = p
                        
                idx0 = choice[:,ti] == 0
                # idx1 = choice[:,ti] == 1
                msequencep[idx0,ti] = q
                msequencep[~idx0,ti] = 1 - q

                # if choice[ti] == 0:
                #     msequencep[ti] = q
                # else:
                #     msequencep[ti] = 1 - q

            mlikelihood[:, reverse, t1] = np.prod(msequencep, axis=1) # product along trials

    mZ = np.sum(np.sum(mlikelihood, axis=2), axis=1)
    mZ_unsqueeze = np.expand_dims(mZ, axis=(1,2))
    mlikelihood  = mlikelihood/mZ_unsqueeze
    mreverse    = torch.from_numpy(np.sum(mlikelihood, axis=2))

    return mreverse
