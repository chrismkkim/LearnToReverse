import argparse 
import numpy as np

def load_param(mode):

    '''
    parameters to adjust
    hro_random   = True (train), False (test)
    ntrials      = 24 (train), 36 (test)
    which_device = gpu (train), cpu (test)
    '''
    # batch size
    Nbatch = 256 # 256

    if mode == 'train':
        which_device = 'cuda'
        hro_random = True        
        ntrials = 24 
    elif mode == 'test':
        which_device = 'cpu' 
        hro_random = False        
        ntrials = 36 
    
    reversal_start = int(ntrials/2) - 5 # int(ntrials/2) - 10 
    reversal_end = int(ntrials/2) + 5 # int(ntrials/2) + 10 

    # network size
    Ncells = 400 # 500
    mue = 2.0/np.sqrt(20)*np.sqrt(Ncells*0.1)
    scale_plastic = 0.2 # 0.5
    scale_extinp = 0.9 # 0.15 # 0.5
    scale_wout = 0.04 #scale_wout_list[ll] # 0.1

    # feedback parameters
    noise = 0.5
    fbnoise = 0.3

    # choice delay
    choice_delay = 5

    # target probability
    target_pr = 0.7 # 0.7

    # trial duration
    trial_duration_random_var = True
    trial_min = 5 # 4, 10 inclusive
    trial_max = 6 # 6, 11 exclusive
    trial_duration = 100 * np.random.randint(trial_min,trial_max,size=ntrials)
    SimTime = sum(trial_duration) # ntrials * trial_duration # 2400, 28*28


    parser = argparse.ArgumentParser(description='Simulate SNN')

    parser.add_argument('--run_saved', type=bool, default=True, help='run a saved model (default: false)')
    parser.add_argument('--run_replacedParam', type=bool, default=True, help='run a saved model with replaced parameters (default: false)')

    parser.add_argument('--dt', type=float, default=1.0, help='time step to discretize the simulation, in ms (default: 1)')
    parser.add_argument('--SimTime', type=float, default=SimTime, help='duration for each simulation, in ms (default: 500)')
    parser.add_argument('--stim_duration', type=float, default=50, help='duration of stimulus at the beginning of each trial, in ms (default: 50)')
    parser.add_argument('--feedback_duration', type=float, default=50, help='duration of feedback at the beginning of each trial, in ms (default: 50)')
    parser.add_argument('--intertrial_duration', type=float, default=100, help='duration of feedback at the beginning of each trial, in ms (default: 50)')
    parser.add_argument('--out_duration', type=float, default=50, help='duration of output response, in ms (default: 100)')
    parser.add_argument('--trial_min', type=float, default=trial_min, help='min duration of a trial, in ms (default: 400)')
    parser.add_argument('--trial_max', type=float, default=trial_max, help='max duration of a trial, in ms (default: 400)')
    parser.add_argument('--trial_duration', type=float, default=trial_duration, help='duration of a trial, in ms (default: 400)')
    parser.add_argument('--trial_duration_random', type=bool, default=trial_duration_random_var, help='trial duration is random (default: False)')
    parser.add_argument('--ntrials', type=float, default=ntrials, help='number of trials in a block (default: 12)')
    parser.add_argument('--reversal_start', type=float, default=reversal_start, help='first trial allowed to reverse (default: 4)')
    parser.add_argument('--reversal_end', type=float, default=reversal_end, help='last trial allowed to reverse (default: ntrials - 4)')
    parser.add_argument('--choice_delay', type=int, default=choice_delay, help='additional trials past the ideal reverse (default: 4)')
    parser.add_argument('--hro_random', type=bool, default=hro_random, help='randomize high reward option (default: False)')
    parser.add_argument('--trialType_random', type=str, default='fixed', help='randomize trialType (default: False)')

    parser.add_argument('--Ncells', type=float, default=Ncells, help='number of cells (default: 40)')
    parser.add_argument('--Ne', type=float, default=0, help='number of exc cells (default: 40)')
    parser.add_argument('--Ni', type=float, default=Ncells, help='number of inh cells (default: 40)')
    parser.add_argument('--thr', type=float, default=1.0, help='spike threshold (default: 1)')
    parser.add_argument('--Nout', type=float, default=1, help='number of outputs (default: 10)')
    parser.add_argument('--taus', type=float, default=20.0, help='synaptic time constant, in ms (default: 3)')
    parser.add_argument('--taum', type=float, default=20.0, help='membrane time constant, in ms (default: 10)')
    parser.add_argument('--tauout', type=float, default=20.0, help='output synaptic time constant, in ms (default: 30)')
    parser.add_argument('--noise', type=float, default=noise, help='std of gaussian noise (default: 0.5)')
    parser.add_argument('--fbnoise', type=float, default=fbnoise, help='std of gaussian noise to feedback (default: 0.5)')

    parser.add_argument('--device', type=str, default=which_device, help='device type: cpu or cuda')

    parser.add_argument('--wee', type=float, default=0.1, help='synaptic weights (default: 0.1)')
    parser.add_argument('--wei', type=float, default=-0.2, help='synaptic weights (default: -0.2)')
    parser.add_argument('--wie', type=float, default=0.1, help='synaptic weights (default: 0.1)')
    parser.add_argument('--wii', type=float, default=-0.2, help='synaptic weights (default: -0.2)')

    parser.add_argument('--fixation_penalty', type=float, default=0.5, help='penalty for fixation')
    parser.add_argument('--accuracy', type=float, default=0.59, help='stop training when reached (default: 0.8)')
    parser.add_argument('--target_pr', type=float, default=target_pr, help='probability of giving correct target (default: 0.8)')
    parser.add_argument('--batch_size', type=float, default=Nbatch, help='batch size (default: 256)')
    parser.add_argument('--savenum', type=float, default=25, help='number of saved batch size (default: 25)')
    parser.add_argument('--lr', type=float, default=1e-2, help='optimizer learning rate (default: 1e-2)')
    parser.add_argument('--beta', type=float, default=0, help='scaling factor of adaptation current (default: 1.8)')
    parser.add_argument('--mue', type=float, default=mue, help='external input to neurons (default: 1.0)')
    parser.add_argument('--scale_plastic', type=float, default=scale_plastic, help='scaling factor of plastic weights (default: 1.0)')
    parser.add_argument('--scale_wout', type=float, default=scale_wout, help='scaling factor of output weights (default: 1.0)')
    parser.add_argument('--scale_extinp', type=float, default=scale_extinp, help='scaling factor of output weights (default: 1.0)')
    parser.add_argument('--pr_plastic', type=float, default=0.1, help='connection probability (default: 0.2)')
    parser.add_argument('--pr_fixed', type=float, default=0.0, help='connection probability (default: 0.2)')
    parser.add_argument('--Nepoch', type=float, default=1, help='number of epochs (default: 4)')

    args = parser.parse_args(args=[])

    return args