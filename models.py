import torch
import torch.nn as nn
import torch.nn.functional as F
import tools_data

class RNN(nn.Module):

    def __init__(self, args):
        super(RNN, self).__init__()

        # fixed param
        self.register_buffer("dt", torch.tensor(args.dt))
        self.register_buffer("SimTime", torch.tensor(args.SimTime))
        self.register_buffer("stim_duration", torch.tensor(args.stim_duration))
        self.register_buffer("feedback_duration", torch.tensor(args.feedback_duration))
        self.register_buffer("intertrial_duration", torch.tensor(args.intertrial_duration))
        self.register_buffer("out_duration", torch.tensor(args.out_duration))
        self.register_buffer("trial_min", torch.tensor(args.trial_min))
        self.register_buffer("trial_max", torch.tensor(args.trial_max))
        self.register_buffer("trial_duration", torch.tensor(args.trial_duration))
        self.register_buffer("trial_duration_random", torch.tensor(args.trial_duration_random))
        self.register_buffer("ntrials", torch.tensor(args.ntrials))
        self.register_buffer("reversal_start", torch.tensor(args.reversal_start))
        self.register_buffer("reversal_end", torch.tensor(args.reversal_end))
        self.register_buffer("batch_size", torch.tensor(args.batch_size))
        self.register_buffer("savenum", torch.tensor(args.savenum))
        self.register_buffer("Nsteps", torch.tensor(int(args.SimTime / args.dt)))
        self.register_buffer("Ncells", torch.tensor(args.Ncells))
        self.register_buffer("Ne", torch.tensor(args.Ne))
        self.register_buffer("Ni", torch.tensor(args.Ni))
        self.register_buffer("pr_fixed", torch.tensor(args.pr_fixed))
        self.register_buffer("pr_plastic", torch.tensor(args.pr_plastic))
        self.register_buffer("scale_plastic", torch.tensor(args.scale_plastic))
        self.register_buffer("scale_wout", torch.tensor(args.scale_wout))
        self.register_buffer("mue", torch.tensor(args.mue))
        self.register_buffer("Nout", torch.tensor(args.Nout))
        self.register_buffer("taum", torch.tensor(args.taum))
        self.register_buffer("taus", torch.tensor(args.taus))
        self.register_buffer("dt_tausinv", torch.tensor(args.dt / args.taus))
        self.register_buffer("noise", torch.tensor(args.noise))
        self.register_buffer("fbnoise", torch.tensor(args.fbnoise))


        # fixed weights
        #   - doesn't play much role in this code
        wfixed = torch.zeros(args.Ncells, args.Ncells, dtype=torch.float32)
        self.register_buffer("weight_fixed", wfixed.clone())

        # --- plastic weights
        wrec = torch.zeros(args.Ncells, args.Ncells, dtype=torch.float32)
        if self.pr_plastic > 0.0:
            wrec_init = self.scale_plastic * self.taum * 1.0/torch.sqrt(torch.tensor(args.Ncells*args.pr_plastic, dtype=torch.float32)) 
            L = int(args.Ncells * args.pr_plastic)
            for ci in range(args.Ncells):
                preCells = torch.squeeze(torch.nonzero(wfixed[ci,:] == 0)).long()
                npreCells = preCells.shape[0]
                preCells_rnd = preCells[torch.randperm(npreCells)[:L]]
                wrec[ci, preCells_rnd] = wrec_init    
        wsign = torch.cat((torch.ones(args.Ne),-torch.ones(args.Ni)))     
        self.register_buffer("mask", (torch.abs(wrec) > 0).float())
        self.register_buffer("weight_sign", wsign)
        self.weight_rec = nn.Parameter(wrec, requires_grad=True)

        # output weights 
        wout = self.scale_wout * torch.randn(args.Ncells, args.Nout, dtype=torch.float32) 
        self.register_buffer("mask_out", torch.ones_like(wout))
        self.weight_out = nn.Parameter(wout, requires_grad=True)


    def weight_rec_dale(self):
        # weight_sign keeps inhibitory synapses negative.
        output = ( self.mask * F.relu(self.weight_rec) ) * torch.unsqueeze(self.weight_sign, 0)
        return output
    

    def initState(self, args):
        # initialize variables
        extIn = torch.zeros(self.batch_size, self.Ncells, requires_grad=False, device=args.device)
        u = torch.randn(self.batch_size, self.Ncells, requires_grad=False, device=args.device)
        out = torch.zeros(self.batch_size, self.Nout, requires_grad=False, device=args.device)
        outchoice = torch.zeros(self.batch_size, self.ntrials, self.Nout, device=args.device) # outchoice: Nbatch x Ntrials x Nout
        fixation  = torch.zeros(self.batch_size, self.ntrials, self.Nout, device=args.device) # outstd: Nbatch x Ntrials x Nout
        feedback_idx = torch.zeros(self.batch_size, self.ntrials, device=args.device)

        return extIn, u, out, outchoice, feedback_idx, fixation

    def saveState(self):
        # save variables
        Ntotal = self.Nsteps + self.feedback_duration
        extInsave = torch.zeros(self.savenum, Ntotal, self.Ncells)
        usave = torch.zeros(self.savenum, Ntotal, self.Ncells)
        outsim = torch.zeros(self.savenum, Ntotal, self.Nout)
        return extInsave, usave, outsim

    def activation(self, u):
        r = torch.sigmoid(3*u-2)
        return r

    def one_step(self, stimt, feedback, extNoise):    
        
        self.r     = self.activation(self.u)
        self.extIn = self.mue + feedback + stimt + extNoise
        self.u   = self.u * (1 - self.dt_tausinv)  + (torch.matmul(self.r, self.weight_rec_dale().t()) + self.extIn) * self.dt_tausinv
        self.out = torch.matmul(self.r, self.weight_out)

        return self.extIn, self.u, self.out

    def calcfeedback(self, trial_prev, trialType, TF, pred, A_wrong, A_correct, B_wrong, B_correct):
        # -------------------------------------------------------------
        # pred:   0, 1
        # target: 0, 1
        # TF:     0 (pred != target)
        #         1 (pred = target)
        # -------------------------------------------------------------
        # feedback0:    trialType = 0                | feedback_idx
        # feedback_A:   A_wrong:   TF = 0, pred = 0  |        0
        #               A_correct: TF = 1, pred = 0  |        1 
        # feedback_B:   B_wrong:   TF = 0, pred = 1  |        2
        #               B_correct: TF = 1, pred = 1  |        3
        # -------------------------------------------------------------
        # feedback1:    trialType = 1                | feedback_idx
        # feedback_A:   A_wrong:   TF = 0, pred = 1  |        0
        #               A_correct: TF = 1, pred = 1  |        1
        # feedback_B:   B_wrong:   TF = 0, pred = 0  |        2
        #               B_correct: TF = 1, pred = 0  |        3
        # -------------------------------------------------------------
        trialType_prev = trialType[:,trial_prev]
        feedback_A = torch.matmul((1-TF).unsqueeze(1), A_wrong.unsqueeze(0)) + torch.matmul(TF.unsqueeze(1), A_correct.unsqueeze(0)) 
        feedback_B = torch.matmul((1-TF).unsqueeze(1), B_wrong.unsqueeze(0)) + torch.matmul(TF.unsqueeze(1), B_correct.unsqueeze(0))
        feedback0 = (1-pred).unsqueeze(1)*feedback_A + pred.unsqueeze(1)*feedback_B # batch size x Ncells
        feedback1 = pred.unsqueeze(1)*feedback_A + (1-pred).unsqueeze(1)*feedback_B # batch size x Ncells                    
        feedback_input = (1-trialType_prev).unsqueeze(1)*feedback0 + trialType_prev.unsqueeze(1)*feedback1

        return feedback_input

    def calcfeedback_idx(self, trial_prev, trialType, TF, pred, onevec):
        trialType_prev = trialType[:,trial_prev]
        feedback_A_saved = (1-TF)*0.0*onevec + TF*1.0*onevec
        feedback_B_saved = (1-TF)*2.0*onevec + TF*3.0*onevec
        feedback0_saved = (1-pred)*feedback_A_saved + pred*feedback_B_saved
        feedback1_saved = pred*feedback_A_saved + (1-pred)*feedback_B_saved
        feedback_idx = (1-trialType_prev)*feedback0_saved + trialType_prev*feedback1_saved  

        return feedback_idx                      

    def forward(self, stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, args, saved):
        if self.trial_duration_random:
            # trial duration can be random, but here trial_min = 5 and trial_max = 6, so trial_duration = 500 and not random
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        # initial state
        self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials, self.fixation = self.initState(args)

        # save variables
        if saved:
            self.extInsave, self.usave, self.outsim = self.saveState()

        onevec = torch.ones(self.batch_size, device=args.device)
        bnum = torch.arange(self.savenum)
        trial_idx = 0
        out_idx = 0
        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
        Ntotal = self.Nsteps + self.feedback_duration
        for ti in range(Ntotal):

            ti_within = torch.tensor(ti) - torch.sum(self.trial_duration[0:trial_idx])

            # (1) feedback
            if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                # compute TF at the first time step
                if ti_within == 0:
                    trial_prev = trial_idx - 1
                    outsym_prevtrial = tools_data.symmetric_output_onetrial(self.outchoice[:,trial_prev,:]) # outchoice: Nbatch x Ntrials x Nout
                    pred = torch.argmax(outsym_prevtrial, axis=1) 
                    TF = pred.eq(target[:,trial_prev]).float() # TF: batch_size
                    feedback_input = self.calcfeedback(trial_prev, trialType, TF, pred, A_wrong, A_correct, B_wrong, B_correct).to(args.device)
                    self.feedback_idx_allTrials[:,trial_prev] = self.calcfeedback_idx(trial_prev, trialType, TF, pred, onevec)
                else:
                    feedback_noise = torch.normal(0, self.fbnoise, size=(self.batch_size, self.Ncells), device=args.device)
                    feedback = feedback_input + feedback_noise
            else:
                feedback = torch.zeros(self.batch_size, self.Ncells, device=args.device)

            # (2) cue 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = (torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))).to(args.device)
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells, device=args.device)
            
            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells), device=args.device)

            # run snn one step
            self.extIn, self.u, self.out = self.one_step(stimt, feedback, extNoise)
            
            # (3) choice output
            if trial_idx < self.ntrials:                
                feedback_end          = self.feedback_duration + self.intertrial_duration + self.stim_duration
                choice_duration_start = self.feedback_duration + self.intertrial_duration + self.stim_duration + 50
                choice_duration_end   = choice_duration_start + self.out_duration
                # fixation: out^2 = 0 from start to feedback_end
                if ti_within < feedback_end:
                    self.fixation[:,trial_idx,:] = self.fixation[:,trial_idx,:] + self.out**2/feedback_end # fixation: batch x ntrial
                    
                # outchoice: mean of out during choice_duration
                if (ti_within >= choice_duration_start) and (ti_within < choice_duration_end):
                    outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
                    out_idx += 1

                    if ti_within == choice_duration_end-1:
                        self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
                        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
                        out_idx = 0
                        trial_idx += 1
                    
            if saved:
                self.extInsave[:,ti,:], self.usave[:,ti,:], self.outsim[:,ti,:] = self.extIn[bnum,:], self.u[bnum,:], self.out[bnum,:]

        if not saved:
            return self.outchoice, self.feedback_idx_allTrials, self.fixation
        else:
            return self.outchoice, self.extInsave, self.usave, self.outsim,  self.feedback_idx_allTrials
