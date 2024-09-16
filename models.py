import torch
import torch.nn as nn
import torch.nn.functional as F
import tools_data

class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, thr, dampening_factor):
        ctx.save_for_backward(v, thr, dampening_factor)
        z = torch.gt(v - thr, 0.0).float()
        return z

    @staticmethod
    def backward(ctx, grad_output):
        v, thr, dampening_factor = ctx.saved_tensors
        grad_v = None
        vscaled = (v - thr) / thr
        # torch.max has an issue: x and y in torch.max(x,y) must be on the same device.
        dz_dvscaled = torch.max(torch.tensor(0.0, device=vscaled.device), 1.0 - torch.abs(vscaled))
        dz_dvscaled *= dampening_factor
        grad_v = grad_output * dz_dvscaled
        # return gradients wrt v (grad_v), thr (None), and dampening_factor (None)
        return grad_v, None, None

class SparseMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, weight, mask):
        ctx.save_for_backward(z, weight, mask)
        output = z.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z, weight, mask = ctx.saved_tensors
        grad_z, grad_w, grad_mask = None, None, None
        grad_z = grad_output.mm(weight)
        grad_w = grad_output.t().mm(z) * mask
        return grad_z, grad_w, grad_mask


class ALIF(nn.Module):

    def __init__(self, args):
        super(ALIF, self).__init__()

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
        self.register_buffer("phase_wgt", torch.tensor(args.phase_wgt))
        self.register_buffer("batch_size", torch.tensor(args.batch_size))
        self.register_buffer("savenum", torch.tensor(args.savenum))
        self.register_buffer("Nsteps", torch.tensor(int(args.SimTime / args.dt)))
        self.register_buffer("Ncells", torch.tensor(args.Ncells))
        self.register_buffer("Ne", torch.tensor(args.Ne))
        self.register_buffer("Ni", torch.tensor(args.Ni))
        self.register_buffer("beta", torch.tensor(args.beta))
        self.register_buffer("thr", torch.tensor(args.thr))
        self.register_buffer("reset", torch.tensor(args.reset))
        self.register_buffer("maxRate", torch.tensor(args.maxRate))
        self.register_buffer("pr_fixed", torch.tensor(args.pr_fixed))
        self.register_buffer("pr_plastic", torch.tensor(args.pr_plastic))
        self.register_buffer("scale_plastic", torch.tensor(args.scale_plastic))
        self.register_buffer("scale_wout", torch.tensor(args.scale_wout))
        self.register_buffer("scale_extinp", torch.tensor(args.scale_extinp))
        self.register_buffer("mue", torch.tensor(args.mue))
        self.register_buffer("dampening_factor", torch.tensor(args.dampening_factor))
        self.register_buffer("Nout", torch.tensor(args.Nout))
        self.register_buffer("weight_in", args.win*torch.randn(1, args.Ncells)) #shape: 1 x Ncells
        self.register_buffer("wstim", torch.tensor(args.wstim))        
        self.register_buffer("wstim", torch.tensor(args.wstim))  
        self.register_buffer("taum", torch.tensor(args.taum))
        self.register_buffer("taus", torch.tensor(args.taus))
        self.register_buffer("taua", torch.tensor(args.taua))
        self.register_buffer("tauout", torch.tensor(args.tauout))
        self.register_buffer("dt_tauoutinv", torch.tensor(args.dt / args.tauout))      
        self.register_buffer("dt_tausinv", torch.tensor(args.dt / args.taus))
        self.register_buffer("dt_tauainv", torch.tensor(args.dt / args.taua))
        self.register_buffer("noise", torch.tensor(args.noise))
        self.register_buffer("fbnoise", torch.tensor(args.fbnoise))

        taus_list = torch.zeros(1,args.Ncells, dtype=torch.float32)
        # taus_list[0,:int(args.Ncells/2)] = args.taus
        # taus_list[0,int(args.Ncells/2):] = args.taus_long
        taus_list[0,:] = args.taus_long
        self.register_buffer("taus_list", taus_list.clone())
        self.register_buffer("dt_tausinv_list", args.dt / taus_list.clone())
        # self.tausinv = nn.Parameter(1/taus_list, requires_grad=True)

        # excitatory-inhibitory
        # --- fixed weights
        wfixed = torch.zeros(args.Ncells, args.Ncells, dtype=torch.float32)
        wfixed_init = torch.tensor(0)
        wfixed[:args.Ne, :args.Ne] = wfixed_init*(torch.rand(args.Ne, args.Ne) < args.pr_fixed)
        wfixed[:args.Ne, args.Ne:] = -wfixed_init*(torch.rand(args.Ne, args.Ni) < args.pr_fixed)
        wfixed[args.Ne:, :args.Ne] = wfixed_init*(torch.rand(args.Ni, args.Ne) < args.pr_fixed)
        wfixed[args.Ne:, args.Ne:] = -wfixed_init*(torch.rand(args.Ni, args.Ni) < args.pr_fixed)
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
        winit = wrec * torch.unsqueeze(wsign, 0)     
        self.register_buffer("mask", (torch.abs(wrec) > 0).float())
        self.register_buffer("weight_sign", wsign)
        self.register_buffer("weight_init", winit)
        self.weight_rec = nn.Parameter(wrec, requires_grad=True)

        # # synaptic time constants
        # self.taus_inv = nn.Parameter(1 / args.taus * torch.ones(args.Ncells), requires_grad=True)

        #----- output weights -----#
        # (1) weight_out mixed sign
        wout = self.scale_wout * torch.randn(args.Ncells, args.Nout, dtype=torch.float32) ####### MIXED weight_out SIGNS & No neuron groups
        self.register_buffer("mask_out", torch.ones_like(wout))
        self.weight_out = nn.Parameter(wout, requires_grad=True)

        # # (2) weight_out > 0
        # wout = torch.zeros(args.Ncells, args.Nout, dtype=torch.float32)
        # group1 = torch.arange(0, args.Ncells/2).long()
        # group2 = torch.arange(args.Ncells/2, args.Ncells).long()
        # self.group1 = group1
        # self.group2 = group2
        # wout[group1,0] = torch.rand(group1.size()[0]) ####### POSITIVE weight_out 
        # wout[group2,1] = torch.rand(group2.size()[0]) ####### POSITIVE weight_out 
        # self.register_buffer("mask_out", (wout > 0).float())
        # self.weight_out = nn.Parameter(wout, requires_grad=True)

    def weight_out_positive(self):
        output = self.mask_out * F.relu(self.weight_out)
        return output

    def weight_out_mixed(self):
        output = self.weight_out
        return output


    def weight_dale(self):
        # weight_dale = mask * relu(weight_plastic) x diag[1,..,1,-1,..,-1]
        # mask: enforce sparse connectivity
        # relu: enforce dale's principle
        # diag: flip the column signs
        # *   : hadamard product
        # x   : matrix product

        output = ( self.mask * F.relu(self.weight_rec) ) * torch.unsqueeze(self.weight_sign, 0)

        # output_tmp = ( self.mask * F.relu(self.weight_rec) ) * torch.unsqueeze(self.weight_sign, 0)
        # output = torch.max(torch.tensor(-1.2), output_tmp)

        return output
    
    def tausinv_pos(self):        
        output = F.relu(self.tausinv)
        return output

    def initState(self, args):
        # initialize variables
        extIn = torch.zeros(self.batch_size, self.Ncells, requires_grad=False, device=args.device)
        u = torch.randn(self.batch_size, self.Ncells, requires_grad=False, device=args.device)
        # a = torch.zeros(self.batch_size, self.Ncells, requires_grad=False)
        out = torch.zeros(self.batch_size, self.Nout, requires_grad=False, device=args.device)
        outchoice = torch.zeros(self.batch_size, self.ntrials, self.Nout, device=args.device) # outchoice: Nbatch x Ntrials x Nout
        # outstd    = torch.zeros(self.batch_size, self.ntrials, self.Nout) # outstd: Nbatch x Ntrials x Nout
        feedback_idx = torch.zeros(self.batch_size, self.ntrials, device=args.device)

        return extIn, u, out, outchoice, feedback_idx

    def saveState(self):
        # save variables
        Ntotal = self.Nsteps + self.feedback_duration
        extInsave = torch.zeros(self.savenum, Ntotal, self.Ncells)
        usave = torch.zeros(self.savenum, Ntotal, self.Ncells)
        outsim = torch.zeros(self.savenum, Ntotal, self.Nout)

        return extInsave, usave, outsim

    def saveStatePerturbed(self):
        # save variables
        extInsave = torch.zeros(self.batch_size, self.ntrials, self.Ncells)
        usave     = torch.zeros(self.batch_size, self.ntrials, self.Ncells)
        outsim    = torch.zeros(self.batch_size, self.ntrials, self.Nout)

        # # save variables
        # Ntotal = self.Nsteps + self.feedback_duration
        # extInsave = torch.zeros(10, Ntotal, self.Ncells)
        # usave     = torch.zeros(10, Ntotal, self.Ncells)
        # outsim    = torch.zeros(10, Ntotal, self.Nout)

        return extInsave, usave, outsim

    def saveStatePerturbed_longtime(self):
        # # save variables
        # extInsave = torch.zeros(self.batch_size, self.ntrials, self.Ncells)
        # usave     = torch.zeros(self.batch_size, self.ntrials, self.Ncells)
        # outsim    = torch.zeros(self.batch_size, self.ntrials, self.Nout)

        # save variables
        Ntotal = self.Nsteps + self.feedback_duration
        extInsave = torch.zeros(10, Ntotal, self.Ncells)
        usave     = torch.zeros(10, Ntotal, self.Ncells)
        outsim    = torch.zeros(self.batch_size, Ntotal, self.Nout)
        xreverse  = torch.zeros(self.batch_size, Ntotal)

        return extInsave, usave, outsim, xreverse

    def homotopyInitState(self, nhomotopy, uinit):
        # initialize variables
        # extIn = torch.zeros(nhomotopy, self.Ncells, requires_grad=False)
        u = uinit
        # a = torch.zeros(nhomotopy, self.Ncells, requires_grad=False)
        # out = torch.zeros(nhomotopy, self.Nout, requires_grad=False)

        return u
    
    def homotopySaveState(self, nhomotopy, Ntotal):
        # save variables
        usave = torch.zeros(nhomotopy, Ntotal, self.Ncells)

        return usave


    def activation(self, u):
        # torch.sigmoid(x) = 1/(1+exp(-x))
        r = torch.sigmoid(3*u-2)
        return r

    def activation_inverse(self, r):

        # rate -> input
        rinv = 1/3*(2 - torch.log(1/r - 1))
        return rinv

    def derivative_activation(self, u):
        
        y = torch.exp(-(3*u-2))
        dr = 3*y / torch.pow(1 + y,2)
        return dr

    def second_derivative_activation(self, u):
        
        y = torch.exp(-(3*u-2))
        # dr = 3*y / torch.pow(1 + y,2)
        ddr = 9*y*(torch.pow(y,2)-1) / torch.pow(1+y,4)
        return ddr

    def dudt_ratemodel(self, u, extIn, weight):
        rate = self.activation(u)
        dudt = 1/self.taus * (-u + torch.matmul(rate, weight.t()) + extIn)

        return dudt

    def jacobian_onestep(self, u0, wrec):
        # Calculate the jacobian of one_step update map. The formula was calculated by hand
        # DF(u) = (1 - dt/taus)*Id + dt/taus*W*diag[f'(u)]
        #   where Id is identity matrix, f' is the derivative of activation
        diag_df = torch.diag(self.derivative_activation(u0))
        jac = (1-self.dt_tausinv)*torch.eye(self.Ncells) + self.dt_tausinv * torch.matmul(wrec, diag_df)
        return jac

    def modified_gram_schmidt(self, V, nvec, alpha):
        U        = torch.zeros_like(V)
        alpha[0] = torch.linalg.vector_norm(V[:,0])
        U[:,0]   = V[:,0] / alpha[0]      
        for ii in range(1,nvec):
            U[:,ii] = V[:,ii]
            for jj in range(ii):
                U[:,ii] = U[:,ii] - torch.dot(U[:,ii], U[:,jj])*U[:,jj] 
            alpha[ii] = torch.linalg.vector_norm(U[:,ii])
            U[:,ii]   = U[:,ii] / alpha[ii]
            
        return U, alpha
        

    def one_step(self, stimt, feedback, extNoise):    
        
        self.r     = self.activation(self.u)
        self.extIn = self.mue + feedback + stimt + extNoise

        self.u   = self.u * (1 - self.dt_tausinv_list)  + (torch.matmul(self.r, self.weight_dale().t()) + self.extIn) * self.dt_tausinv_list
        self.out = torch.matmul(self.r, self.weight_out_mixed())

        return self.extIn, self.u, self.out

    def one_step_learnTaus(self, stimt, feedback, extNoise):    
        
        self.r     = self.activation(self.u)
        self.extIn = self.mue + feedback + stimt + extNoise

        self.u   = self.u * (1 - self.dt * self.tausinv_pos())  + (torch.matmul(self.r, self.weight_dale().t()) + self.extIn) * self.dt * self.tausinv_pos()
        self.out = torch.matmul(self.r, self.weight_out_mixed())

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
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        # initial state
        self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)

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

            # feedback
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

            # target presentation 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = (torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))).to(args.device)
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells, device=args.device)
            
            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells), device=args.device)

            ###################################
            ## static synaptic time constant ##
            ###################################
            # run snn one step
            self.extIn, self.u, self.out = self.one_step(stimt, feedback, extNoise)
            
            ##################################
            ## learn synaptic time constant ##
            ##################################
            # # run snn one step
            # self.extIn, self.u, self.out = self.one_step_learnTaus(stimt, feedback, extNoise)

            # #####################################
            # ## choice made at the end of trial ##
            # #####################################
            # # compute output
            # if trial_idx < self.ntrials:
            #     if ti_within >= self.trial_duration[trial_idx] - self.out_duration:
            #         outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
            #         out_idx += 1

            #         if ti_within == self.trial_duration[trial_idx]-1:
            #             self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
            #             # reset temporary variables
            #             outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
            #             out_idx = 0
            #             trial_idx += 1

            ###########################################
            ## choice made at the beginning of trial ##
            ###########################################
            # compute output
            if trial_idx < self.ntrials:                
                choice_duration_start = self.feedback_duration + self.intertrial_duration + self.stim_duration + 50
                choice_duration_end   = choice_duration_start + self.out_duration
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
            return self.outchoice, self.feedback_idx_allTrials
        else:
            return self.outchoice, self.extInsave, self.usave, self.outsim,  self.feedback_idx_allTrials

    def homotopy(self, xintp, uinit, nhomotopy, Nstep):

        # homotopy initial state
        self.u            = uinit
        # save variables
        self.usave        = self.homotopySaveState(nhomotopy, Nstep)
        self.usave[:,0,:] = self.u

        for ti in range(Nstep-1):
            
            self.r = self.activation(self.u)
            extIn  = xintp[:,ti,:]
            self.u = self.u * (1 - self.dt_tausinv_list)  + (torch.matmul(self.r, self.weight_dale().t()) + extIn) * self.dt_tausinv_list

            self.usave[:,ti+1,:] = self.u

        return self.usave


    def integrate(self, xintp, uinit, nhomotopy, Nstep, vreverse, vchoice):

        # homotopy initial state
        self.u            = uinit
        # save projection activity
        vfield   = torch.zeros(nhomotopy, Nstep)
        prj_r    = torch.zeros(nhomotopy, Nstep)
        prj_leak = torch.zeros(nhomotopy, Nstep)
        prj_rec  = torch.zeros(nhomotopy, Nstep)
        prj_ext  = torch.zeros(nhomotopy, Nstep)
        self.usave = self.homotopySaveState(nhomotopy, Nstep)
        # save cosine angle
        ang_rec  = torch.zeros(nhomotopy, Nstep)
        ang_ext  = torch.zeros(nhomotopy, Nstep)
        # save initial variables
        self.usave[:,0,:] = self.u
        prj_r[:,0]        = torch.matmul(self.u, vreverse)

        for ti in range(Nstep):
            
            if ti < Nstep-1:
                self.r = self.activation(self.u)
                recIn  = torch.matmul(self.r, self.weight_dale().t())
                ############################################
                # Use the next time step for extIn         #
                # then u and extIn are the same time step  #
                #------------------------------------------#
                # When sim data were saved, 
                #  - u was updated to next time step
                #  - extIn was not updated
                ############################################
                extIn  = xintp[:,ti+1,:]
                self.u = self.u * (1 - self.dt_tausinv_list)  + (recIn + extIn) * self.dt_tausinv_list
                
                diag_df       = self.derivative_activation(self.u)
                vfield[:,ti]  = torch.norm((-self.u + recIn + extIn) * diag_df, dim=1)
                prj_leak[:,ti]= torch.matmul(self.u * diag_df, vreverse)
                prj_rec[:,ti] = torch.matmul(recIn  * diag_df, vreverse)
                prj_ext[:,ti] = torch.matmul(extIn  * diag_df, vreverse)
                prj_r[:,ti+1] = torch.matmul(self.activation(self.u), vreverse)
                self.usave[:,ti+1,:] = self.u
                
                # angle
                ang_rec[:,ti] = prj_rec[:,ti] / torch.norm(recIn,dim=1) / torch.norm(vreverse) 
                ang_ext[:,ti] = prj_ext[:,ti] / torch.norm(extIn,dim=1) / torch.norm(vreverse) 

            if ti == Nstep-1:
                self.r = self.activation(self.u)
                recIn  = torch.matmul(self.r, self.weight_dale().t())
                prj_rec[:,ti] = torch.matmul(recIn * diag_df, vreverse)

        return self.usave, vfield, ang_rec, ang_ext, prj_r, prj_leak, prj_rec, prj_ext


    # def integrate(self, xintp, uinit, nhomotopy, Nstep, vreverse, vchoice):

    #     # homotopy initial state
    #     self.u            = uinit
    #     # save projection activity
    #     vfield  = torch.zeros(nhomotopy, Nstep)
    #     prj_u   = torch.zeros(nhomotopy, Nstep)
    #     prj_rec = torch.zeros(nhomotopy, Nstep)
    #     prj_ext = torch.zeros(nhomotopy, Nstep)
    #     self.usave = self.homotopySaveState(nhomotopy, Nstep)
    #     # save cosine angle
    #     ang_rec  = torch.zeros(nhomotopy, Nstep)
    #     ang_ext  = torch.zeros(nhomotopy, Nstep)
    #     # save initial variables
    #     self.usave[:,0,:] = self.u
    #     prj_u[:,0]        = torch.matmul(self.u, vreverse)

    #     for ti in range(Nstep):
            
    #         if ti < Nstep-1:
    #             self.r = self.activation(self.u)
    #             recIn  = torch.matmul(self.r, self.weight_dale().t())
    #             ############################################
    #             # Use the next time step for extIn         #
    #             # then u and extIn are the same time step  #
    #             #------------------------------------------#
    #             # When sim data were saved, 
    #             #  - u was updated to next time step
    #             #  - extIn was not updated
    #             ############################################
    #             extIn  = xintp[:,ti+1,:]
    #             self.u = self.u * (1 - self.dt_tausinv_list)  + (recIn + extIn) * self.dt_tausinv_list
                
    #             diag_df       = torch.diag(self.derivative_activation(self.u))
    #             vfield[:,ti]  = torch.norm(-self.u + recIn + extIn, dim=1)
    #             prj_rec[:,ti] = torch.matmul(recIn,  vreverse)
    #             prj_ext[:,ti] = torch.matmul(extIn,  vreverse)
    #             prj_u[:,ti+1] = torch.matmul(self.u, vreverse)
    #             self.usave[:,ti+1,:] = self.u
                
    #             # angle
    #             ang_rec[:,ti] = prj_rec[:,ti] / torch.norm(recIn,dim=1) / torch.norm(vreverse) 
    #             ang_ext[:,ti] = prj_ext[:,ti] / torch.norm(extIn,dim=1) / torch.norm(vreverse) 

    #         if ti == Nstep-1:
    #             self.r = self.activation(self.u)
    #             recIn  = torch.matmul(self.r, self.weight_dale().t())
    #             prj_rec[:,ti] = torch.matmul(recIn, vreverse)

    #     return self.usave, vfield, ang_rec, ang_ext, prj_u, prj_rec, prj_ext



    def perturb(self, uinit, npert, vpert, trevL, trevR, Ntotal):
        if self.trial_duration_random:
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        # homotopy initial state
        self.extIn, self.u, self.a, self.out = self.homotopyInitState(npert, uinit)

        # save variables
        self.usave = self.homotopySaveState(npert, Ntotal)

        # save initial state
        self.usave[:,0,:] = self.u

        for ti in range(1,Ntotal):
            # perturbation
            if (ti > trevL) and (ti < trevR):
                perturbation = vpert
            else:
                perturbation = 0

            # run snn one step
            self.extIn, self.u, self.a, self.out = self.one_step(perturbation, 0, 0)

            self.usave[:,ti,:] = self.u

        return self.usave


    def perturb_propagate(self, stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, saved, run_reference, run_perturbation, uinit_reference, feedback_reference, perturb_trial, vpert, pert_start, pert_end, args):
        if self.trial_duration_random:
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        ##################
        ## PERTURBATION ##
        ##################
        # initial state
        if run_reference:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            uinit_reference = self.u
        elif run_perturbation:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            self.u = uinit_reference

        ##################
        ## PERTURBATION ##
        ##################
        # convert feedback_idx to feedback_input
        convertIdx2Input = torch.stack((A_wrong, A_correct, B_wrong, B_correct))

        # save variables
        if saved:
            self.extInsave, self.usave, self.outsim = self.saveStatePerturbed()

        onevec = torch.ones(self.batch_size, device=args.device)
        bnum = torch.arange(self.savenum)
        trial_idx = 0
        out_idx = 0
        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
        Ntotal = self.Nsteps + self.feedback_duration
        for ti in range(Ntotal):

            ti_within = torch.tensor(ti) - torch.sum(self.trial_duration[0:trial_idx])

            # feedback
            # if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.stim_duration:
            if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                # compute TF at the first time step
                if ti_within == 0:
                    trial_prev = trial_idx - 1
                    outsym_prevtrial = tools_data.symmetric_output_onetrial(self.outchoice[:,trial_prev,:]) # outchoice: Nbatch x Ntrials x Nout
                    pred_prev = torch.argmax(outsym_prevtrial, axis=1) 
                    TF = pred_prev.eq(target[:,trial_prev]).float() # TF: batch_size
                    ##################
                    ## PERTURBATION ##
                    ##################
                    if run_reference:
                        feedback_input                            = self.calcfeedback(trial_prev, trialType, TF, pred_prev, A_wrong, A_correct, B_wrong, B_correct)
                        self.feedback_idx_allTrials[:,trial_prev] = self.calcfeedback_idx(trial_prev, trialType, TF, pred_prev, onevec)
                    elif run_perturbation:
                        #---------------------------------------#
                        # something wrong with pertfeedback_idx #
                        #---------------------------------------#
                        feedback_idx                              = self.pertfeedback_idx(feedback_reference, perturb_trial, trial_prev, pred_prev, TF, trialType)
                        feedback_input                            = self.feedback_convertIdx2Input(feedback_idx, convertIdx2Input)
                        self.feedback_idx_allTrials[:,trial_prev] = feedback_idx
                else:
                    feedback_noise = torch.normal(0, self.fbnoise, size=(self.batch_size, self.Ncells))
                    feedback = feedback_input + feedback_noise
            else:
                feedback = torch.zeros(self.batch_size, self.Ncells)

            # target presentation 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells)

            ##################
            ## PERTURBATION ##
            ##################
            # add perturbation input
            if run_reference:
                perturbation_input = 0
            elif run_perturbation:
                # set up perturbation input for the trial
                if ti_within == pert_start:
                    pertidx = (trial_idx == perturb_trial).to(torch.float32)
                    perturbation_input_tidx = torch.matmul(pertidx.view(self.batch_size,-1), vpert.view(-1,self.Ncells))
                # apply perturbation input
                if (ti_within > pert_start) and (ti_within < pert_end):
                    perturbation_input = perturbation_input_tidx
                else: 
                    perturbation_input = 0

            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells))
            
            # run snn one step
            total_input = feedback + perturbation_input
            self.extIn, self.u, self.out = self.one_step(stimt, total_input, extNoise)

            #####################################
            ## choice made at the end of trial ##
            #####################################
            # # compute output
            # if trial_idx < self.ntrials:
            #     if ti_within >= self.trial_duration[trial_idx] - self.out_duration:
            #         outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
            #         out_idx += 1

            #         if ti_within == self.trial_duration[trial_idx]-1:
            #             self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
            #             # reset temporary variables
            #             outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
            #             out_idx = 0
            #             trial_idx += 1

            ###########################################
            ## choice made at the beginning of trial ##
            ###########################################
            # compute output
            if trial_idx < self.ntrials:                
                choice_duration_start = self.feedback_duration + self.intertrial_duration + self.stim_duration + 50
                choice_duration_end   = choice_duration_start + self.out_duration
                if (ti_within >= choice_duration_start) and (ti_within < choice_duration_end):
                    outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
                    out_idx += 1

                    if ti_within == choice_duration_end-1:
                        self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
                        # reset temporary variables
                        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
                        out_idx = 0
                        trial_idx += 1
                        

            if saved:
                if ti_within == self.feedback_duration + self.intertrial_duration + self.stim_duration:
                    self.extInsave[:,trial_idx,:], self.usave[:,trial_idx,:], self.outsim[:,trial_idx,:] = self.extIn, self.u, self.out
      
                # self.extInsave[:,ti,:], self.usave[:,ti,:], self.outsim[:,ti,:] = self.extIn[:10,:], self.u[:10,:], self.out[:10,:]

        return self.outchoice, self.extInsave, self.usave, self.outsim, self.feedback_idx_allTrials, uinit_reference
        
    def pertfeedback_idx(self, feedback_reference, perturb_trial, trial_prev, pred_prev, TF, trialType):

        '''
        perturb_trial: nbatch
        '''
        trial_idx = trial_prev + 1        
        # feedback used in the reference simulation
        ###########################################################
        ## MAKE SURE TO CLONE IT. WILL MODIFY feedback_idx BELOW ##
        ###########################################################
        feedback_idx = feedback_reference[:,trial_prev].clone()

        # prior to perturbation: 
        #       no change to feedback 
        #       *trial* | perturbation | reversal
        # idx1 = trial_prev  <  perturb_trial # no change

        # after perturbation & before reversal:
        #       feedback is A0 
        #       perturbation | *trial* | reversal
        bidx2 = (trial_idx > perturb_trial) * (pred_prev == 0) # choose A
        feedback_idx[bidx2] = 0 # feedback A0

        # after reversal:
        #      feedback is B0/B1 stochastically
        #      perturbation | reversal | *trial*
        onevec = torch.ones(self.batch_size)
        bidx3 = (trial_idx > perturb_trial) * (pred_prev == 1) # choose B
        feedback_idx_postrev = self.calcfeedback_idx(trial_prev, trialType, TF, pred_prev, onevec) # stochastic B0/B1
        feedback_idx[bidx3] = feedback_idx_postrev[bidx3]

        return feedback_idx

    def feedback_convertIdx2Input(self, feedback_idx, convertIdx2Input):
        '''
        feedback_idx    : nbatch         (elements are 0,1,2,3 corresponding to A0,A1,B0,B1)
        feedback_onehot : nbatch x 4     (onehot vector of feedback_idx. needed to convert index to input)
        convertIdx2Input: 4 x ncell      (stacked feedback inputs A0, A1, B0, B1)
        feedback_input  : nbatch x ncell (actual feedback input to the network)
        '''
        feedback_onehot = F.one_hot(feedback_idx.long(), num_classes=4).to(torch.float32)
        feedback_input  = torch.matmul(feedback_onehot, convertIdx2Input)

        return feedback_input



    def perturb_propagate_clip_eigenvector(self, stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, saved, run_reference, run_perturbation, uinit_reference, feedback_reference, perturb_trial, vpert, pert_start, pert_end, clip_eigenvec):
        if self.trial_duration_random:
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        ##################
        ## PERTURBATION ##
        ##################
        # initial state
        if run_reference:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(saved)
            uinit_reference = self.u
        elif run_perturbation:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(saved)
            self.u = uinit_reference

        ##################
        ## PERTURBATION ##
        ##################
        # convert feedback_idx to feedback_input
        convertIdx2Input = torch.stack((A_wrong, A_correct, B_wrong, B_correct))

        # save variables
        if saved:
            self.extInsave, self.usave, self.outsim = self.saveState()

        bnum = torch.arange(self.savenum)
        trial_idx = 0
        out_idx = 0
        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
        Ntotal = self.Nsteps + self.feedback_duration
        for ti in range(Ntotal):

            ti_within = torch.tensor(ti) - torch.sum(self.trial_duration[0:trial_idx])

            # feedback
            # if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.stim_duration:
            if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                # compute TF at the first time step
                if ti_within == 0:
                    trial_prev = trial_idx - 1
                    outsym_prevtrial = tools_data.symmetric_output_onetrial(self.outchoice[:,trial_prev,:]) # outchoice: Nbatch x Ntrials x Nout
                    pred_prev = torch.argmax(outsym_prevtrial, axis=1) 
                    TF = pred_prev.eq(target[:,trial_prev]).float() # TF: batch_size
                    ##################
                    ## PERTURBATION ##
                    ##################
                    if run_reference:
                        feedback_input                            = self.calcfeedback(trial_prev, trialType, TF, pred_prev, A_wrong, A_correct, B_wrong, B_correct)
                        self.feedback_idx_allTrials[:,trial_prev] = self.calcfeedback_idx(trial_prev, trialType, TF, pred_prev)
                    elif run_perturbation:
                        #---------------------------------------#
                        # something wrong with pertfeedback_idx #
                        #---------------------------------------#
                        feedback_idx                              = self.pertfeedback_idx(feedback_reference, perturb_trial, trial_prev, pred_prev, TF, trialType)
                        feedback_input                            = self.feedback_convertIdx2Input(feedback_idx, convertIdx2Input)
                        self.feedback_idx_allTrials[:,trial_prev] = feedback_idx
                else:
                    feedback_noise = torch.normal(0, self.fbnoise, size=(self.batch_size, self.Ncells))
                    feedback = feedback_input + feedback_noise
            else:
                feedback = torch.zeros(self.batch_size, self.Ncells)

            # target presentation 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells)

            ##################
            ## PERTURBATION ##
            ##################
            # add perturbation input
            if run_reference:
                perturbation_input = 0
            elif run_perturbation:
                # set up perturbation input for the trial
                if ti_within == pert_start:
                    pertidx = (trial_idx == perturb_trial).to(torch.float32)
                    perturbation_input_tidx = torch.matmul(pertidx.view(self.batch_size,-1), vpert.view(-1,self.Ncells))
                # apply perturbation input
                if (ti_within > pert_start) and (ti_within < pert_end):
                    perturbation_input = perturbation_input_tidx
                else: 
                    perturbation_input = 0

            ##############################
            ## CLIP MAX EIGEN DIRECTION ##
            ##############################
            if clip_eigenvec:
                # if (ti_within > pert_start+25) and (ti_within < pert_end):
                if (ti_within > pert_start) and (ti_within < pert_end):
                    for bidx in range(self.batch_size):
                        # print(bidx)
                        u_slice = self.u[bidx,:]
                        jacF    = self.jacobian_onestep(u_slice, self.weight_dale())
                        eigval, eigvec = torch.linalg.eig(jacF)

                        #-- clip along max eigen direction --#
                        max_idx = torch.argmax(torch.abs(eigval))
                        max_vec = eigvec[:,max_idx]
                        imag_part = torch.sum(torch.abs(torch.imag(max_vec)))
                        if imag_part == 0:
                            max_vec_ = torch.real(max_vec)
                            max_coef = torch.inner(u_slice, max_vec_)
                            u_clip = u_slice - max_coef * max_vec_
                        elif imag_part > 0:
                            max_vec_real = torch.real(max_vec) / torch.sqrt(torch.sum(torch.real(max_vec)**2))
                            max_vec_imag = torch.imag(max_vec) / torch.sqrt(torch.sum(torch.imag(max_vec)**2))

                            max_coef_real = torch.inner(u_slice, max_vec_real)
                            u_clip = u_slice - max_coef_real * max_vec_real

                            max_coef_imag = torch.inner(u_clip, max_vec_imag)
                            u_clip = u_clip - max_coef_imag * max_vec_imag
                        self.u[bidx,:] = u_clip


            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells))
            
            # run snn one step
            total_input = feedback + perturbation_input
            self.extIn, self.u, self.out = self.one_step(stimt, total_input, extNoise)

            # compute output
            if trial_idx < self.ntrials:
                if ti_within >= self.trial_duration[trial_idx] - self.out_duration:
                    outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
                    out_idx += 1

                    if ti_within == self.trial_duration[trial_idx]-1:
                        self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
                        # reset temporary variables
                        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
                        out_idx = 0
                        trial_idx += 1

            if saved:
                self.extInsave[:,ti,:], self.usave[:,ti,:], self.outsim[:,ti,:] = self.extIn[bnum,:], self.u[bnum,:], self.out[bnum,:]

        return self.outchoice, self.extInsave, self.usave, self.outsim, self.feedback_idx_allTrials, uinit_reference




    def perturb_propagate_longtime(self, stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, saved, run_reference, run_perturbation, uinit_reference, feedback_reference, perturb_trial, vpert, pert_start, pert_end, args, vreverse_unitlen):
        if self.trial_duration_random:
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        ##################
        ## PERTURBATION ##
        ##################
        # initial state
        if run_reference:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            uinit_reference = self.u
        elif run_perturbation:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            self.u = uinit_reference

        initRate = torch.zeros(self.batch_size, self.Ncells, self.ntrials, device=args.device) # outchoice: Nbatch x Ntrials x Nout

        ##################
        ## PERTURBATION ##
        ##################
        # convert feedback_idx to feedback_input
        convertIdx2Input = torch.stack((A_wrong, A_correct, B_wrong, B_correct))

        # save variables
        if saved:
            self.extInsave, self.usave, self.outsim, self.xreverse = self.saveStatePerturbed_longtime()

        onevec    = torch.ones(self.batch_size, device=args.device)
        bnum      = torch.arange(self.savenum)
        outtrial  = torch.zeros(self.batch_size, self.out_duration, self.Nout)
        trial_idx = 0
        out_idx   = 0
        stim_on               = self.feedback_duration + self.intertrial_duration
        stim_off              = self.feedback_duration + self.intertrial_duration + self.stim_duration
        choice_duration_start = self.feedback_duration + self.intertrial_duration + self.stim_duration + 50
        choice_duration_end   = choice_duration_start + self.out_duration        
        Ntotal                = self.Nsteps + self.feedback_duration
        for ti in range(Ntotal):

            ti_within = torch.tensor(ti) - torch.sum(self.trial_duration[0:trial_idx])

            # feedback
            # if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.stim_duration:
            if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                # compute TF at the first time step
                if ti_within == 0:
                    trial_prev = trial_idx - 1
                    outsym_prevtrial = tools_data.symmetric_output_onetrial(self.outchoice[:,trial_prev,:]) # outchoice: Nbatch x Ntrials x Nout
                    pred_prev = torch.argmax(outsym_prevtrial, axis=1) 
                    TF = pred_prev.eq(target[:,trial_prev]).float() # TF: batch_size
                    ##################
                    ## PERTURBATION ##
                    ##################
                    if run_reference:
                        feedback_input                            = self.calcfeedback(trial_prev, trialType, TF, pred_prev, A_wrong, A_correct, B_wrong, B_correct)
                        self.feedback_idx_allTrials[:,trial_prev] = self.calcfeedback_idx(trial_prev, trialType, TF, pred_prev, onevec)
                    elif run_perturbation:
                        #---------------------------------------#
                        # something wrong with pertfeedback_idx #
                        #---------------------------------------#
                        feedback_idx                              = self.pertfeedback_idx(feedback_reference, perturb_trial, trial_prev, pred_prev, TF, trialType)
                        feedback_input                            = self.feedback_convertIdx2Input(feedback_idx, convertIdx2Input)
                        self.feedback_idx_allTrials[:,trial_prev] = feedback_idx
                else:
                    feedback_noise = torch.normal(0, self.fbnoise, size=(self.batch_size, self.Ncells))
                    feedback = feedback_input + feedback_noise
            else:
                feedback = torch.zeros(self.batch_size, self.Ncells)

            # target presentation 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells)

            ##################
            ## PERTURBATION ##
            ##################
            # add perturbation input
            if run_reference:
                perturbation_input = 0
            elif run_perturbation:
                # set up perturbation input for the trial
                if ti_within == pert_start:
                    pertidx = (trial_idx == perturb_trial).to(torch.float32)
                    perturbation_input_tidx = torch.matmul(pertidx.view(self.batch_size,-1), vpert.view(-1,self.Ncells))
                # apply perturbation input
                if (ti_within > pert_start) and (ti_within < pert_end):
                    perturbation_input = perturbation_input_tidx
                else: 
                    perturbation_input = 0

            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells))
            
            # run snn one step
            total_input = feedback + perturbation_input
            self.extIn, self.u, self.out = self.one_step(stimt, total_input, extNoise)

            #####################################
            ## choice made at the end of trial ##
            #####################################
            # # compute output
            # if trial_idx < self.ntrials:
            #     if ti_within >= self.trial_duration[trial_idx] - self.out_duration:
            #         outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
            #         out_idx += 1

            #         if ti_within == self.trial_duration[trial_idx]-1:
            #             self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
            #             # reset temporary variables
            #             outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
            #             out_idx = 0
            #             trial_idx += 1

            ###########################################
            ## choice made at the beginning of trial ##
            ###########################################
            # compute output
            if trial_idx < self.ntrials:                
                ###################################
                # save initial rate of each trial #
                ###################################
                if (ti_within >= stim_on) and (ti_within < stim_off):
                    initRate[:,:,trial_idx] = initRate[:,:,trial_idx] + self.activation(self.u) / (stim_off - stim_on)

                #############################
                # save output of each trial #
                #############################
                if (ti_within >= choice_duration_start) and (ti_within < choice_duration_end):
                    outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
                    out_idx += 1                    
                    if ti_within == choice_duration_end-1:
                        self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
                        # reset temporary variables
                        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
                        out_idx = 0
                        trial_idx += 1
                                             
            if saved:                
                ############################################
                ## save projected activity of all batches ##
                ############################################
                self.xreverse[:,ti]    = torch.matmul(self.activation(self.u), vreverse_unitlen)
                self.outsim[:,ti,:]    = self.out
                self.extInsave[:,ti,:] = self.extIn[:10,:]
                self.usave[:,ti,:]     = self.u[:10,:]

        return initRate, self.xreverse, self.outchoice, self.extInsave, self.usave, self.outsim, self.feedback_idx_allTrials, uinit_reference
    


    def attractors_around_fixedpoints(self, stim0, stim1, trialType, target, A_correct, A_wrong, B_correct, B_wrong, saved, run_reference, run_perturbation, uinit_reference, feedback_reference, perturb_trial, vpert, pert_start, pert_end, args, vreverse_unitlen):
        if self.trial_duration_random:
            self.trial_duration = 100 * torch.randint(self.trial_min,self.trial_max, (self.ntrials,))
            self.SimTime = torch.sum(self.trial_duration)
            self.Nsteps = torch.tensor(int(self.SimTime / self.dt))
        else:
            None

        ##################
        ## PERTURBATION ##
        ##################
        # initial state
        if run_reference:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            uinit_reference = self.u
        elif run_perturbation:
            self.extIn, self.u, self.out, self.outchoice, self.feedback_idx_allTrials = self.initState(args)
            self.u = uinit_reference

        ##################
        ## PERTURBATION ##
        ##################
        # convert feedback_idx to feedback_input
        convertIdx2Input = torch.stack((A_wrong, A_correct, B_wrong, B_correct))

        # save variables
        if saved:
            self.extInsave, self.usave, self.outsim, self.xreverse = self.saveStatePerturbed_longtime()

        onevec = torch.ones(self.batch_size, device=args.device)
        bnum = torch.arange(self.savenum)
        trial_idx = 0
        out_idx = 0
        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
        Ntotal = self.Nsteps + self.feedback_duration
        for ti in range(Ntotal):

            ti_within = torch.tensor(ti) - torch.sum(self.trial_duration[0:trial_idx])

            # feedback
            # if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.stim_duration:
            if trial_idx > 0 and ti_within >= 0 and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                # compute TF at the first time step
                if ti_within == 0:
                    trial_prev = trial_idx - 1
                    outsym_prevtrial = tools_data.symmetric_output_onetrial(self.outchoice[:,trial_prev,:]) # outchoice: Nbatch x Ntrials x Nout
                    pred_prev = torch.argmax(outsym_prevtrial, axis=1) 
                    TF = pred_prev.eq(target[:,trial_prev]).float() # TF: batch_size
                    ##################
                    ## PERTURBATION ##
                    ##################
                    if run_reference:
                        feedback_input                            = self.calcfeedback(trial_prev, trialType, TF, pred_prev, A_wrong, A_correct, B_wrong, B_correct)
                        self.feedback_idx_allTrials[:,trial_prev] = self.calcfeedback_idx(trial_prev, trialType, TF, pred_prev, onevec)
                    elif run_perturbation:
                        #---------------------------------------#
                        # something wrong with pertfeedback_idx #
                        #---------------------------------------#
                        feedback_idx                              = self.pertfeedback_idx(feedback_reference, perturb_trial, trial_prev, pred_prev, TF, trialType)
                        feedback_input                            = self.feedback_convertIdx2Input(feedback_idx, convertIdx2Input)
                        self.feedback_idx_allTrials[:,trial_prev] = feedback_idx
                else:
                    feedback_noise = torch.normal(0, self.fbnoise, size=(self.batch_size, self.Ncells))
                    feedback = feedback_input + feedback_noise
            else:
                feedback = torch.zeros(self.batch_size, self.Ncells)

            # target presentation 
            if ti_within >= self.feedback_duration + self.intertrial_duration and ti_within < self.feedback_duration + self.intertrial_duration + self.stim_duration:
                if ti_within == self.feedback_duration + self.intertrial_duration:
                    stimt_on = torch.matmul((1 - trialType[:,trial_idx]).unsqueeze(1), stim0.unsqueeze(0)) + torch.matmul(trialType[:,trial_idx].unsqueeze(1), stim1.unsqueeze(0))
                else:
                    stimt = stimt_on
            else:
                stimt = torch.zeros(self.batch_size, self.Ncells)

            ##################
            ## PERTURBATION ##
            ##################
            # # add perturbation input
            # if run_reference:
            #     perturbation_input = 0
            # elif run_perturbation:
            #     # set up perturbation input for the trial
            #     if ti_within == pert_start:
            #         pertidx = (trial_idx == perturb_trial).to(torch.float32)
            #         perturbation_input_tidx = torch.matmul(pertidx.view(self.batch_size,-1), vpert.view(-1,self.Ncells))
            #     # apply perturbation input
            #     if (ti_within > pert_start) and (ti_within < pert_end):
            #         perturbation_input = perturbation_input_tidx
            #     else: 
            #         perturbation_input = 0
            perturbation_input = 0
            # external noise
            extNoise = torch.normal(0, self.noise, size=(self.batch_size, self.Ncells))
            
            # run snn one step
            total_input = feedback + perturbation_input
            
            
            #######################
            ## REMOVE ALL INPUTS ##
            #######################
            rmvInput     = (trial_idx < perturb_trial).to(torch.float32)
            diagRmvInput = torch.diag(rmvInput)
            stimt        = diagRmvInput @ stimt
            total_input  = diagRmvInput @ total_input

            self.extIn, self.u, self.out = self.one_step(stimt, total_input, extNoise)

            #####################################
            ## choice made at the end of trial ##
            #####################################
            # # compute output
            # if trial_idx < self.ntrials:
            #     if ti_within >= self.trial_duration[trial_idx] - self.out_duration:
            #         outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
            #         out_idx += 1

            #         if ti_within == self.trial_duration[trial_idx]-1:
            #             self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
            #             # reset temporary variables
            #             outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout)
            #             out_idx = 0
            #             trial_idx += 1

            ###########################################
            ## choice made at the beginning of trial ##
            ###########################################
            # compute output
            if trial_idx < self.ntrials:                
                choice_duration_start = self.feedback_duration + self.intertrial_duration + self.stim_duration + 50
                choice_duration_end   = choice_duration_start + self.out_duration
                if (ti_within >= choice_duration_start) and (ti_within < choice_duration_end):
                    outtrial[:,out_idx,:] = outtrial[:,out_idx,:] + self.out # out: batch x Nout
                    out_idx += 1

                    if ti_within == choice_duration_end-1:
                        self.outchoice[:,trial_idx,:] = torch.mean(outtrial, dim=1)
                        # reset temporary variables
                        outtrial = torch.zeros(self.batch_size, self.out_duration, self.Nout, device=args.device)
                        out_idx = 0
                        trial_idx += 1
                        
            if saved:
                ############################################
                ## save projected activity of all batches ##
                ############################################
                self.xreverse[:,ti]    = torch.matmul(self.activation(self.u), vreverse_unitlen)
                self.outsim[:,ti,:]    = self.out
                self.extInsave[:,ti,:] = self.extIn[:10,:]
                self.usave[:,ti,:]     = self.u[:10,:]

        return self.xreverse, self.outchoice, self.extInsave, self.usave, self.outsim, self.feedback_idx_allTrials, uinit_reference
    
