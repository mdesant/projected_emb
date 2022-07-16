import numpy as np
import psi4
import re
#FnT_ene include the nuclear contribution
# might be possible other wf other than  ccsd, ie ccsd(t), fno- ..
def wfn_in_dft(molA,molB,isoA,cc_outfile,ene_sup,FnT_ene,wf_type='ccsd'):
        psi4.core.set_output_file(cc_outfile,False)
        #psi4.set_options({'r_convergence': 1e-7})
        dum0 = psi4.energy(wf_type,ref_wfn=isoA)

        with open(cc_outfile) as file_in: 
             string_file = file_in.readlines()

        
        if 'fno'  in wf_type: 
           wf_type = wf_type[4:] 
        wf_type = wf_type.upper() 
        starget =  '  ' + wf_type + ' correlation energy' 

        for a in string_file: 
             if starget in a: 
               print("From output : %s" %a) 
               res=re.split('(= |: | )',a) 
        ecorr = float(res[-1])

        print("CC correlation energy = %1.15f" % ecorr)

        # the {A} basis functions 
        bset = isoA.basisset()
        nAA = bset.nbf()
        #same from molA.Da()
        Da = isoA.Da()

        mints = psi4.core.MintsHelper(bset)
        
        S = np.asarray(mints.ao_overlap())
 
        #Gmat object may contain J+K or J+ Vxc depending on the 'func'
        G_termA, twoel_eneA = molA.G()
 
        
        #the Hcore_A_in_B from DFT_in_DFT
        # Femb() is an handle to the last computed embedded Fock
        hcore_A_in_B = molA.Femb() - G_termA
        Ecore = 2.0*np.trace(np.matmul(hcore_A_in_B,Da))
        print("E. core (A in B)  : %.8f" % Ecore)
        ### CC
    
        #the effective Hamiltonian for the WF level
        F_A = molA.Femb()   # since the 'functional' is set at the beginning of the computation
                            # the fock matrix reads as : hcore(embedding) + G_term(functional). 
                            # In other words this lead to a  post HF-like calculation 
                            # in which F =  Fock(functional) is used as starting point 
        E_DFT_in_DFT = FnT_ene # E_nuclear is included
 
 
        E_inter2=np.einsum('pq,pq->', F_A +hcore_A_in_B , Da) 
 
        dft_correction = ene_sup-FnT_ene
        tE_WF_in_DFT =  E_DFT_in_DFT -(Ecore + twoel_eneA) +E_inter2 +  ecorr
        E_WF_in_DFT = tE_WF_in_DFT +dft_correction 
        #print("E (WF-in-DFT) = %.8f" % E_WF_in_DFT)
        return E_WF_in_DFT
