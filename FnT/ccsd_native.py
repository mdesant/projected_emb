import numpy as np
import psi4
import re
#FnT_ene include the nuclear contribution
# might be possible other wf other than  ccsd, ie ccsd(t), fno- ..
def wfn_in_dft(frag_mag,wfn_cc,cc_outfile,ene_sup,FnT_ene,wf_type='ccsd'):
        # frag_mag should be a results dataclass        
        # the {A} basis functions 

        frag_cc = frag_mag.frag
        basis_wfn = wfn_cc.basisset()
        num_bf = basis_wfn.nbf()
        #same from wfn_cc.Da()
        Da = frag_cc.Da()

        if num_bf != Da.shape[0] : #
            raise Exception("ccsd_native: check basis set dimension\n")


        psi4.core.set_output_file(cc_outfile,False)
        #psi4.set_options({'r_convergence': 1e-7})
        dum0 = psi4.energy(wf_type,ref_wfn=wfn_cc)

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

        #mints = psi4.core.MintsHelper(basis_wfn)
        #S = np.asarray(mints.ao_overlap())
        S = frag_cc.S()
 
        
        #the Hcore_A_in_B from DFT_in_DFT
        hcore_A_in_B = frag_mag.Hcore
        Ecore = 2.0*np.trace(np.matmul(hcore_A_in_B,Da))
        print("E. core (A in B)  : %.8f" % Ecore)
        ### CC
    
        #the reference Fock
        F_A = wfn_cc.Fa()  
        
        E_DFT_in_DFT = FnT_ene # E_nuclear is included
 
        
        E_inter2=np.einsum('pq,pq->', F_A +hcore_A_in_B , Da) # F_A is the WF (ref) Hamiltonian (modified by hcore), correlation energy summed below 
    
 
        dft_correction = ene_sup-FnT_ene
        tE_WF_in_DFT =  E_DFT_in_DFT -(Ecore + frag_mag.twoel_ene) +E_inter2 +  ecorr
        E_WF_in_DFT = tE_WF_in_DFT +dft_correction 
        #print("E (WF-in-DFT) = %.8f" % E_WF_in_DFT)
        return E_WF_in_DFT
