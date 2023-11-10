import sys
import os
import psi4
import numpy as np

sys.path.insert(0, "../common")


import bo_helper
import helper_HF
import LIST_help
from pkg_resources import parse_version
import scipy.linalg
from scipy.linalg import fractional_matrix_power


############################################################################
# Diagonalize routine
def build_orbitals(diag,Lowdin,fragdocc,nbasfrag):
    Fp = psi4.core.triplet(Lowdin, diag, Lowdin, True, False, True)

    Cp = psi4.core.Matrix(nbasfrag, nbasfrag)
    eigvals = psi4.core.Vector(nbasfrag)
    Fp.diagonalize(Cp, eigvals, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(Lowdin, Cp, False, False)

    Cocc = psi4.core.Matrix(nbasfrag, fragdocc)
    Cocc.np[:] = C.np[:, :fragdocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D,eigvals

# replacement for build_orbitals()
def get_orbitals(scf_accel,atype,Amat,Fock,D_in,ndocc,\
        diis_err,use_transpose=False,debug=False,opt_container=None):

    if atype == 'diis':
       scf_accel.add(Fock, diis_err)
       Fock_extra = scf_accel.extrapolate()
       #diagonalize
       Fp = Amat.dot(Fock_extra).dot(Amat)
       e, C2 = np.linalg.eigh(Fp)
       C = Amat.dot(C2)
       Cocc = C[:, :ndocc]
       Dold = D_in 
       D = np.matmul(Cocc,Cocc.T) 
        

    elif atype == 'list':
       ######################################################## 
  
       # from the guess (input) occupied orbitals -> Fock
       Fock_extra, Dm = scf_accel.extrapolate(use_transpose)
  
       # check extrapolated Fock
       if debug:
           if SCF_ITER > 1:  
              print("Max diff (F_i -F_e) : %.8f" % np.max( Fock-Fock_extra))
       
  
       Dold = D_in 
       Cocc,e = scf_accel.diagonalize(Fock_extra,Amat,ndocc)
       D = np.matmul(Cocc,Cocc.T) 
       # D_out
       cHKS_e = scf_accel.finalize(Fock_extra)

       #cHKS_old = opt_container['eHKS']
       #opt_container['EDiff'].append(cHKS_e - cHKS_old)
       #store cHKS_e as the old energy value 
       #opt_container['eHKS'] = cHKS_e
       ########################################################
    else:
       raise TypeError("invalid acceleration type")
    return Cocc,e,D,Dold
############################################################################
#FntFactory class
# this implementation account only for two subsystem, namely A and B
class FntFactory():
    def __init__(self,debug=False,loewdin=False,outfile=sys.stderr):
       self.__thaw = None
       self.__frozn = None # should contain a 'list' of frozen  molecular fragments
       self.__Vminus = None
       self.__Vplus = None
       self.__supermol = False
       self.__debug = debug
       self.__loewdin = loewdin
       self.__prop_ortho_mtx = None # contains the matrix to transform rt quantities to the "ortho" basis
       self.__rt_Cocc = None        # contains the propagated coeff
       self.__rt_mid_mat = None     # a midpoint aux matrix  for rt_Cocc
       self.__outfile = outfile
       self.__fnt_mag = None

    def initialize(self,frags_container):
      
      self.__fnt_mag = frags_container 
      #here we have to assign the subsystems
      self.__thaw = frags_container[0]
      self.__frozn = frags_container[1] # is a list of frozen fragments

      if self.is_full_basis():
           self.__supermol = True
           fullovap = self.__thaw.full_ovapm()
           self.__Vminus = fractional_matrix_power(fullovap, -0.5)
    
    def status(self):
        print("relaxed : '%i'" % self.__thaw.whoIam())
        frozen_ids = str()
        for elm in self.__frozn:
            frozen_ids += str( elm.whoIam() ) + ', '
        print("frozen : '%s'" % frozen_ids)

    def thawed_id(self):
        frag_label = self.__thaw.whoIam()
        return frag_label

    def clean(self):
        self.__thaw = None
        self.__frozn = None
    def is_full_basis(self):
        res = False
        if self.__thaw.full_ovapm().shape[0] ==  self.__thaw.S().shape[0]:
           res = True  
        return res
############################################################################

    
    def thaw_active(self):

        #Amat is S^{-0.5}
        Amat =self.__thaw.Amat()
        Ovap =psi4.core.Matrix.from_array( self.__thaw.S() ) # overlap
        nbf = self.__thaw.nbf()
        if self.__supermol:
            if nbf != self.__thaw.full_ovapm().shape[0]:
                raise
        ndoccA =self.__thaw.ndocc()

        #if self.__debug:
        #  print('------------------------------------------')
        #  print("I am fragment '%s'\n" % self.__thaw.whoIam())
        #  print("Print infos: ...")
        #  print("n doubly occupied : %i" % ndoccA)
        #  print("n basis set functions : %i" % nbf)
        #  print('------------------------------------------')
        
        
         
        # a local copy
        ffrozen_list = self.__frozn.copy() # necessary?

        # res_gather, a tuple containg a list of matrix of MO(occ) coeff and 
        # a regular supermolecular MO(occ) coeff matrix formed, arranging the 
        # MO(occ frag) matrix slices side-by-side
        
        res_gather = self.__thaw.Cocc_gather(ffrozen_list)
        Cocc_sup = res_gather[1]
        #print("len(res_gather) : %i\n" % len(res_gather))
        #print("len(res_gather[0]) : %i\n" % len(res_gather[0]))
        #print("type(res_gather[0] : %s\n" % type(res_gather[0]))
        #print("type(res_gather[1] : %s\n" % type(res_gather[1]))

        #apply orthogonalization ?
        #check if orthogonal
        if self.__debug:
          full_ovap = self.__thaw.full_ovapm()
          # Oplus is the C.T S C product
          Oplus= np.matmul(Cocc_sup.T, np.matmul(full_ovap,Cocc_sup) )
          check_orth = np.allclose(np.eye(Cocc_sup.shape[1]),Oplus)
          self.__outfile.write("Cocc[sup] orthogonal : %s\n" % check_orth)
        else:
          Oplus = None

        if self.__loewdin:
            if Oplus is None:
                full_ovap = self.__thaw.full_ovapm()
                Oplus = np.matmul(Cocc_sup.T, np.matmul(full_ovap,Cocc_sup) )
            loewdin_mat = fractional_matrix_power(Oplus,-0.5)
            # do loewdin
            Cocc_sup = np.matmul(Cocc_sup,loewdin_mat)
            #test_ovap =np.matmul(Cocc_sup.T, np.matmul(full_ovap,Cocc_sup) )
            #check_orth = np.allclose(np.eye(Cocc_sup.shape[1]),test_ovap)
            #print("Cocc(sup) orth: %s\n" % check_orth)
             

        F_emb,projector,ene = self.__thaw.get_Fock(res_gather,return_ene=True) # res_gather contains Cocc_sup
        F_emb += projector
        # for test
        #if self.__thaw.Da().shape[0] == self.__thaw.full_ovapm().shape[0]:
        #    print("n.el in thawed frag: %.5f" % ( np.trace(np.matmul(self.__thaw.Da(),self.__thaw.full_ovapm())).real) )
        #    print("D dim : %i,%i\n" % (self.__thaw.Da().shape) )
        #    print("Ovap of full basis: %s\n" % np.allclose(Ovap.np,self.__thaw.full_ovapm()))
        
        D_AA = psi4.core.Matrix.from_array( self.__thaw.Da() )
        G_AA, twoel_ene = self.__thaw.G()
        core = F_emb - G_AA
        # SCF energy and update : E (DFT_in_DFT) = Tr(Hcore [D_AA+D_BB]) +J[D_AA +D_BB] + Exc[D_AA + D_BB]
        #SCF_E, no nuclear contribution
        try:
           one_el_trace = np.matmul(D_AA,core)
        except ValueError:
           print("IndexError catched\n")
           tmp_AA = np.zeros_like(core) 
           l1,l2 = self.__thaw.fake_limits()
           tmp_AA[l1:l2+1,l1:l2+1] = D_AA
           one_el_trace = np.matmul(tmp_AA,core)
           # alias
           D_AA = psi4.core.Matrix.from_array(tmp_AA)
        one_el = 2.0*np.trace(one_el_trace)   
        SCF_E = one_el + twoel_ene
        self.__thaw.set_e_scf(SCF_E) 
        
        # DIIS error build and update
        diis_e = np.matmul(F_emb, np.matmul( D_AA.np, Ovap.np))
        diis_e -= np.matmul(Ovap.np, np.matmul( D_AA.np, F_emb))
        diis_e = np.matmul(Amat, np.matmul(diis_e, Amat) )
 
        F_emb = psi4.core.Matrix.from_array(F_emb)

        # psi4.core matmul functions
        #diis_e = psi4.core.triplet(F_emb, D_AA, Ovap, False, False, False)
        #diis_e.subtract(psi4.core.triplet(Ovap, D_AA, F_emb, False, False, False))
        #diis_e = psi4.core.triplet(psi4.core.Matrix.from_array(Amat), diis_e, psi4.core.Matrix.from_array(Amat), False, False, False)
        
        #dRMS = diis_e.rms()
        
        #define
        eigvals = None
        #print(self.__thaw.niter())
        dRMS = np.mean(diis_e**2)**0.5
        #pure DIIS
        if '_diis' in self.__thaw.acc_scheme(): 
          #pure diis
          if self.__thaw.acc_scheme() == '_diis':
              self.__thaw.diis()[1].add(F_emb, psi4.core.Matrix.from_array(diis_e))
              # extrapolation step    
              F_emb = psi4.core.Matrix.from_array(self.__thaw.diis()[1].extrapolate())
          
          else: 
              # a_diis/e_diis
              # codition on the iteration

              try:
                max_ediis = int(self.__thaw.acc_param()[2])
              except ValueError:
                if self.__debug:
                    print("using default for max_ediis\n")
                    max_ediis = 6

              # in the intial scf iteration(max_ediis) we use A/E-DIIS
              if self.__thaw.niter() <= max_ediis:
                 #start filling the diis vector? <- NO 
                 #self.__thaw.diis()[1].add(F_emb, psi4.core.Matrix.from_array(diis_e))
                 # populate A/E-diis quantities
                 self.__thaw.diis()[0].add(np.asarray(F_emb), np.asarray(D_AA),SCF_E)
                 F_emb = psi4.core.Matrix.from_array(self.__thaw.diis()[0].extrapolate())
              else:
                 #switch to DIIS
                 self.__thaw.acc_scheme('_diis')

          self.__thaw.set_Femb( np.array(F_emb) )
          # Diagonalize Fock matrix

          #old -> replace
          C_AA, Cocc_AA, dummy,eigvals = build_orbitals(F_emb,psi4.core.Matrix.from_array(Amat),ndoccA,nbf)
          #update the orbitals of the thawed fragment
          self.__thaw.set_Ca(np.array(C_AA))

        # imaginary time prop
        elif 'imag_time' in self.__thaw.acc_scheme():
              try:
                max_ediis = int(self.__thaw.acc_param()[2])
              except ValueError:
                if self.__debug:
                    print("using default for max_ediis\n")
                    max_ediis = 6

              # in the intial scf iteration(max_ediis) we use A/E-DIIS
              if self.__thaw.niter() <= max_ediis:
                 self.__thaw.imag_time()[0].add_F(F_emb)
                 self.__thaw.imag_time()[1].add(np.asarray(F_emb), np.asarray(D_AA),SCF_E)  # diis step

                 # extrapolation step 
                 #print(type(self.__thaw.imag_time()[1]))
                 #print(type(self.__thaw.imag_time()[0]))
                 F_emb = self.__thaw.imag_time()[1].extrapolate()
                 if not isinstance(F_emb,np.ndarray):
                     raise ValueError("F_emb must be np.ndarray\n")
                 F_emb=psi4.core.Matrix.from_array(F_emb) #cast into psi4.core.Matrix
                 C_AA, Cocc_AA, dummy,eigvals = build_orbitals(F_emb,psi4.core.Matrix.from_array(Amat),\
                                                                    ndoccA,nbf)
              else:   
                 self.__thaw.acc_scheme('imag_time')
                 self.__thaw.set_Femb( np.array(F_emb) )
                 self.__thaw.imag_time()[0].compute()
                 Cocc_AA = self.__thaw.imag_time()[0].Cocc()


              self.__thaw.set_Ca(np.asarray(Cocc_AA),'OCC')


        elif self.__thaw.acc_scheme() == 'list':
              self.__thaw.LiST().add_Fock(F_emb.np)
              # D_m not used
              F_emb,D_m = self.__thaw.LiST().extrapolate()

              C_AA, Cocc_AA, eigvals = self.__thaw.LiST().diagonalize(F_emb,Amat,ndoccA)

              if self.__supermol:
                  if C_AA.shape[0] != nbf:
                      raise ValueError("wrong dimension from LiST.diagonalize()\n")

              self.__thaw.set_Ca(C_AA)

              # use the new Cocc_sup to calculate the Fock_out in the finalize()
              Cocc_gather = self.__thaw.Cocc_gather(ffrozen_list)
              #print("! Cocc_sup is %s" % type(Cocc_sup))

              self.__thaw.LiST().set_Csup(Cocc_gather)

              # F_emb (extrapolated) is used as input for the next step (through the finalize step)
              self.__thaw.LiST().finalize(self.__thaw, F_emb)  
              self.__thaw.set_Femb( np.array(F_emb) )
        elif self.__thaw.acc_scheme() == 'lv_shift':
                   #check the determinant of Ctrial matrix, in the case of supermolecular basis  the matrix may be ill-conditioned
                   Ctrial = self.__thaw.Ca_subset('ALL')
                   
                   if self.__supermol:
                      if Ctrial.shape[0] != nbf:
                          # get the trial MO
                          F_tmp = np.matmul(self.__Vminus.T, np.matmul(F_emb,self.__Vminus) )
                          eigs, Ctrial = np.linalg.eigh(F_tmp)
                          Ctrial = np.matmul(self.__Vminus,Ctrial)
                      try:   
                         test_det = np.linalg.det(Ctrial)
                      except np.linalg.LinAlgError:
                          print("shape Ctrial: %i,%i\n" % Ctrial.shape)
                      if abs(test_det) < 1.0e-10:  # a matrix with null determinant is tricky to invert  
                          raise

                   muval = self.__thaw.acc_param()[1]
                   Fp = np.matmul(Ctrial.T, np.matmul(F_emb,Ctrial) )   # Express the Fock in the trial MO basis
                   lv = np.empty(nbf-ndoccA)
                   lv.fill(muval)
                   diag = np.zeros(nbf)
                   diag[ndoccA:] = lv
                   lvmat = np.diagflat(diag)
    
                   Fp = Fp + lvmat
            
                   eigs, C2 = np.linalg.eigh(Fp)             # get the 'improved' eigenvectors
                   idx = eigs.argsort()[::]
                   C_AA = Ctrial.dot(C2)                     # Back transform, Eqn. 3.174
                   eigvals = eigs[idx]
                   #for back-compatibility
                   C_AA = C_AA[:,idx]
                   self.__thaw.set_Ca(C_AA)
                   C_inv = np.linalg.inv(Ctrial)  # invert Ctrial
                   F_emb = np.matmul(C_inv.T, np.matmul(Fp,C_inv) )   # Express the Fock in the trial MO basis
                   self.__thaw.set_Femb( np.array(F_emb) )
        else:
                   raise Exception("wrong keyword for scf acceleration\n")

        
        self.__thaw.set_eps(eigvals)
        self.__thaw.finalize()
        return SCF_E, ene,dRMS

    def actual_boost(self):
        return self.__thaw.acc_scheme()

    def finalize(self): 
        if not isinstance(self.__fnt_mag,list):
            raise TypeError("input must be list [of lists]\n")
        if len(self.__fnt_mag) <2:
           return None
        self.__fnt_mag[1].append(self.__fnt_mag[0])
        self.__fnt_mag[0] = self.__fnt_mag[1].pop(0)
        
        #update intrenal frags_list var

        self.__thaw = self.__fnt_mag[0]
        self.__frozn = self.__fnt_mag[1] # is a list of frozen fragments
        return 0

    def set_rt_common(self,pulse_opts,ortho_mtx):
        self.__prop_ortho_mtx = ortho_mtx
        return None

    def get_Ca(self):
        return self.__C_AA  # this matrix in the "supermolecular" setting should include also the projeted-out MOs of the other fragment.

    def rt_step(self): # only for non-hybrid funcs

        return None
