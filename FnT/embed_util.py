import sys
import os
import psi4
import numpy as np
sys.path.append(os.environ['PSI4_BOMME_RLMO'])
import bo_helper
import helper_HF
import LIST_help
from pkg_resources import parse_version
import scipy.linalg
from scipy.linalg import fractional_matrix_power


# Huzinaga projector has to be added explicitly
def Hembed(Hcore,jk,CoccA,CoccB,func,basisset):
    
    nbf = Hcore.shape[0]
    # Build G(D_AA + D_BB) on monomolecular basis
    nbf_tot = basisset.nbf()

    ndoccA = CoccA.shape[1] 
    ndoccB = CoccB.shape[1] 
    totocc = ndoccA + ndoccB
    
    limit = nbf
    
    Cocc = np.zeros((nbf_tot,totocc)) 
    Cocc[:limit,:ndoccA] = CoccA
    if CoccB.shape[0] > (nbf_tot-nbf): # we are passing Cocc_BB (nb_tot,ndoccB) -> supermolecular basis set
       Cocc[:,ndoccA:] = CoccB 
    else:
       Cocc[limit:,ndoccA:] = CoccB 

    Dmat = np.matmul(Cocc,Cocc.T)

    jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
    jk.compute()
    jk.C_clear()
    J=np.array(jk.J()[0])[:nbf:,:nbf] #copy into J
    # Build Vxc matrix 
    #D must be a psi4.core.Matrix object not a numpy.narray 
    if func == 'hf':
        K=np.array(jk.K()[0])[:nbf:,:nbf] #copy into K
        fock = Hcore + (2.00*J -K)
    else:

        restricted = True
        
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
           build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
           build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(func, restricted)[0] 
        sup.set_deriv(2)
        sup.allocate() 
        vname = "RV" 
        if not restricted: vname = "UV"
        potential=psi4.core.VBase.build(basisset,sup,vname)
        Dm=psi4.core.Matrix.from_array(Dmat.real) 
        potential.initialize()
        potential.set_D([Dm])
        
        V=psi4.core.Matrix(nbf_tot,nbf_tot)
        potential.compute_V([V])
        potential.finalize()
        #compute the corresponding XC energy (low level)
        #Exc= potential.quadrature_values()["FUNCTIONAL"]

        if sup.is_x_hybrid():
          #
          #raise Exception("Low level theory functional is Hybrid?\n")
          alpha = sup.x_alpha()
          K = np.array(jk.K()[0])
          V.add(psi4.core.Matrix.from_array(-alpha*K))
          #Exc += -alpha*np.trace(np.matmul(D,K))
        V = np.asarray(V)[:nbf,:nbf] 
        fock = Hcore + (2.00*J + V)
    return fock
############################################################################
def extract_fock_emb(nbf,Fmat,frag_id):
    nbf = self.__thaw.nbf()
    # extract [h+ G(D_AA + D_BB)] on monomolecular basis
    nbf_tot = Fmat.shape[0]
    
    
    if frag_id == 'A':
        f_thaw = Fmat[:nbf,:nbf]
    
    elif frag_id == 'B':
        f_thaw = Fmat[-nbf:,-nbf:]
    else:
        print("check fragment labels")
        f_thaw = None
    return f_thaw
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
    def __init__(self,debug=False):
       self.__thaw = None
       self.__frozn = None
       self.__debug = debug


    def initialize(self,moldict):
      
      #here we have to broadcast the subsystems
      self.__thaw = moldict['thaw']
      self.__frozn = moldict['frozn']
      nbf = self.__thaw.nbf()
      name = self.__thaw.whoIam()
    
    def status(self):
        print("relaxed : '%s'" % self.__thaw.whoIam())
        print("frozen : '%s'" % self.__frozn.whoIam())
    def clean(self):
        self.__thaw = None
        self.__frozn = None

############################################################################

    
    def thaw_active(self):

        #Amat is S^{-0.5}
        Amat =self.__thaw.Amat()
        Ovap =psi4.core.Matrix.from_array( self.__thaw.S() ) # overlap
        nbf = self.__thaw.nbf()
        ndoccA =self.__thaw.ndocc()
        if self.__debug:
          print('------------------------------------------')
          print("I am fragment '%s'\n" % self.__thaw.whoIam())
          print("Print infos: ...")
          print("n doubly occupied : %i" % ndoccA)
          print("n basis set functions : %i" % nbf)
          print('------------------------------------------')
        
        
        
        acc_scheme = self.__thaw.acc_scheme() 

        Cocc_frozn = self.__frozn.Ca_subset('OCC')
        Cocc_sup = self.__thaw.Cocc_sum(Cocc_frozn)
        
        frag_label = self.__thaw.whoIam()
        
       
        F_emb = self.__thaw.get_Fock(Cocc_sup)
        
        D_AA = psi4.core.Matrix.from_array( self.__thaw.Da() )
        G_AA, twoel_ene = self.__thaw.G()
        core = F_emb - G_AA
        # SCF energy and update : E (DFT_in_DFT) = Tr(Hcore [D_AA+D_BB]) +J[D_AA +D_BB] + Exc[D_AA + D_BB]
        #SCF_E, no nuclear contribution
        one_el =2.0*np.trace(np.matmul(D_AA,core))
        SCF_E = one_el + twoel_ene
        
        
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

        dRMS = np.mean(diis_e**2)**0.5
        if acc_scheme == 'diis':
          #print("DIIS")
          self.__thaw.diis().add(F_emb, psi4.core.Matrix.from_array(diis_e))

          # extrapolation step    
          F_emb = psi4.core.Matrix.from_array(self.__thaw.diis().extrapolate())
          self.__thaw.set_Femb( np.array(F_emb) )
          # Diagonalize Fock matrix

          #old -> replace
          C_AA, Cocc_AA, Dens,eigvals = build_orbitals(F_emb,psi4.core.Matrix.from_array(Amat),ndoccA,nbf)
          #update the orbitals of the thawed fragment
          self.__thaw.set_Ca(C_AA)

        # under construction

        elif acc_scheme == 'list':
              self.__thaw.LiST().add_Fock(F_emb.np)
              # D_m not used
              F_emb,D_m = self.__thaw.LiST().extrapolate()

              C_AA, Cocc_AA, eigvals = self.__thaw.LiST().diagonalize(F_emb,Amat,ndoccA)


              self.__thaw.set_Ca(C_AA)

              # use the new Cocc_sup to calculate the Fock_out in the finalize()
              Cocc_sup = self.__thaw.Cocc_sum(Cocc_frozn)
              #print("! Cocc_sup is %s" % type(Cocc_sup))

              self.__thaw.LiST().set_Csup(Cocc_sup)

              # F_emb (extrapolated) is used as input for the next step (through the finalize step)
              self.__thaw.LiST().finalize(F_emb)  
              self.__thaw.set_Femb( np.array(F_emb) )
        elif acc_scheme == 'lv_shift':
                   muval = 0.5
                   Ctrial = self.__thaw.Ca_subset('ALL')
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
                   C_inv = np.linalg.inv(Ctrial)
                   F_emb = np.matmul(C_inv.T, np.matmul(Fp,C_inv) )   # Express the Fock in the trial MO basis
                   self.__thaw.set_Femb( np.array(F_emb) )


        
        
        self.__thaw.set_eps(eigvals)
        return SCF_E, dRMS
