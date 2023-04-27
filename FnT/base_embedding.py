"""
Base class for projected embeddding
"""
__authors__   =  "Matteo De Santis"
__credits__   =  ["Matteo De Santis"]

__copyright__ = "(c) 2022, MDS"
__license__   = "BSD-3-Clause"
__date__      = "2022-03-01"

import sys
import os
sys.path.insert(0,'../common')
import it_util
import numpy as np
import psi4
from pkg_resources import parse_version
import helper_HF
import LIST_help
import scipy
from scipy.linalg import fractional_matrix_power
##################################################################
# acc_opts = [kind, maxvec, variant=None]
#    kind : 'diss' | 'list_(i/d)' (type=str)
#    maxvec
#    variant : Use the transposed 'B' matrix? (type=bool)
class RHF_embedding_base():

  def __init__(self,nb_frag,ndocc_frag,nb_super,\
                                ndocc_super,funcname,tag,fout=sys.stderr,supermol=False):
      self.__eps = None
      self.__Cocc = None
      self.__Ccoeff = None
      self.__C_midb = None      # aux MO coeff at previous midpont for imaginary time prop
      self.__tagname = None
      self.__scfboost = None
      self.__Femb = None   # ?
      self.__accel = None
      self.__acc_opts = None
      self.__bset_mono = None   # fragment basisset
      self.__bset_sup = None    # supermolecular basisset
      self.__jk_mono = None     # JK object built  from fragment basis
      self.__jk_sup = None      # JK object built from supermol basis
      self.__Honeel = None      # one electron supramolecular Hamiltonian
      self.__S = None           # {AB} basis function overlap
      self.__ortho = None       # Ortho = S_xx^{-0.5} x= A,B
      self.__nbf = nb_frag
      self.__ndocc = ndocc_frag
      self.__nb_super = nb_super
      self.__ndocc_super = ndocc_super
      self.__funcname = funcname
      self.set_tag(tag)
      self.__stdout = fout
      self.__supermol = supermol
  def initialize(self,ovap,basis_mono,basis_sup,Hsup,Ccoeff,acc_opts,target='direct',debug=False):
      self.__S = ovap
      if self.__tagname == 'A':
        ovap_sub = ovap[:self.__nbf,:self.__nbf]
      elif self.__tagname == 'B':
        ovap_sub = ovap[-self.__nbf:,-self.__nbf:]
      self.__ortho = fractional_matrix_power(ovap_sub, -0.5)



      self.__Cocc =   np.array(Ccoeff)[:,:self.__ndocc]
      self.__Ccoeff = np.array(Ccoeff)

      #JK object (monomol basis)
      self.__bset_mono =  basis_mono
      basis_name = basis_mono.name()

      target = target.upper()
      if target == 'DF' or target == 'MEM_DF' or target == 'DISK_DF':
          auxb = psi4.core.BasisSet.build(embmol,"DF_BASIS_SCF", "", fitrole="JKFIT",other=basis_name)
          jk = psi4.core.JK.build_JK(basis_mono,auxb)
      else:
          jk = psi4.core.JK.build(basis_mono)
      jk.set_memory(int(4.0e9))  # 1GB
      jk.set_do_wK(False)
      jk.initialize()

      self.__jk_mono = jk
      if (self.__jk_mono is None):
         raise Exception("Error in JK instance")
      
      # set the total basis set object
      self.__bset_sup =  basis_sup

      if not self.__supermol:          # if the mono molecular basis set != super mol. basis set
          #JK object (supermol basis)
          basis_name = basis_sup.name()
          if target == 'DF' or target == 'MEM_DF' or target == 'DISK_DF':
              auxb = psi4.core.BasisSet.build(embmol,"DF_BASIS_SCF", "", fitrole="JKFIT",other=basis_name)
              jk_sup = psi4.core.JK.build_JK(basis_sup,auxb)
          else:
              jk_sup = psi4.core.JK.build(basis_sup)
          jk_sup.set_memory(int(4.0e9))  # 1GB
          jk_sup.set_do_wK(False)
          jk_sup.initialize()

          self.__jk_sup = jk_sup
      else:
          self.__jk_sup = jk

      if (self.__jk_sup is None):
         raise Exception("Error in JK instance")

      # H core
      if (Hsup.shape[0] != self.__nb_super):
         raise Exception("Wrong Hsup dim[0]")

      self.__Honeel = Hsup

      if self.__supermol: # if the supermolecular basis is used , for the imaginary time  propagation we opt for the orthogonalized atomic basis , as propagation basis 
        Vminus =   fractional_matrix_power(ovap, -0.5)      
      else:
        Vminus = Ccoeff
      # prepare a dictionary of commons
      scf_common = {'nbf' : self.__nbf, 'nbf_tot': self.__nb_super, 'occ_num' : self.__ndocc,  'Hcore' : Hsup, 'ovap':ovap,'mono_basis': basis_mono, 'sup_basis': basis_sup,'jk' : self.__jk_sup, 'jk_mono' : self.__jk_mono, 'ftype' : self.__funcname, 'frag_id': self.__tagname}
 
      self.__acc_opts = acc_opts
 
      # acceleration engine name
      self.__accel = acc_opts[0]
      # set the acceleration method and initialize
      max_vec = int(acc_opts[1])
      if self.__accel == 'diis':
          self.__scfboost = helper_HF.DIIS_helper(max_vec) 
      elif self.__accel == 'list':
          self.__scfboost = list_baseclass(self.__Cocc,scf_common,acc_opts,debug)
      elif self.__accel == 'imag_time':
          self.__scfboost = itime_base(Vminus,self.__Cocc,acc_opts,debug)   # D^{AO} = Vminus D^{orth} Vminus.T
      elif self.__accel == 'lv_shift':
          self.__scfboost = None

  def diis(self):
      acc_type  = self.__accel
      if acc_type != 'diis':
         raise Exception("Not supposed to use diis")
      return self.__scfboost
  def imag_time(self):
      acc_type  = self.__accel
      if acc_type != 'imag_time':
         raise Exception("Not supposed to use imaginary time propagation")
      return self.__scfboost
  # LiST : TO BE REMOVED or IMPROVED
  def LiST(self): 
      acc_type  = self.__accel
      if acc_type != 'list':
         raise Exception("Not supposed to use list")
      return self.__scfboost
  
  def acc_param(self):
      return self.__acc_opts

  def acc_scheme(self):
      res=self.__accel
      return res

  def jk_super(self):
      res = self.__jk_sup
      return res

  def get_Fock(self,Csup,return_ene=False):
      
      nbf = self.__nbf
      nbf_tot = self.__nb_super
      Hcore = self.__Honeel
      ovap = self.__S
      basis = self.__bset_sup
      jk = self.__jk_sup
      ftype = self.__funcname
      frag_id = self.__tagname
      occ_num = self.__ndocc

      fock,ene =  Fock_emb(nbf,nbf_tot,occ_num,Hcore,ovap,Csup,basis,jk,ftype,frag_id)

      if return_ene:
       return fock,ene
      else:
       return fock

  def G(self,replace_func=None):

    Cocc = self.__Cocc
    basis = self.__bset_mono
    if replace_func is not None:
      Gmat,ene = twoel(Cocc,basis,self.__jk_mono,replace_func)
    else:
      Gmat,ene = twoel(Cocc,basis,self.__jk_mono,self.__funcname)
    return Gmat, ene

  def set_tag(self,label):
      self.__tagname = label
      
  def set_Femb(self,Fmat):
      if (Fmat.shape[0] != self.__nbf):
         raise Exception("Wrong Fmat dim[0]")
      if (Fmat.shape[1] != self.__nbf):
         raise Exception("Wrong Fmat dim[1]")
      if not isinstance(Fmat,(np.ndarray)):
         raise TypeError("input must be a numpy.ndarray")
      self.__Femb = Fmat

  def Femb(self):         # function to return the Fock for correlated calculation
      return self.__Femb

  def set_Ca(self,Cmat, dest='ALL'):
      #check dimensions
      if not isinstance(Cmat,np.ndarray):
          raise Exception("not np.ndarray")
      if dest == 'ALL':  
         if (Cmat.shape[0] != self.__nbf):
            raise Exception("Wrong Cmat dim[0]")
         if (Cmat.shape[1] != self.__nbf):
            raise Exception("Wrong Cmat dim[1]")
         self.__Cocc =   np.array(Cmat)[:,:self.__ndocc]
         self.__Ccoeff = np.array(Cmat)
      elif dest == 'OCC':
         self.__Cocc =   np.array(Cmat)[:,:self.__ndocc]
      else :
          raise Exception("check MOs usage")
          
  def Ca_subset(self,tag='ALL'):
      if tag == 'OCC':
          res =self.__Cocc
      elif tag == 'VIRT':
          res = self.__Ccoeff[:,self.__ndocc:]
      elif tag == 'ALL':
          res = self.__Ccoeff
      return res
  def Cocc_sum(self,Cmat):
      #print("Cmat dim : %i,%i\n" % (Cmat.shape)) 
      nb_tot = self.__nb_super
      ndocc_tot = self.__ndocc_super
      Cocc_super = np.zeros( (nb_tot,ndocc_tot) )
      #print("DEBUG Cocc_super is [%i,%i]" % (Cocc_super.shape[0],Cocc_super.shape[1]))
      #print("DEBUG Cocc (frag) is [%i,%i]" % (self.__Cocc.shape[0],self.__Cocc.shape[1]))
      if self.__tagname == 'A':
        #if self.__Cocc.shape[0] >self.nbf():
        #  Cocc_super[:,:self.ndocc()] = self.__Cocc
        #else:
          Cocc_super[:self.nbf(),:self.ndocc()] = self.__Cocc
          if self.__supermol:
            Cocc_super[:,self.ndocc():] = Cmat
          else:    
            Cocc_super[self.nbf():,self.ndocc():] = Cmat
      elif self.__tagname == 'B':
        if self.__supermol:
          Cocc_super[:,-self.ndocc():] = self.__Cocc
          Cocc_super[:,:-self.ndocc()] = Cmat
        else: 
          Cocc_super[-self.nbf():,-self.ndocc():] = self.__Cocc
          Cocc_super[:-self.nbf(),:-self.ndocc()] = Cmat
      
      return Cocc_super
 

  def set_eps(self,epsA):

      self.__eps=epsA

  def molecule(self):
      return self.__tagname

  def Da(self):
      nbf = self.__nbf
      Cocc = np.array(self.Ca_subset('OCC'))
      dens = np.matmul(Cocc,Cocc.T)
      return dens
  
  def eigvals(self):

      return self.__eps

  def Amat(self):
  
     return self.__ortho

  def S(self):
      if self.__tagname == 'A':
        ovap_sub = self.__S[:self.__nbf,:self.__nbf]
      elif self.__tagname == 'B':
        ovap_sub = self.__S[-self.__nbf:,-self.__nbf:]
      return ovap_sub
  def full_ovapm(self):
      return self.__S
  #def H(self):
  #
  #    return self.__Honeel
  def whoIam(self):
      return self.__tagname

  def basis(self):

      return self.__bset

  def ndocc(self):
      return self.__ndocc

  def nbf(self):
      if (self.__bset_mono.nbf() != self.__nbf):
         raise Exception("Inconsistent fragment basis dimension")
      return self.__nbf

  def finalize(self):
      if self.__accel == 'imag_time':
        if self.__supermol:
            ovap = self.full_ovapm()
        else:
            ovap = self.S()
        Fock = self.Femb()    

        try :
           eigval,C = scipy.linalg.eigh(Fock, ovap)   

        except scipy.linalg.LinAlgError:
           print("finalize(); Error in linal.eigh")
        
        self.__eps = eigval
        self.set_Ca(C,'ALL')
        self.set_Ca(C,'OCC')
      ##             
      #self.__Ccoeff = None
      #self.__Cocc = None
  def func_name(self):
      return self.__funcname
############################################################################
def make_Huzinaga(F_sub,ovap_sub,Cocc):
   # Cocc of the 'embedding' fragment
   #check dimension consistency
   dimF = F_sub.shape[1]

   density = np.matmul(Cocc,Cocc.T)
   dimD = density.shape[0]

   dimS = ovap_sub.shape[0]
   if  (dimF != dimD) or (dimS != dimD):
       raise Exception("Wrong matrix shape.\n")
   tmp = np.matmul(F_sub,np.matmul(density,ovap_sub))
   projector = -1.*(tmp + tmp.T)
   #projector = -0.5*(tmp + tmp.T)# <- the 0.5 factor is already accounted in the density matrix (n/2 electrons)
   return projector
############################################################################
def twoel(Cocc,basis,jk_mono,funcname):
    nbf = basis.nbf()
    jk_mono.C_left_add(psi4.core.Matrix.from_array(Cocc))
    jk_mono.compute()
    jk_mono.C_clear()
    J = jk_mono.J()[0]
    dens = np.matmul(Cocc,Cocc.T)
    Jene = 2.00*np.trace( np.matmul(J,dens) )
      
    if funcname == 'hf':
      V = np.float_(-1.0)*np.array( jk_mono.K()[0] )
      Exc = np.trace( np.matmul(dens,V) )
    else: 
        # Build Vxc matrix 
        #D must be a psi4.core.Matrix object not a numpy.narray 
        
        restricted = True
        
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
           build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
           build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(funcname, restricted)[0] 
        sup.set_deriv(2)
        sup.allocate() 
        vname = "RV" 
        if not restricted: vname = "UV"
        potential=psi4.core.VBase.build(basis,sup,vname)
        Dm=psi4.core.Matrix.from_array(dens.real) 
        potential.initialize()
        potential.set_D([Dm])
        
        V=psi4.core.Matrix(nbf,nbf)
        potential.compute_V([V])
        potential.finalize()
        #compute the corresponding XC energy (low level)
        Exc= potential.quadrature_values()["FUNCTIONAL"]
        
        if sup.is_x_hybrid():
          #
          #raise Exception("Low level theory functional is Hybrid?\n")
          alpha = sup.x_alpha()
          K = np.array(jk_mono.K()[0])
          Exc += -alpha*np.trace(np.matmul(dens,K))
          V.add(psi4.core.Matrix.from_array(-alpha*K))
      
    G =   np.float_(2.0)*J + np.array(V)
    ene = Jene +Exc
    return G, ene
############################################################################
# Fock associated to Csup, and system-dependent appropriate block selection
# the projector is added up
# occ_boundary bounds the number of columns in C_sup  corresponding to 'A' frag
# Csup has a 'fixed' structure Cocc[AB] = Cocc[A] (+) Cocc[B]
def Fock_emb(nbf,nbf_tot,occ_num,Hcore,ovap,Csup,sup_basis,jk,ftype,frag_id):


    Dmat = np.matmul(Csup,Csup.T)

    jk.C_left_add(psi4.core.Matrix.from_array(Csup))
    jk.compute()
    jk.C_clear()
    #set the Fock corresponding to 'super'  basis set
    Fock_tmp = Hcore + np.float_(2.0)*jk.J()[0]
    if ftype == 'hf':

        Fock_tmp -= np.array(jk.K()[0])
        # get the energy
        ene = np.matmul( (Hcore+Fock_tmp),Dmat)
        ene = np.trace(ene)
        if frag_id == 'A':
            fock = Fock_tmp[:nbf,:nbf]

            # get slice of each term % debug
            #subHcore = Hcore[:nbf,:nbf]
        
            #J=np.array(jk.J()[0])[:nbf,:nbf] #frag 'A' basis set is in the left upper corner (J_AA)
            #K=np.array(jk.K()[0])[:nbf,:nbf] #frag 'A' : K_AA
        elif frag_id == 'B':
            fock = Fock_tmp[-nbf:,-nbf:]
             
            # get slice of each term % debug
            #subHcore = Hcore[-nbf:,-nbf:]
            #J=np.array(jk.J()[0])[-nbf:,-nbf:] #frag 'B' basis set is in the right bottom corner (J_BB)
            #K=np.array(jk.K()[0])[-nbf:,-nbf:] #frag 'B' : K_BB
        else:
            print("check fragment labels")

        #put the terms together
        #fock = subHcore + np.float_(2.0)*J -K
    else:
        # Build Vxc matrix 
        #D must be a psi4.core.Matrix object not a numpy.narray 
        
        restricted = True
        
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
           build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
           build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(ftype, restricted)[0] 
        sup.set_deriv(2)
        sup.allocate() 
        vname = "RV" 
        if not restricted: vname = "UV"
        
        potential=psi4.core.VBase.build(sup_basis,sup,vname)
        Dm=psi4.core.Matrix.from_array(Dmat.real) 
        potential.initialize()
        potential.set_D([Dm])
        V=psi4.core.Matrix(nbf_tot,nbf_tot)
        potential.compute_V([V])
        potential.finalize()
        #compute the corresponding XC energy (low level)
        Exc= potential.quadrature_values()["FUNCTIONAL"]
        
        if sup.is_x_hybrid():
          #
          #raise Exception("Low level theory functional is Hybrid?\n")
          alpha = sup.x_alpha()
          K = np.array(jk.K()[0])
          V.add(psi4.core.Matrix.from_array(-alpha*K))
          #contribution to the XC energy
          Exc += -alpha*np.trace(np.matmul(Dmat,K))
        # Fock calculated on the 'super' basis set
        Fock_tmp += np.array(V)
        ene = np.matmul(Dmat,Hcore)
        ene = 2.0*np.trace(ene)
        ene += 2.0*np.trace(np.matmul(Dmat,np.array(jk.J()[0])) )
        ene =+ Exc

        if frag_id == 'A':
            fock = Fock_tmp[:nbf,:nbf]
            #subHcore = Hcore[:nbf,:nbf]
            #J=np.array(jk.J()[0])[:nbf,:nbf] #frag 'A' basis set is in the left upper corner (J_AA)
            #V = np.asarray(V)[:nbf,:nbf] 
        elif  frag_id == 'B':
            fock = Fock_tmp[-nbf:,-nbf:]
            #subHcore = Hcore[-nbf:,-nbf:]
            #J=np.array(jk.J()[0])[-nbf:,-nbf:] #frag 'B' basis set is in the right bottom corner (J_BB)
            #V = np.asarray(V)[-nbf:,-nbf:] 
        else:
            print("check fragment labels")
        # sum up
        #fock = subHcore + (2.00*J + V)

    
    # make the projector
    if nbf == nbf_tot: # the case of a supermolecul setting
       if frag_id == 'A':
           D_ext = np.matmul(Csup[:,occ_num:],Csup[:,occ_num:].T)
       
       elif  frag_id == 'B':
           D_ext = np.matmul(Csup[:,:-occ_num],Csup[:,:-occ_num].T)
       
       if D_ext.shape[0] != nbf_tot:
           raise Exception("Check dimension of D_ext (in Huzinaga)")
       projector = -1.*( np.matmul(Fock_tmp,np.matmul(D_ext,ovap)) + np.matmul(ovap,np.matmul(D_ext,Fock_tmp)) ) 
    else:    
       if frag_id == 'A':
           Fock_sub = Fock_tmp[:nbf,nbf:]
           ovap_sub = ovap[:nbf,nbf:].T
           #Cocc_ext is the  sublock of Cocc_sup  representing the orbitals of the frozen fragment
           Cocc_ext = Csup[nbf:,occ_num:]
       elif  frag_id == 'B':
           Fock_sub = Fock_tmp[-nbf:,:-nbf]
           ovap_sub = ovap[-nbf:,:-nbf].T
           Cocc_ext = Csup[:-nbf,:-occ_num]
       projector = make_Huzinaga(Fock_sub,ovap_sub,Cocc_ext)

    return fock + projector, ene
############################################################################
# helper class
class F_builder():
    def __init__(self,scf_common):
        #unpack data for Fock evaluation
        # the number of basis function can be defined from basis.nbf()
        self.__nbf = scf_common['nbf']               
        self.__nbf_tot = scf_common['nbf_tot']
        self.__fragocc = scf_common['occ_num']
        self.__Honel = scf_common['Hcore']
        self.__ovap = scf_common['ovap']
        self.__sup_bas = scf_common['sup_basis']
        self.__mono_bas = scf_common['mono_basis']
        self.__jk = scf_common['jk']
        self.__jk_frag = scf_common['jk_mono']
        self.__funcname = scf_common['ftype']
        self.__frag_name = scf_common['frag_id']
    def get_Fock(self,Csup,return_ene=False):
        fock,ene= Fock_emb(self.__nbf,self.__nbf_tot,self.__frag_occ,self.__Honel,self.__ovap,Csup,\
                         self.__sup_bas,self.__jk,self.__funcname,self.__fragname)
        if return_ene :
            return fock,ene
        else:
            return fock
############################################################################
class list_baseclass():
    ###
    # Cocc : occupied MO coeffs.
    # scf_common : common data strucure for SCF calculation
    # list_opts = [scheme,maxvec,kind]; >> scheme : value not checked, we assume 'LIST' <<
    #    kind : flavor of LIST (type=str) 
    #    maxvec: number of vectors for the scf accelerator (type=int)
    #    ! 'kind : better' corresponds to a transposed (LIST) B matrix from the 'direct' case
    ###
    def __init__(self,Cocc,scf_common,list_opts,debug):

        self.__e_list = None
        self.__Dmat   = None
        self.__D_m     = None  # Density matrix extrapolated
        self.__Fock_init = None # the intial Fock
        self.__Cocc   = Cocc 
        self.__Csup   = None # the Csup matrix
        self.__Dold   = None
        self.__kind   = list_opts[2]
        self.__debug = debug

        if self.__kind == 'indirect':
            self.__e_list = LIST_help.LISTi( int(list_opts[1]) ) 
        elif (self.__kind == 'direct') or (self.__kind == 'better'):
            self.__e_list = LIST_help.LISTd( int(list_opts[1]) )
        else:
            raise TypeError("Invalid Keyword")
        #definition of scf_common = [
        #nbf,nbf_tot,occ_num,Hcore,ovap,sup_basis,jk,ftype,frag_id]
        
        #unpack data for Fock evaluation
        # the number of basis function can be defined from basis.nbf()
        self.__nbf = scf_common['nbf']               
        self.__nbf_tot = scf_common['nbf_tot']
        self.__fragocc = scf_common['occ_num']
        self.__Honel = scf_common['Hcore']
        self.__ovap = scf_common['ovap']
        self.__sup_bas = scf_common['sup_basis']
        self.__mono_bas = scf_common['mono_basis']
        self.__jk = scf_common['jk']
        self.__jk_frag = scf_common['jk_mono']
        self.__funcname = scf_common['ftype']
        self.__frag_name = scf_common['frag_id']



    def set_Csup(self,Cmat):
        if not isinstance(Cmat,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__Csup = Cmat

    def add_Fock(self,Fmat):
        if not isinstance(Fmat,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        counter = self.__e_list.list_count()
        if counter < 2:
           self.__Fock_init = Fmat


    def extrapolate(self):
        # set the transpose option: solve  B^T c = d instead of B c = d 
        transpose = False
        if self.__kind == 'better':
            traspose = True
        counter = self.__e_list.list_count()
        
        D_actual = np.matmul(self.__Cocc,self.__Cocc.T)
        # store for later use
        self.__Dold = D_actual
       #this is to prevent extrapolation till the list vector has 2 elements
        if counter < 2:
          D_m = D_actual
          Fock = self.__Fock_init # only for the iteration = 1
        else:
          # D_m is obtained as linear combination of previous density matrices
          Fock,D_m = self.__e_list.extrapolate(transpose)
        self.__D_m = D_m
        return Fock, D_m
           
    def diagonalize(self,Fock,Amat,ndocc):
    
        # Diagonalize Fock matrix
        Fp = Amat.dot(Fock).dot(Amat)
        e, C2 = np.linalg.eigh(Fp)
        C = Amat.dot(C2)
        self.__Cocc = C[:, :ndocc]
        return C, C[:, :ndocc],e

    

    def finalize(self,Fock_in):
        # Note : in SCF DIIS procedure, error vector should only be computed using <non-extrapolated> quantities
        # in LIST methods the input Fock is the extrapolated one
        #the output density; local variable
        Dout =np.matmul(self.__Cocc,self.__Cocc.T)
        ### get Fock corresponding to the output
        # get  Fock_out
        
        frag_id = self.__frag_name
        Fock_out, dum = Fock_emb(self.__nbf,self.__nbf_tot,self.__fragocc,self.__Honel,\
                                 self.__ovap,self.__Csup,self.__sup_bas,self.__jk,self.__funcname,frag_id)
        # update Fock_init, to be used to estimate the diis_error in thie next iteration
        # the out Fock of iteration 'i' is the input Fock for iteraion 'i+1'
        self.__Fock_init = Fock_out

        Gmat,twoelene = twoel(self.__Cocc,self.__mono_bas,self.__jk_frag,self.__funcname)
        # out
        Hcore = Fock_out - Gmat
        SCF_E_out = 2.0*np.trace(np.matmul(Dout,Hcore)) + twoelene
        if self.__debug:
           print("Max diff (F_out -F_ex) : %.8f" % np.max( Fock_out-Fock_in))
    
        #calculate the corrected Hohenberg-Kohn-Sham functional (cHKS)

        if self.__kind == 'list_i':
           Delta_F =(Fock_out-Fock_in)*0.5 #two different formulations, see papers
        else:
           Delta_F =(Fock_out-Fock_in)

        
        cHKS_e = SCF_E_out + np.trace(np.matmul(self.__D_m-Dout,Delta_F))
        #
        if self.__kind == 'indirect':
            self.__e_list.diff_D(Dout-self.__Dold)
            self.__e_list.diff_Fock( (Fock_out-Fock_in)*0.5 )
        if (self.__kind ==  'direct') or (self.__kind ==  'better') :
           self.__e_list.E_out(SCF_E_out)
           self.__e_list.diff_Fock( (Fock_out -Fock_in) )

        self.__e_list.Fock_out(Fock_out)
        self.__e_list.D_out(Dout)


      
        #return Fock_out, cHKS_e

############################################################################
class itime_base():
    def __init__(self,Vminus,Cocc,acc_opts,debug,fout=sys.stderr):
        self.__Fock = None
        self.__Cocc = None
        self.__C_midb = None
        self.__debug = debug
        self.__stdout = fout
        #self.__maxiter = int(acc_opts[2])
        self.__lstep = -1.0j*np.float_(acc_opts[1]) # the only required parameter
        self.__Vminus = Vminus # the ndim x ndim  transformation matrix: AO-> prop basis -> TODO: use Vminus=S^-0.5
        self.__Vplus = None # the inverse matrix of Vminus

        try:
            self.__Vplus = np.linalg.inv(Vminus)  # still usefull
        except np.linalg.LinAlgError:
            print("Error in numpy.linalg.inv in itime_base")

        self.__Cocc = np.matmul(self.__Vplus,Cocc) #initially served on the MO basis. MDS : Why is it needed?
        
    def add_F(self,Fmat): 
        self.__Fock=Fmat
    def compute(self):
        fock_ti = self.__Fock
        C_new,C_midf = it_util.prop_mmut(fock_ti,self.__Cocc,self.__C_midb,self.__lstep,self.__Vminus,fout=sys.stderr)

        #update 
        self.__C_midb = C_midf
        self.__Cocc = C_new
    def Cocc(self,basis='AO'):
        if basis == 'prop':
            tmp = np.matmul(self.__Vplus,self.__Cocc) #served on the MO basis
            return tmp
        else:
            return self.__Cocc
    def Vminus(self):
        return self.__Vminus
