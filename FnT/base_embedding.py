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
modpaths = os.environ.get('RTUTIL_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

import it_util
import numpy as np
import psi4
#deprecated
#from pkg_resources import parse_version
from packaging import version
from helper_HF import DIIS_helper
from ediis_helper import EDIIS_helper
import LIST_help
import scipy
from scipy.linalg import fractional_matrix_power

from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')
@dataclass
class fock_common:
    nbf: int 
    nbf_tot: int 
    docc_num  : int   # the number of doubly occuopied MO of the fragment
    Hcore : np.ndarray 
    ovap  : np.ndarray 
    mono_basis : psi4.core.BasisSet
    sup_basis  : psi4.core.BasisSet
    jk      : psi4.core.JK
    jk_mono : psi4.core.JK
    func_name : str = "blyp"
    frag_id   : int = 1

    
def get_geom_frag(frag_id,fragment_list):
    if not isinstance(fragment_list,list):
        raise TypeError("Check frag_list\n")
    atoms_id = fragment_list[frag_id-1] # frag_id >=1; atoms_id is a tuple (init_at,final_at)
    if not isinstance(atoms_id,tuple):
        raise TypeError("Check atoms' id\n")
    natom = atoms_id[1]-atoms_id[0]
    #print("n. atoms : %i\n" % natom)
    #the atom centers in the the total molecule
    atom_centers = [int(m) for m in range(atoms_id[0],atoms_id[1])] 
    #print(atom_centers)
    return atom_centers

def get_JK(target,psi_mol,basis_object):
      if target == 'DF' or target == 'MEM_DF' or target == 'DISK_DF':
          print("DEBUG->target: %s\n" % target)
          auxb = psi4.core.BasisSet.build(psi_mol, "DF_BASIS_SCF", "",\
                                          fitrole="RIFIT",other=basis_object.name())
          #import pdb; pdb.set_trace();
          jk_factory = psi4.core.JK.build(basis_object,auxb)
          mem_val = jk_factory.memory_estimate()
      else:
          #print("DEBUG->target: %s\n" % target)
          jk_factory = psi4.core.JK.build(basis_object)
          mem_val = jk_factory.memory_estimate()
 
      jk_factory.set_do_wK(False)
      jk_factory.set_memory(mem_val)
      jk_factory.initialize()
      jk_factory.print_header()
      return jk_factory 
##################################################################
# acc_opts = [kind, maxvec, variant=None]
#    kind : 'diss' | 'list_(i/d)' (type=str)
#    maxvec
#    variant : Use the transposed 'B' matrix? (type=bool)
class RHF_embedding_base():

  def __init__(self,molobj, ndocc_frag,\
                    ndocc_super,funcname,id_frag,fout=sys.stderr,supermol=False,flag_lv=False):
      # molobj is the molecule (psi4.core.Molecule) corresponding to the total system
      self.__eps = None
      self.__Cocc = None
      self.__Ccoeff = None
      self.__C_midb = None      # aux MO coeff at previous midpont for imaginary time prop
      self.__frag_id = None
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
      self.__ovap_mono = None   # 'fragX' basis function overlap , X=A,B
      self.__ortho = None       # Ortho = S_xx^{-0.5} x= A,B

      self.__nbf = None
      self.__nb_super = None

      self.__ndocc = ndocc_frag
      self.__ndocc_super = ndocc_super
      self.__funcname = funcname

      self.__frag_id = id_frag
      
      self.__stdout = fout
      self.__supermol = supermol
      self.__sup_mol = molobj # the 'super' molecule object 
      self.__frag_list = molobj.get_fragments()
      self.__sup_mol.activate_all_fragments()
      self.__frag_mol = molobj.extract_subsets(id_frag) # the fragment molecule extracted
                                                        # from the 'super'
      #check natom of the fragment and point group
      print("frag %i contains %i atoms\n" % (self.__frag_id,self.__frag_mol.natom()) )
      print(".. belongs to point group %s\n" % self.__frag_mol.point_group().full_name())
      self.__frag_mask = None
      self.__frag_notmask = None
      self.__limit = None
      self.__fake_limit = None
      self.__fake_mask = None
      self.__e_scf = None

      #for lv_shift projector
      self.__muval = None
      self.__do_lv = flag_lv
      
      self.__scf_iter = 1
###########################################
  def mol(self,supmol=False):
      if supmol:
          return self.__sup_mol
      else:
          return self.__frag_mol
  def set_jk(self,jk_factory):
      self.__jk_mono = jk_factory
  def set_e_scf(self,ene):
      self.__e_scf = ene
  def e_scf(self):
      return self.__e_scf
  def  niter(self):
      return self.__scf_iter
###########################################
  def initialize(self,ovap_sup,basis_sup,basis_mono,\
                     Hsup,Ccoeff,acc_opts,target='direct',debug=False,muval=1.0e6):
      self.__debug = debug
      self.__muval = muval
      # assign
      self.__bset_sup =   basis_sup

      if self.__supermol:
         self.__bset_mono =  basis_sup
      else:
         self.__bset_mono =  basis_mono


      self.__nbf = self.__bset_mono.nbf()
      self.__nb_super = self.__bset_sup.nbf()
      
      # needed only if the fock has to be evaluated in place
      scf_common = fock_common

      scf_common.nbf     = self.__nbf
      scf_common.nbf_tot = self.__nb_super
      scf_common.docc_num = self.__ndocc  # the number of doubly occuopied MO
                                          # of the fragment
      scf_common.Hcore    = Hsup
      scf_common.ovap     = ovap_sup 
      scf_common.ovap_mono = None
      scf_common.mono_basis = self.__bset_mono
      scf_common.sup_basis  = self.__bset_sup
      scf_common.func_name  = self.__funcname
      scf_common.frag_id    = self.__frag_id
      ############################################
      #assign
      self.__S = ovap_sup

      # call set_mask, for the general absolute-locazation case
      self.set_mask()

      if self.__supermol:
          self.__fake_limit = self.__limit
          self.__fake_mask  = self.__frag_mask
          
          # re-define:
          # self.__attr (attr=frag_mask ..) will point to a different memory location
          self.__frag_mask = [True for m in range(self.__nb_super)]
          self.__frag_notmask = [True for m in range(self.__nb_super)]
          self.__limit = (0,self.__nb_super-1)

      #get mask
      mask = self.__frag_mask
      
      ovap_mono = ovap_sup[mask,:][:,mask]
      self.__ovap_mono = ovap_mono
      #check
      if ovap_mono.shape[0] != self.__nbf:
         print("ovap_mono.dim[0] : %i, nbf : %i\n" % ( ovap_mono.shape[0],self.__nbf ) )
         raise Exception("Something went wrong with basis set dimension\n")
      
      scf_common.ovap_mono = ovap_mono
      
      self.__ortho = fractional_matrix_power(ovap_mono, -0.5)

      self.__Cocc =   np.array(Ccoeff)[:,:self.__ndocc]
      self.__Ccoeff = np.array(Ccoeff)

      target = target.upper() # JK target 
      # test debug
      #
      try:
        result_sup = get_JK(target,self.__sup_mol,basis_sup)
      except ValueError:
          raise

      result_sup.initialize()

      self.__jk_sup= result_sup
      if (self.__jk_sup is None):
         raise Exception("Error in JK instance")
      #
      if not self.__supermol  :
         # get JK on the mono-molecular basis
         result_mono = get_JK(target,self.__frag_mol,basis_mono)
         #result_mono.initialize()

         self.__jk_mono = result_mono
      else:
         self.__jk_mono = result_sup # it points to the same memory location of __jk_sup

      if (self.__jk_mono is None):
         raise Exception("Error in JK instance")
      
      print("JK of the frag(%i) is built on a basis counting %i funcs\n" \
                  % (self.whoIam(),self.__jk_mono.basisset().nbf() ))

      # H core
      if (Hsup.shape[0] != self.__nb_super):
         raise Exception("Wrong Hsup dim[0]")

      self.__Honeel = Hsup
      

      # set jk in scf_common
      scf_common.jk = self.__jk_sup
      scf_common.jk_mono = self.__jk_mono
 
      self.__acc_opts = acc_opts
 
      # acceleration engine name
      self.__accel = acc_opts[0]
      # set the acceleration method and initialize
      max_vec = int(acc_opts[1])
      
      # if the supermolecular basis is used , for the imaginary time  propagation 
      # we opt for the orthogonalized atomic basis , as propagation basis (?)
      if self.__supermol and (self.__accel == 'imag_time'): 
        print("here")
        Vminus =   self.__ortho 
      else:
        Vminus = Ccoeff
 
      # define the initial Cocc  used in list and imag_time if supermol=True
      if self.__supermol :
          Cocc_in = self.Ca_subset('OCC',Csup_format=True)
      else:
          Cocc_in = self.__Cocc 
      
      # set acceleration engines
      if '_diis' in self.__accel:
          #pure diis
          diis_engine = DIIS_helper(max_vec=6)
          self.__scfboost = diis_engine
          self.__scfboost = (None,diis_engine)
          if self.__accel == 'a_diis' or self.__accel == 'e_diis':
             self.__scfboost = (EDIIS_helper(max_vec=6,engine=self.__accel),diis_engine)

      elif self.__accel == 'list':
          #raise Exception("UNSAFE, to fixed \n")
          print("using list acceleration, untrusted\n")
      # list_baseclass ->    __init__(self,Cocc,scf_common,active_frag,list_opts,debug)
          self.__scfboost = list_baseclass(Cocc_in, scf_common, acc_opts, debug)

      elif 'imag_time' in self.__accel :
          ediis_engine = EDIIS_helper(max_vec=4,engine='a_diis')   
          self.__scfboost = (itime_base(Vminus,Cocc_in,acc_opts,debug),ediis_engine)   # D^{AO} = Vminus D^{orth} Vminus.T

      elif self.__accel == 'lv_shift':
          self.__scfboost = None

      else:
          raise ValueError("wrong keyword/not implemented\n")

  def fake_limits(self):
      if self.__fake_limit is None:
        l1 = self.__limit[0]
        l2 = self.__limit[1]
      else:
        l1 = self.__fake_limit[0]
        l2 = self.__fake_limit[1]
      return l1,l2  
###################################################
  def set_mask(self):
    basis_mol = self.__bset_sup  
    frag_list = self.__frag_list
    id_frag   = self.__frag_id

    frag_centers = get_geom_frag(id_frag,frag_list)
 
    # test methods of the basis object
    total_shell_frag = 0
 
    tmp_ishell_list = []
    nbf_tot = basis_mol.nbf()
    nbf_mask = [False for m in range(nbf_tot)]
 
    for catom in frag_centers:
    # do loop on atoms of the fragment
        nshell = basis_mol.nshell_on_center(catom)
        #print("center : %i has %i shells\n" % (catom,nshell))
 
        for shell_num in range(nshell):
 
            total_shell_frag +=1
 
            ishell_on_center = basis_mol.shell_on_center(catom,shell_num) #Return the 
                                                                     #i’th shell on center.??
            tmp_ishell_list.append(ishell_on_center)
 
    #end loop atoms of the fragment   


    max_num_ishell = max(tmp_ishell_list)
    min_num_ishell = min(tmp_ishell_list)

    func_to_shell = []

    for func_id  in range(nbf_tot):
        func_to_shell.append( basis_mol.function_to_shell(func_id) )

    for idx,el in enumerate(func_to_shell):
        if (min_num_ishell  <= el <= max_num_ishell):
           nbf_mask[idx] = True

    func_id_list = np.array([int(m) for m in range(nbf_tot)])
    func_id_list = func_id_list[nbf_mask]
    l1= func_id_list[0]
    l2 = func_id_list[-1]
    
    self.__frag_mask = nbf_mask
    self.__frag_notmask = [not x for x in nbf_mask]
    self.__limit = (l1,l2)
#####################################################################

  def core_guess(self,Cocc):

       # Cocc must be a tuple
       if not isinstance(Cocc,tuple):
           raise TypeError("Csup must be tuple\n")
       Csup = Cocc[1]
       #trace= np.matmul(self.__S,np.matmul(Csup,Csup.T))
       #print(np.trace(trace))
       
       Cocc = Cocc[0].copy() # make a copy so that the reordering will not affect other fragments
       check_ndocc = 0
       for mtx in Cocc:
           check_ndocc += mtx.shape[1]
       if check_ndocc != self.__ndocc_super:
           raise ValueError("wrong n. doubly occupied MO\n")

       frag_id = self.__frag_id
       #reorder
       Cocc_in = Cocc.pop(frag_id - 1)
       if self.__ndocc != Cocc_in.shape[1]:
           raise ValueError("wrong n. doubly occupied MO of the frag: %s\n" % frag_id)
       Cocc.insert(0,Cocc_in)

       # get the fock from Csup, return Fock,proj
       Fock = self.get_Fock((Cocc,Csup))

       # subtract the G[D] part to obtain h[A_in_env]_core
       G,dum = self.G()
       # hcore is Fock[sup] + proj - two_el_mtx[frag]
       hcore = Fock[0] +Fock[1] - G 
       
       #diagonalize hcore
       # Amat = ovap^-.5

       Amat = self.Amat()
       Fp = np.matmul(Amat.T,np.matmul(hcore,Amat))
       
       e, C2 = np.linalg.eigh(Fp)
       print('Orbital Energies (occ) [Eh] for frag. %i guess\n' % self.whoIam() )
       for k in range(self.__ndocc):
            print(e[k])
       C = Amat.dot(C2)
       self.set_Ca(C)
        
  #def set_tag(self,frag_iden):
  #    if not isinstance(frag_iden,int)
  #    self.__frag_id = frag_iden # a int type

  def diis(self):
      diis_on = False
      if '_diis'  in self.__accel:
          diis_on = True
      if not diis_on:
         raise Exception("wrong keyword, not supposed to use diis")
      if not isinstance(self.__scfboost,tuple):
         raise ValueError("check diis __scfboost\n")
      return self.__scfboost

  def imag_time(self):
      acc_type  = self.__accel
      if not 'imag_time' in acc_type:
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

  def acc_scheme(self,force_type=None):
      if force_type is not None:
          self.__accel = force_type
      res=self.__accel
      return res

  def get_jk(self,kind='mono'):
      if kind == 'mono':
         res = self.__jk_mono
      elif kind == 'super':
         res = self.__jk_sup
      else:
          raise ValueError("Invalid keyword\n")
      return res

  def get_Fock(self,Csup,return_ene=False,debug=False): #local debug
      # C_sup is a tuple
      if not isinstance(Csup,np.ndarray):
          raise TypeError("Csup must be a ndarray\n")
      #nbf = self.__nbf
      #nbf_tot = self.__nb_super
      Hcore = self.__Honeel
      ovap = self.__S   #contains the sup-basis overlap mtx
      basis = self.__bset_sup
      jk = self.__jk_sup
      ftype = self.__funcname
      frag_id = self.__frag_id

      fock,ene =  Fock_emb(Hcore,Csup,basis,jk,ftype,frag_id)#TODO
      #make the projector

      tmp = Csup[:,self.__ndocc:]  # the  (nbas x ndocc) slice (thawed fragment occ. MOs coeff) is piled-up on the frozen coeff

      
      # TEST
      sup_D_frozen = np.matmul(tmp,tmp.T)

      # get the slice of Fock and ovap and MO coeff
      mask = self.__frag_mask
      not_mask = self.__frag_notmask
      #slice tmp (the gathered frozen Occ MO)
      tmp = tmp[not_mask,:]
      frozn_D = np.matmul(tmp,tmp.T)
      #test the trace of frozn_D
      #get a suitable ovapm
      
      ## TEST ##
      lv_shift = self.__muval*np.matmul(ovap,np.matmul(sup_D_frozen,ovap))
      lv_shift = lv_shift[mask,:][:,mask] 
      
      # for debug
      #ovap_frozn = ovap[not_mask,:][:,not_mask]
      #trace_frzn = np.trace(np.matmul(ovap_frozn,frozn_D))
      #print("Tr[frozen_D S]: %.4e\n" % trace_frzn)
      #print("Frozen D mat has dim (%i,%i)\n" % frozn_D.shape)
      
      #the off-diagonal block of fock and overlap
      F_off = fock[mask,:][:,not_mask]
      ovap_off = ovap[mask,:][:,not_mask] # ovap is symm
      #print("F and S off-diagonal block have dim (%i,%i)\n" % (F_off.shape))

      if self.__do_lv:
        proj = lv_shift
      else:
        proj =  make_Huzinaga(F_off,ovap_off.T,tmp)

      if debug :
          proj = (proj,fock)

      if return_ene:
          return fock[mask,:][:,mask] ,proj, ene
      else:
       return fock[mask,:][:,mask],proj

  def G(self,replace_func=None):
    #check dimension consistency
    Cocc = self.__Cocc
    basis = self.__bset_mono
    if Cocc.shape[0] != basis.nbf():
        print("Cocc.dim[0] : %i, nbf : %i\n" %  (Cocc.shape[0], basis.nbf()) )
        # handle the exception
        nelm = basis.nbf()
        Cocc = np.zeros((nelm,Cocc.shape[1]),dtype=np.float_)
        try:        
            Cocc[self.__fake_limit[0]:self.__fake_limit[1]+1, :] = self.__Cocc
        except ValueError:
            raise

    if replace_func is not None:
      Gmat,ene = twoel(Cocc,basis,self.__jk_mono,replace_func)
    else:
      Gmat,ene = twoel(Cocc,basis,self.__jk_mono,self.__funcname)
    return Gmat, ene

      
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
          
  def Ca_subset(self,tag='ALL',Csup_format=False):#TODO 0
      if tag == 'OCC':
          tmp =self.__Cocc
      elif tag == 'VIRT':
          tmp = self.__Ccoeff[:,self.__ndocc:]
      elif tag == 'ALL':
          tmp = self.__Ccoeff

      res = tmp
      #print("Ca_subset has dim : %i,%i\n" % res.shape)
      
      if Csup_format : # 
          nbf =self.__nb_super
          if self.__frag_mask is None:
              raise Exception("frag. basis mask not intialized\n")
          tmp_mtx = np.zeros((nbf,tmp.shape[1]),dtype=np.float_)
          mask = self.__frag_mask
          l1 = self.__limit[0]
          l2 = self.__limit[1]

          #print("l1,l2 = (%i,%i)\n" % (l1,l2))
          if (l2-l1+1) !=self.__nbf:
              print(l2-l1+1) 
              raise Exception("check dimension\n")

          if res.shape[0] != (l2-l1+1) :                                            # use fake_mask and fake_limit
              tmp_mtx[self.__fake_limit[0]:self.__fake_limit[1]+1, :] = tmp.real    # this occurs in the first FnT 
                                                                                    # iteration of a supermolecule-basis setting (-s)
          else:
              #if self.__debug:
                  #print("Csup_format needed here\n")
              tmp_mtx[l1:l2+1, :] = tmp.real

          if self.__debug:
             if np.iscomplexobj(tmp):
                print("max |mtx.imag| %.4e\n" % np.max(np.abs(tmp.imag)) )
          res = tmp_mtx

      return res
  #TODO 1
  def Cocc_gather(self,frags_frozn,dest='OCC'):
      if not isinstance(frags_frozn,list):
          raise TypeError("check frag list\n")
      #print("Cmat dim : %i,%i\n" % (Cmat.shape)) 
      #nb_tot = self.__nb_super
      #ndocc_tot = self.__ndocc_super
      #Cocc_super = np.zeros( (nb_tot,ndocc_tot) )

      #print("DEBUG Cocc_super is [%i,%i]" % (Cocc_super.shape[0],Cocc_super.shape[1]))
      #print("DEBUG Cocc (frag) is [%i,%i]" % (self.__Cocc.shape[0],self.__Cocc.shape[1]))
      
      #set True
      Csup_shape = True
    
      tmp = []

      tmp.append(self.Ca_subset(dest,Csup_format=Csup_shape) )
      for elm in frags_frozn:
          mtx =elm.Ca_subset(dest,Csup_format=Csup_shape)
          tmp.append(mtx)
      res = np.concatenate(tmp,axis=1)
      return res
 

  def set_eps(self,epsA):

      self.__eps=np.asarray(epsA)

  #def molecule(self):
  #    return self.__tagname

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
      return self.__ovap_mono
  
  def full_ovapm(self):
      return self.__S
  
  def H(self):
  
      return self.__Honeel
  
  def whoIam(self):
      return self.__frag_id

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
        if  not isinstance(self.Femb(),np.ndarray):
            raise ValueError("Fock must be numpy.ndarray\n")
        Fock = np.asarray(self.Femb()) 


        #if not isinstance(self.__eps,np.ndarray):
        try :
           eigval,C = scipy.linalg.eigh(Fock, ovap)   
        
        except scipy.linalg.LinAlgError:
           print("finalize(); Error in linal.eigh")
        
        self.__eps = eigval
        self.set_Ca(C,'ALL')
        self.set_Ca(C,'OCC')
      ## clean ?             
      #self.__Ccoeff = None
      #self.__Cocc = None

      #increment scf_iter counter
      self.__scf_iter += 1
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
        
        if version.parse(psi4.__version__) >= version.parse('1.3a1'):
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
def Fock_emb(Hcore,Csup,sup_basis,jk,ftype,frag_id=1):#TODO
    
    if not isinstance(Csup,np.ndarray):
        raise TypeError("Csup must be numpy.ndarray\n")
    # check also dimension
    if Csup.shape[0] != sup_basis.nbf():
        print("Csup.dim[0] : %i, sup_bf: %i\n" % ( Csup.shape[0],sup_basis.nbf() ))
        raise Exception("wrong dimension of Csup\n")
    nbf_tot = sup_basis.nbf()
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

        #put the terms together
        #fock = subHcore + np.float_(2.0)*J -K
    else:
        # Build Vxc matrix 
        #D must be a psi4.core.Matrix object not a numpy.narray 
        
        restricted = True
        
        if version.parse(psi4.__version__) >= version.parse('1.3a1'):
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
        ene += Exc
        
        # sum up
        #fock = subHcore + (2.00*J + V)

    
    return Fock_tmp , ene
############################################################################
# helper class
class F_builder():
    def __init__(self,scf_common):
        #unpack data for Fock evaluation
        # the number of basis function can be defined from basis.nbf()
        self.__nbf = scf_common.nbf               
        self.__nbf_tot = scf_common.nbf_tot
        self.__fragocc = scf_common.docc_num
        self.__Honel = scf_common.Hcore
        self.__ovap = scf_common.ovap
        self.__sup_bas = scf_common.sup_basis
        self.__mono_bas = scf_common.mono_basis
        self.__jk = scf_common.jk
        self.__jk_frag = scf_common.jk_mono
        self.__funcname = scf_common.func_name
        #self.__frag_name = scf_common.frag_id
    def get_Fock(self,Csup_gather,Dmat=None,frag_id='A',return_ene=False):

        #input :nbf,
        #       nbf_tot,
        #       occ_num,
        #       Hcore,ovap,
        #       Csup,
        #       sup_basis,
        #       jk,ftype,
        #       frag_id
        
        if not isinstance(Csup_gather,np.ndarray):
            raise TypeError("Csup_gather must be a np.ndarray\n")
        
        if np.iscomplexobj(Csup) or not isinstance(Csup,np.ndarray):
                #diagonalize D.real
                if not isinstance(Dmat,np.ndarray):
                    raise TypeError("Dmat is not np.ndarray\n")
                tmp = np.matmul(self.__ovap,np.matmul(Dmat.real,self.__ovap))
                w,eigvec = scipy.linalg.eigh(tmp,self.__ovap)
                idx = w.argsort()[::-1]
                eigvec = (eigvec[:,idx])[:,:self.__fragocc]
                Csup_inp = eigvec

        else:
                Csup_inp =  Csup
        
        #def Fock_emb(Hcore,Csup,sup_basis,jk,ftype,frag_id=1):#TODO
        

        fock,ene= Fock_emb(self.__Honel, Csup_inp,\
                         self.__sup_bas, self.__jk, self.__funcname)
        proj = np.zeros_like(fock) # <- dummy,please define a regular projector
        if return_ene :
            return fock,proj,ene
        else:
            return fock,proj
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
        #print(("list_baseclass.__init__() -> Cocc: (%i,%i)\n" % self.__Cocc.shape))
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

    def set_Csup(self,Csup_gather): # the gathering has been done elsewhere
        if not isinstance(Csup_gather,np.ndarray):
                raise TypeError("input must be a numpy.ndarray")
        self.__Csup = Csup_gather

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
        #print("extrapolate: Cocc is (%i,%i) \n" % self.__Cocc.shape)
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
        self.__Cocc = C[:, :ndocc]  # update Mo occ
        return C, C[:, :ndocc],e

    

    def finalize(self,frag_act,Fock_in):
        # Note : in SCF DIIS procedure, error vector should only be computed using <non-extrapolated> quantities
        # in LIST methods the input Fock is the extrapolated one
        #the output density; local variable
        Dout =np.matmul(self.__Cocc,self.__Cocc.T)
        ### get Fock corresponding to the output
        # get  Fock_out
        
        Fock_out, proj_out = frag_act.get_Fock(self.__Csup)
        #check dimension
        if Fock_out.shape[0] != Fock_in.shape[0]:
            raise ValueError("check fock (in|out) in LiST\n")



        # update Fock_init, to be used to estimate the diis_error in thie next iteration
        # the out Fock of iteration 'i' is the input Fock for iteraion 'i+1'
        self.__Fock_init = Fock_out+proj_out

        Gmat,twoelene = frag_act.G()
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

        #debug 
        print(self.__D_m.shape,Dout.shape,Delta_F.shape)
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
        if not isinstance(Vminus,np.ndarray):
            raise ValueError("Vminus must be numpy.ndarray\n")
        try:
            self.__Vplus = np.linalg.inv(Vminus)  # still usefull
        except np.linalg.LinAlgError:
            print("Error in numpy.linalg.inv in itime_base")
        self.__Cocc = np.matmul(self.__Vplus,Cocc) #initially served on the MO basis. MDS : Why is it needed?
        
    def add_F(self,Fmat): 
        self.__Fock=Fmat
    def compute(self):
        fock_ti = self.__Fock
        if not isinstance(fock_ti,np.ndarray):
            raise ValueError("Fock(t) must be numpy.ndarray\n")
        if not isinstance(self.__Vminus,np.ndarray):
            raise ValueError("Vminus must be numpy.ndarray\n")
        #print("V matrix [%i,%i]\n" % self.__Vminus.shape)
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
