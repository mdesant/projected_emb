import sys
import os
import psi4
import numpy as np
sys.path.insert(0,'../common')
import it_util
from base_embedding import itime_base

# in get_Fock() we need to detect the psi4 version
from pkg_resources import parse_version
##################################################################

def get_Fock(D, Hcore, I, f_type, basisset):
    # Build J,K matrices
    J = np.einsum('pqrs,rs->pq', I, D)
    if (f_type=='hf'):
        K = np.einsum('prqs,rs->pq', I, D)
        F = Hcore + J*np.float_(2.0) - K
        Exc=0.0
        J_ene = 0.0
    else:
        #D must be a psi4.core.Matrix object not a numpy.narray
        restricted = True
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
                build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
                build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(f_type, restricted)[0]
        sup.set_deriv(2)
        sup.allocate()
        vname = "RV"
        if not restricted:
            vname = "UV"
        potential=psi4.core.VBase.build(basisset,sup,vname)
        Dm=psi4.core.Matrix.from_array(D.real)
        potential.initialize()
        potential.set_D([Dm])
        nbf=D.shape[0]
        V=psi4.core.Matrix(nbf,nbf)
        potential.compute_V([V])
        potential.finalize()
        F = Hcore + J*np.float_(2.0) +V
        Exc= potential.quadrature_values()["FUNCTIONAL"]
        if sup.is_x_hybrid():
          alpha = sup.x_alpha()
          K = np.einsum('prqs,rs->pq', I, D)
          F += -alpha*K
          Exc += -alpha*np.trace(np.matmul(D,K))
        J_ene=2.00*np.trace(np.matmul(D,J))
    return J_ene,Exc,F
##################################################################
import argparse
####################################
# parse arguments from std input
####################################
parser = argparse.ArgumentParser()

parser.add_argument("--func", help="Specify the functional", required=False,
        default="hf", type=str)
parser.add_argument("--basis", help="Specify the orbital basis set", required=False,
        default="def2-svp", type=str)
parser.add_argument("--tstep", help="Step parameter of i-time evolution",required=False,
        default=0.5, type = float)

parser.add_argument("-l", "--loewdin", help="Set the orthogonalized AO basis (Loewdin) as prop. basis", required=False,
        default=False, action="store_true")
parser.add_argument("--maxit", help="Max number of iterations",required=False,
        default=30, type = int)
args = parser.parse_args()
# test for water molecule

mol = psi4.geometry("""
 O -1.4626 0.0000 0.0000 
 H -1.7312 0.9302 0.0000 
 H -0.4844 0.0275 0.0000 
 symmetry c1
 no_com
 no_reorient
 """)
psi4.set_options({'basis': args.basis,
                      'puream': 'False',
                      'scf_type': 'direct',
                      'df_scf_guess' : 'False',
                      'dft_radial_scheme' : 'becke',
                       #'dft_radial_points': 80,
                      'dft_spherical_points' : 434,
                      #'cubeprop_tasks': ['orbitals'],
                      #'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
mol_wfn= psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('basis'))
mol_wfn.basisset().print_detail_out()

ndocc=mol_wfn.nalpha()
nbeta=mol_wfn.nbeta()
if (ndocc!=nbeta) :
    print('not close-shell')
#initialize mints object
mints = psi4.core.MintsHelper(mol_wfn.basisset())
S=np.array(mints.ao_overlap())
#get T,V,dipole_z
T=np.array(mints.ao_kinetic())
V=np.array(mints.ao_potential())
H=T+V
dipole=mints.ao_dipole()
dip_mat=np.copy(dipole[2])
#internal defined functional 
svwn5_func = {
    "name": "SVWN5",
    "x_functionals": {
        "LDA_X": {}
    },
    "c_functionals": {
        "LDA_C_VWN": {}
    }
}

func = args.func
# the basisset object
basisset=mol_wfn.basisset()
nbf = basisset.nbf()
import scipy.linalg

eigval,eigvec =scipy.linalg.eigh(H,S)
np.savetxt("eigval.out", eigval)
Cocc = eigvec[:,:ndocc]

Dmat = np.matmul(Cocc,Cocc.T)


niter = args.maxit
Lambda = -1.0j*args.tstep
C_midb = None
C_new = Cocc
Amat = mints.ao_overlap()
Amat.power(-0.5, 1.e-16)
Amat = np.asarray(Amat)
#set a different transformation matrix (eigvec = S^{-1/2})
#Vmat can be set equale to eigvec : Vmat=C(0), or Vmat = S^{-1/2}

if args.loewdin:
   Vmat = Amat
else:
   Vmat = eigvec
# Run a quick check to make sure everything will fit into memory
numpy_memory = 2
I_Size = (nbf**4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
#Get Eri (2-electron repulsion integrals)
I=np.array(mints.ao_eri())
ene_list = []
Eold = 0.
run_itime = itime_base(Vmat,C_new,(-99,args.tstep),True) 
fo = open("it_err.txt","w")
for k in range(1,niter+1):
    #get the fock
    Jene,Exc, fock = get_Fock(Dmat, H, I, func, basisset)
    diis_e = np.einsum('ij,jk,kl->il', fock, Dmat, S) - np.einsum('ij,jk,kl->il', S, Dmat, fock)
    diis_e = Amat.dot(diis_e).dot(Amat)
    dRMS = np.mean(diis_e**2)**0.5
    if func == 'hf':
      ene = np.trace( np.matmul(H+fock,Dmat))
    else: 
      ene =  2.0*np.trace( np.matmul(H,Dmat))+Jene+Exc
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E' % (k, ene, (ene - Eold), dRMS))
    ene_list.append(abs(ene-Eold))
    Eold = ene
    Dold =Dmat
    #fock setup in itime_base instance
    run_itime.add_F(fock)
    run_itime.compute() # and internal var ara updated
    C_new = run_itime.Cocc()
    Dmat = np.matmul(C_new,C_new.T)
    diff = Dmat-Dold
    test_diff = np.linalg.norm(diff,'fro')
    print("||D-Dold||_2 : %.5e\n" % test_diff)
fo.close()
iter_counter = np.linspace(1,niter,niter)   
np.savetxt('ene.txt', np.c_[iter_counter.real,np.array(ene_list).real], fmt='%.12e')

ref_ene = psi4.energy(func)

Nuc_rep = mol.nuclear_repulsion_energy()
print("Ene diff: %.5e\n" % (ene.real+Nuc_rep-ref_ene))
