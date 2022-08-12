import os
import sys
import re
from scipy.linalg import fractional_matrix_power
#sys.path.append(os.environ['PSIPATH'])
from pkg_resources import parse_version
import argparse


def run(fgeomA, fgeomB, scf_type, numpy_mem, func_l, maxit, e_conv, d_conv, acc_param, bset, bsetAA,\
                          bsetBB, obs1, obs2, puream, embmol, wfn, isoA, isoB, debug):


    nb_tot = bset.nbf()
    ndoccA = isoA.nalpha()
    Cocc_AA = isoA.Ca_subset('AO','OCC')
    print("Cocc_AA check: %s" % np.allclose(Cocc_AA,isoA.Ca().np[:,:ndoccA]))
    D_AA = isoA.Da()
    C_AA = np.array(isoA.Ca())

    Cocc_BB = isoB.Ca_subset('AO','OCC')
    D_BB = isoB.Da()
    ndoccB = isoB.nalpha()
    C_BB = np.array(isoB.Ca())
   
    substr = acc_param.split(";")
    if len(substr) <3:
        raise Exception("Wrong SCF_OPTS input.\n")
    acc_opts = [substr[0],int(substr[1]),substr[2]]
    
    
    nAA=bsetAA.nbf() #the number of basis funcs of subsys A



    #mints object for bset_super
    print("functions in frag A: %i" % nAA)
    mints = psi4.core.MintsHelper(bset)

    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())

    # Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141

    Hcore = T+V

    #the ovap matrix for the {A} U {B} basis set
    S = np.array(mints.ao_overlap())
   
    
    ndocc = wfn.nalpha()
    ref_energy = wfn.energy()


    #DFT_in_DFT block
    # Set tolerances
    #maxiter = 12
    E_conv = e_conv
    D_conv = d_conv
    #
    # Setup data for DIIS
    t = time.time()
    E = 0.0
    Enuc = embmol.nuclear_repulsion_energy()
    Eold = 0.0
    #Fock_list = []
    #DIIS_error = []

    ##initialize Cooc_sup
    Cocc_sup = np.zeros((nb_tot,ndocc))
    ##assemble Cocc_sup
    Cocc_sup[:nAA,:ndoccA] = Cocc_AA
    Cocc_sup[nAA:,ndoccA:] = Cocc_BB
    
    if debug:
       print("check Cocc_AA:  %s" % np.allclose(Cocc_sup[:nAA,:ndoccA],Cocc_AA))
       print("check Cocc_BB:  %s" % np.allclose(Cocc_sup[nAA:,ndoccA:],Cocc_BB))
       print("check Cocc_AB = 0:  %s" % np.allclose(Cocc_sup[:nAA,ndoccA:],np.zeros((nAA,ndoccB))) )
       print("check Cocc_BA = 0:  %s" % np.allclose(Cocc_sup[nAA:,:ndoccA],np.zeros((bsetBB.nbf(),ndoccA))) )
       test_ortho = np.matmul(Cocc_sup.T,np.matmul(S,Cocc_sup))
       print("Cocc_sup are orthonormal ? %s" % np.allclose(test_ortho,np.eye(test_ortho.shape[0])))
    
    Dmat_sup = np.matmul(Cocc_sup,Cocc_sup.T)

    Dold = np.zeros_like(Dmat_sup)
    if args.debug:
        traceDS = np.trace(np.matmul(Dmat_sup,S))
        print("Check trace of initial guess total density: ... %.8f \n" % traceDS)
    #cube_util.denstocube(embmol,bset,Dmat_sup,S,ndocc,"denstot_mono_check",L,Dx)
    
    #populate sub A base object
    zA = args.chargeA

    ########################################
    # class RHF_embedding_base():
    #
    #   def __init__(self,nb_frag,ndocc_frag,nb_super,\
    #                                 ndocc_super,funcname,tag)
    ########################################
    subA = RHF_embedding_base(bsetAA.nbf(), ndoccA, nb_tot, ndocc, func_l, 'A')
    subA.initialize(S, bsetAA, bset, Hcore, C_AA,acc_opts,scf_type)
    print("I am frag '%s' : initializing\n" % subA.whoIam())
    ######################

    #populate sub B base object
    subB = RHF_embedding_base(bsetBB.nbf(), ndoccB, nb_tot, ndocc, func_l, 'B')
    subB.initialize(S, bsetBB, bset, Hcore, C_BB,acc_opts,scf_type)
    print("I am frag '%s' : initializing\n" % subB.whoIam())

    

    superdict ={'thaw' : subA, 'frozn' : subB}
    
    
    jobrun = FntFactory(debug)

    #test
    print("testing section ...")
    jobrun.initialize(superdict)
    jobrun.thaw_active()
    jobrun.clean()
    print("end testing")
    debug_out = open("debug.out", "w")
    MAXITER = maxit
    t = time.time()
     
    
    for FnT_ITER in range(1, MAXITER + 1):

            jobrun.initialize(superdict)
            E_step , dRMS = jobrun.thaw_active() 
            print('FnT Iteration %3d: Frag = %s Energy = %4.12f  dRMS = % 1.5E %s'
                      % (FnT_ITER,superdict['thaw'].whoIam(),E_step ,dRMS,superdict['thaw'].acc_scheme()   ))
            if debug:
               debug_out.write('FnT Iteration %3d: Frag = %s Energy(Im) = %.16e\n' % (FnT_ITER,superdict['thaw'].whoIam(),E_step.imag) )
            #swap dictionary values
            superdict.update({'thaw': superdict['frozn'],'frozn': superdict['thaw']})

            #if FnT_ITER == MAXITER:
            #    psi4.core.clean()
            #    raise Exception("Maximum number of FnT cycles exceeded.\n")
    debug_out.close()    
    ###################################################################################
    #print('Total time for FnT iterations: %.3f seconds \n\n' % (time.time() - t))
    #print("FnT iterations : %i" % FnT_ITER)

    if superdict['thaw'].whoIam() == 'B':   
       superdict.update({'thaw': superdict['frozn'],'frozn': superdict['thaw']})
    Cocc_sup = superdict['thaw'].Cocc_sum( superdict['frozn'].Ca_subset('OCC') )
    Dmat_sup = np.matmul(Cocc_sup,Cocc_sup.T)
    #get jk [ supermolecular]
    jk = subA.jk_super()
    Fock_sup,Jene,Exc = bo_helper.get_AOFock_JK(Dmat_sup,Cocc_sup,Hcore,jk,func_l,bset)
    FH = np.matmul(Dmat_sup,Hcore)
    SCF_E_test = 2.0*np.trace(FH) + Jene + Exc +Enuc


    
    # fragment A orbital energies
    print('Orbital Energies [Eh] for frag. "A"\n')

    print('Doubly Occupied:\n')
    for k in range(ndoccA):
         print('%iA : %.6f' %(k+1,np.asarray(subA.eigvals())[k]))

    print('Virtual:\n')
    for k in range(ndoccA,nAA):
         print('%iA : %.6f'% (k+1,np.asarray(subA.eigvals())[k]))
    print()
    
    # fragment B orbital energies
    print('Orbital Energies [Eh] for frag. "B"\n')

    print('Doubly Occupied:\n')
    for k in range(ndoccB):
         print('%iA : %.6f' %(k+1,np.asarray(subB.eigvals())[k]))

    print('Virtual:\n')
    for k in range(ndoccB,(nb_tot-nAA)):
         print('%iA : %.6f'% (k+1,np.asarray(subB.eigvals())[k]))

    print()

    print('Energy/Eh = %4.16f  dE = % 1.5E LOG10(|dE|) = % 1.5f'
            % (SCF_E_test, SCF_E_test-ref_energy, np.log10(abs(SCF_E_test-ref_energy)) ))
    
    #print("One electron energy : %.8f\n" % (2.0*np.trace(FH)))
    #print("Coulomb energy :  %.8f\n" % Jene)
    #print("DFT XC energy :  %.8f\n" % Exc)

    psi4.core.clean()
    #refresh the molecule object
    psi4.set_options({'cubeprop_tasks': ['density','orbitals'],
                      'cubeprop_orbitals': [1,2,3], # just some  orbitals
                      'CUBIC_GRID_OVERAGE' : [4.5,4.5,4.5],
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'CUBIC_GRID_SPACING' : [0.2,0.2,0.2]})
    
    isoA.Ca().copy(psi4.core.Matrix.from_array(subA.Ca_subset('ALL') ))
    isoA.Cb().copy(psi4.core.Matrix.from_array(subA.Ca_subset('ALL') ))

    isoA.Da().copy(psi4.core.Matrix.from_array(subA.Da() ))
    isoA.Db().copy(psi4.core.Matrix.from_array(subA.Da() ))

    isoA.Fa().copy(psi4.core.Matrix.from_array(subA.Femb()) )
    isoA.Fb().copy(psi4.core.Matrix.from_array(subA.Femb()) ) 
 
    # quick test
    _F = subA.Femb()
    _C = subA.Ca_subset('ALL')
    rtest = np.isrealobj(_F)
    print("Fock (emb) is real : %s" % rtest)
    test = np.matmul(_C.T,np.matmul(_F,_C))
    diag_test = np.diagflat(np.diagonal(test))
    print("Fock is diagonal : %s" % np.allclose(test,diag_test,atol=1.0e-8))
    return subA, subB,SCF_E_test, isoA
############################################################################

if __name__ == "__main__":

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-gA","--geomA", help="Specify geometry file for the subsystem A", required=True, 
            type=str, default="XYZ")
    parser.add_argument("-gB","--geomB", help="Specify geometry file for the subsystem B", required=True, 
            type=str, default="XYZ")
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
            default=False, action="store_true")

    parser.add_argument("-o1","--obs1", help="Specify the orbital basis set for subsys A", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("-o2","--obs2", help="Specify the general orbital basis set", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("--puream", help="Pure AM basis option on", required=False,
            default=False, action="store_true")
    parser.add_argument("--df_guess", help="Density-fitted pre-scf on", required=False,
            default=False, action="store_true")
    parser.add_argument("--scf_type", help="Specify the scf type: direct or df (for now)", required=False, 
            type=str, default='direct')
    parser.add_argument("--cc_type", help="Specify the cc type: conventional ['conv'], density-fitted [df], cholesky-decomp. [cd]", required=False, 
            type=str, default=None)
    parser.add_argument("--cc_maxit", help="Max number of iterations for cc module (default : 20)", required=False,
            default=20, type = int)
    parser.add_argument("--cc_outfile", help="Specify the name of the outfile for CC calculation", required=False,
            default="cc_out.txt", type = str)
    parser.add_argument("-f","--func", help="Specify the low level theory functional", required=False, 
            type=str, default="blyp")
    parser.add_argument("--e_conv", help="Convergence energy threshold",required=False,
            default=1.0e-7, type = float)
    parser.add_argument("--d_conv", help="Convergence density threshold",
            default=1.0e-6, type = float)
    parser.add_argument("--scf_opts", help="Select SCF acceleration options : accel_scheme;maxvec;type (default : diis;6; None )", required=False,
            default="diis;6; None", type=str)
    # accel_scheme : diis|list, max_vec = N, type : direct|indirect|better|None
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)

    parser.add_argument("-z", "--charge", help="Charge of the whole system",
            default=0, type = int)
    parser.add_argument("-zA", "--chargeA", help="Charge of Frag. A",
            default=0, type = int)
    parser.add_argument("-zB", "--chargeB", help="Charge of Frag. B",
            default=0, type = int)
    parser.add_argument("-mA", "--multA", help="Multiplicity (2S+1) of Frag. A",
            default=1, type = int)
    parser.add_argument("-mB", "--multB", help="Multiplicity of Frag. B",
            default=1, type = int)
    parser.add_argument("--maxit", help="Max number of iterations (default : 20)", required=False,
            default=4, type = int)
    parser.add_argument("--mod_path", help="Specify path of common modules", required=False, 
            type=str, default="/home/matteo/projected_emb/common")

    
    args = parser.parse_args()
    # ene_vanilla is the energy of the genuine supermolecular calculation

    modpaths = args.mod_path
    if modpaths is not None :
        for path in modpaths.split(";"):
            sys.path.append(path)

    import time
    import scipy.linalg
    import psi4
    import numpy as np
    import helper_HF
    from util import Molecule
    from init import initialize
    from base_embedding import RHF_embedding_base
    from embed_util import FntFactory
    
    import util
    import bo_helper

    bset,bsetAA,bsetBB,supermol,wfnAB,ene_vanilla,isoA,isoB = initialize(args.scf_type,args.df_guess,args.obs1,\
                                                    args.obs2,args.puream,args.geomA,args.geomB,args.func,args.cc_type,args.cc_maxit,args.e_conv,args.d_conv,\
                                                    args.charge,args.chargeA,args.multA,args.chargeB,args.multB)
   
    # main function ,update isoA
    subA, subB, FnT_ene, isoA = run(args.geomA, args.geomB, args.scf_type,\
                             args.numpy_mem, args.func, args.maxit, args.e_conv, args.d_conv, args.scf_opts,\
                             bset, bsetAA, bsetBB, args.obs1, args.obs2, args.puream, supermol, wfnAB, isoA, isoB, args.debug)
    import ccsd_native
    import ccsd_conv
    if args.cc_type is not None :
        print("Starting ccsd calculation")
        ene_wf_dft = ccsd_native.wfn_in_dft(subA,subB,isoA,args.cc_outfile,ene_vanilla,FnT_ene,wf_type='ccsd')

        print ("Energy : WF-in-%s energy : % 4.12f" %  (args.func.upper(),ene_wf_dft) )
        # hardcoded ccsd 
        #ccsd_conv.wfn_in_dft(subA,subB,isoA,ene_vanilla,FnT_ene,args.numpy_mem)
