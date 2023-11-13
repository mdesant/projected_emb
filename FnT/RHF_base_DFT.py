import os
import sys
import re
from scipy.linalg import fractional_matrix_power
#sys.path.append(os.environ['PSIPATH'])

import argparse
import copy
import numpy as np 
from base_embedding import RHF_embedding_base

class results:
    frag : RHF_embedding_base
    Da : np.ndarray = None 
    Fa : np.ndarray = None 
    Ca : np.ndarray = None
    Hcore : np.ndarray  = None
    twoel_ene : float
    energy : float
    Enuc : float
    functional : None

def run(frag_container, e_conv, maxit, debug, loewdin=False,frag_id=1):

    #DFT_in_DFT block
    # Set tolerances
    #maxiter = 12
    E_conv = e_conv
    #
    # Setup data for DIIS
    t = time.time()
    E = 0.0
    Eold = 0.0
    #Fock_list = []
    #DIIS_error = []

    from embed_util import FntFactory
    #######################################################################
    # class FntFactory():
    #     def __init__(self,debug=False,loewdin=False,outfile=sys.stderr)
    #######################################################################
    
    
    debug_out = open("debug.out", "w")
    
    jobrun = FntFactory(debug,loewdin,debug_out)

    jobrun.initialize(frag_container)

    #test
    print("testing section ...\n")
    acc_type = frag_container[0].acc_scheme()
    print("Using %s\n" % acc_type )
    print("Status :")
    jobrun.status()

    #exit()
    print("Full basis : %s\n" % jobrun.is_full_basis())

    MAXITER = maxit
    t = time.time()
     
    Eold_sup = 0.0 
    D_conv = e_conv 
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@ Energy label refers to the the fragment being thawed                                                     @")
    print("@ dRMS refers to the root mean square of the DIIS error of the fragment being thawed                       @")
    print("@ dE refers to the deviation of the actual genuine supermolecular energy with respect to the previous step @\n")
    for FnT_ITER in range(1, MAXITER + 1):

            E_step, E_sup, dRMS = jobrun.thaw_active() 
            
            print('FnT Iteration %3d: Frag = %s Energy = %4.12f  dE(sup) = % 1.5E  dRMS = % 1.5E %s'
                      % (FnT_ITER,jobrun.thawed_id(), E_step.real, (E_sup-Eold_sup), dRMS, jobrun.actual_boost() ))
            
            if debug:
               debug_out.write('FnT Iteration %3d: Frag = %s Energy(Im) = %.16e\n' % (FnT_ITER,jobrun.thawed_id(),E_step.imag) )
            
            jobrun.finalize()  # the role active-frozen is assigned here
            
            if abs(E_sup-Eold_sup) < E_conv and dRMS < D_conv:
              break
            Eold_sup = E_sup

            if FnT_ITER == MAXITER:
                print("Maximum number of FnT cycles reached.\n")
    debug_out.close()    
    ###################################################################################
    print()
    print('Total time for FnT iterations: %.3f seconds \n\n' % (time.time() - t))
    print("FnT iterations : %i" % FnT_ITER)
    print()

    print("Final Status :")
    jobrun.status()
    # get final eigenvalues for each fragment and identify fragment 1
    #print epsilon_a using a loop

    final_container = frag_container[1].copy()
    final_container.insert(0,frag_container[0])

    #identify fragment 1
    #set from input #TODO

    #frag_id = 1 default arg, acceptable values [1,2..]
    if  len(final_container) < frag_id <1:
        raise ValueError("frag_id must lie between 0 and max_num_frag\n")
    selected_id = 0
    other_frag = []
    for num_idx,frag in enumerate(final_container):
        if frag.whoIam() == frag_id:
           selected_id = num_idx
        else: # gather the other fragments
           other_frag.append(frag)
    #define the supermolecule C_occ

    frag_1 = final_container[selected_id]
    res_gather = frag_1.Cocc_gather(other_frag)

    #check if the MOs are orthonormal (in general this happen only in the super-basis setting)
    #full_ovap = frag_container[0].full_ovapm()
    #res =  np.matmul(Cocc_sup.T,np.matmul(full_ovap,Cocc_sup))
    #print("Final occupied MOs are orthogonal: %s" % np.allclose(np.eye(Cocc_sup.shape[1]),res) )
    
    # the regular scf_energy goes here TODO
    # just for the selected frag[1] basis
    fock_frag_1,proj_1,SCF_E_test=frag_1.get_Fock(res_gather,return_ene=True)
    # get also the two electron part
    
    # STEP for post-hf run
    twoel_mtx, twoel_ene = frag_1.G()
    # the effective Hamiltonian (core) of the selected fragment being embedded in the
    # other  fragments potential
    Hcore_1 = fock_frag_1 - twoel_mtx + proj_1
    Da_frag_1 = frag_1.Da()
    
    jk = frag_1.get_jk()
    jk.C_left_add(psi4.core.Matrix.from_array(frag_1.Ca_subset('OCC')) )
    jk.compute()
    jk.C_clear()
    
    J_mtx = np.asarray( jk.J()[0] )
    K_mtx = np.asarray( jk.K()[0] )
    # we have to add J-K (based on the density/orbitals of frag_1)
    Fock_ref =  Hcore_1 +2.0*J_mtx -K_mtx
    
    # from Fock_ref
    ene_ref = np.trace(np.matmul(Fock_ref+Hcore_1,Da_frag_1))+frag_1.mol().nuclear_repulsion_energy()
    print("REF energy (ref Fock: hcore_a_in_b +2J-k) : %.8e\n" % ene_ref)
     
    print("CHECK: Fock[emb] is (%i x %i)\n" % fock_frag_1.shape)
    jk.finalize()
    
    # for imaginary-time propagation orb. energies are not available
    if frag_1.acc_scheme() != 'imag_time':  
       for frag in final_container: 
           epsilon_a = np.array(frag.eigvals())
           # fragment ith orbital energies
           print('Orbital Energies [Eh] for frag. %i\n' % frag.whoIam() )
     
           print('Doubly Occupied:\n')
           for k in range(frag.ndocc() ):
               print('%iA : %.6f' %(k+1,epsilon_a[k]))
     
           print('Virtual:\n')
           for k in range(frag.ndocc(),frag.nbf()):
               print('%iA : %.6f'% (k+1,epsilon_a[k]))
           print()    
           print("HOMO-LUMO gap\n")
           gap = -epsilon_a[frag.ndocc()-1] + epsilon_a[frag.ndocc()] 
           print("%.5e Eh / %.5e eV\n" % (gap,gap*27.2114) )

    # MOVE    
    #print("One electron energy : %.8f\n" % (2.0*np.trace(FH)))
    #print("Coulomb energy :  %.8f\n" % Jene)
    #print("DFT XC energy :  %.8f\n" % Exc)

    #psi4.core.clean()
    #refresh the molecule object
    psi4.set_options({'cubeprop_tasks': ['density','orbitals'],
                      'cubeprop_orbitals': [1,2,3], # just some  orbitals
                      'CUBIC_GRID_OVERAGE' : [4.5,4.5,4.5],
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'CUBIC_GRID_SPACING' : [0.2,0.2,0.2]})
    
    #TODO
    #dataclass
    res_container = results
    res_container.frag = frag_1
    res_container.Ca = frag_1.Ca_subset('ALL')
    res_container.Fa = Fock_ref
    res_container.energy = ene_ref
    res_container.Hcore = Hcore_1
    res_container.Da = frag_1.Da()
    res_container.twoel_ene = twoel_ene
    res_container.Enuc = frag_1.mol().nuclear_repulsion_energy()
    res_container.functional = frag_1.func_name()

    # quick test
    #_F = subA.Femb()
    #rtest = np.isrealobj(_F)
    #print("Fock (emb) is real : %s" % rtest)

    #if (acc_opts[0] != 'imag_time'):
    #   _C = subA.Ca_subset('ALL')
    #   test = np.matmul(_C.T,np.matmul(_F,_C))
    #   diag_test = np.diagflat(np.diagonal(test))
    #   print("Fock is diagonal : %s" % np.allclose(test,diag_test,atol=1.0e-8))
   
    return E_sup,SCF_E_test,res_container
############################################################################

if __name__ == "__main__":

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-g","--geom", help="Specify geometry file, including xyz, charge, and mult. of fragments,", required=True, 
            type=str, default="XYZ")
    parser.add_argument("-s", "--supermol", help="Do the projection in the supermolecular basis", required=False,
            default=False, action="store_true")
   
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
            default=False, action="store_true")

    parser.add_argument("-o1","--obs1", help="Specify the orbital basis set for subsys A", required=False, 
            type=str, default="6-31G*")

    parser.add_argument("--core", help="Use modified-core guess orbital", required=False,
            default=False, action="store_true")

    # density fitting
    parser.add_argument("--df_basis", help="Specify the aux basis set for SCF", required=False, 
            type=str, default="cc-pvdz-jkfit")
    parser.add_argument("--df_basis_cc", help="Specify the aux basis set for CC", required=False, 
            type=str, default="cc-pvdz-jkfit")

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
    parser.add_argument("--wf_frag", help="Select the fragment for post-hf", required=False,
            default=1, type = int)

    parser.add_argument("--e_conv", help="Convergence energy threshold",required=False,
            default=1.0e-7, type = float)
    parser.add_argument("--d_conv", help="Convergence density threshold",
            default=1.0e-6, type = float)
    parser.add_argument("--scf_opts", help="Select SCF acceleration options : accel_scheme;maxvec;type (default : diis;6; None )", required=False,
            default="diis;6; None", type=str)
    # accel_scheme : diis|list, max_vec = N, type : direct|indirect|better|None
    
    parser.add_argument("--loewdin", help="Activate Intermediate Loewdin orthogonalization (on Cocc[sup]->Fock) ", required=False,
            default=False, action="store_true")
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)

    parser.add_argument("--maxit", help="Max number of iterations (default : 20)", required=False,
            default=20, type = int)
    parser.add_argument("--mod_path", help="Specify path of common modules", required=False, 
            type=str, default="/home/matteo/projected_emb/common")
    parser.add_argument("--cc_native", help="Use cc native psi4 implementation", required=False,
            default=False, action="store_true")
    parser.add_argument("--lv", help="Use level-shift operator instead of Huzinaga", required=False,
            default=False, action="store_true")

    
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
    #from util import Molecule
    from init import initialize, get_ene_wfn
    #from base_embedding import RHF_embedding_base
    #from embed_util import FntFactory
    
    import util
    import bo_helper

    # wfn_list contains the wfn (psi4) object for each frag
    frags_container, psi4mol, wfn_list = initialize(args.scf_type, args.df_guess, args.obs1, args.puream,\
                                                        args.geom, args.func, args.scf_opts, args.e_conv,\
                                                        args.d_conv, args.lv, args.cc_type,args.cc_maxit, args.debug,\
                                                        args.supermol, args.core, args.df_basis,args.df_basis_cc)
    
    #exit() 
    Enuc = psi4mol.nuclear_repulsion_energy()
    # main function ,update isoA
    Esup, scf_e_test, res_container= run(frags_container, args.e_conv, args.maxit, args.debug, args.loewdin,args.wf_frag)

    # do a genuine psi4 calculation on the super-system
    func = res_container.functional
    print("energy calculation of the super-molecule at:\n")
    if not isinstance(func,str):
       print("@@@ "+func["name"].upper()+" @@@")
    else:
       print("@@@ "+func.upper()+" @@@")
    ene_ref_sup =  get_ene_wfn(func,molecule=psi4mol)
    ene_diff = abs(ene_ref_sup - (scf_e_test+Enuc))

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print()
    print("Final results of embedding calculation:\n")
    print(" - DFT-in-DFT energy (eq 10 doi:10.1021/acs.jctc.7b00034)\n")
    print(" - dE : E[super] - E[dft-in-dft]\n")
    print(" - LOG10(|dE|)\n")
    print()
    print('Energy/Eh = %4.16f  dE = % 1.5E LOG10(|dE|) = % 1.5f'
            % (scf_e_test+Enuc, ene_diff, np.log10(abs(ene_diff)) ))

    # TODO: use a molecule with ghosted frozen fragment if the supermolecule (-basis) calculation is performed
    if args.supermol:
        ghost_frags = [x+1 for x in range(len(wfn_list))]
        ghost_frags.pop(args.wf_frag-1)
        psi4mol.set_ghost_fragments(ghost_frags)
        ene_x,wfn_cc=get_ene_wfn(func, molecule=psi4mol, return_wfn=True)
    else:
        wfn_cc = wfn_list[args.wf_frag-1]
    import ccsd_native
    import ccsd_conv

    if args.cc_type is not None :
     
        print("TEST: get ref energy using unmodified Hcore\n")
        basis_frag_1=wfn_cc.basisset()
        mints_frag_1 = psi4.core.MintsHelper(basis_frag_1)
        H_unmod = np.array(mints_frag_1.ao_kinetic()) + np.array(mints_frag_1.ao_potential())
        Fock_0 =  res_container.Fa -res_container.Hcore +H_unmod
        ene_test = np.trace(np.matmul(Fock_0+H_unmod,res_container.Da)) + res_container.Enuc
    
        print("REF energy (ref Fock: H0 +2J-k) : %.8e\n" % ene_test)
        # This reference energy (from umodified Hcore) would pop-up in cc output
        # if Hcore_modified (A_in_B) is not copied in the H() variable of wfn_cc
        # despite not affecting the correlation energy, since we copy-in the "effective"
        # Fock (Hcore_A_in_B +2J-K) and Da, Ca coefficients.
        
        #copy into wfn_cc
        wfn_cc.Da().copy(psi4.core.Matrix.from_array(res_container.Da))
        wfn_cc.Fa().copy(psi4.core.Matrix.from_array(res_container.Fa))
        wfn_cc.H().copy(psi4.core.Matrix.from_array(res_container.Hcore))
        wfn_cc.Ca().copy(psi4.core.Matrix.from_array(res_container.Ca))
        wfn_cc.Db().copy(psi4.core.Matrix.from_array(res_container.Da))
        wfn_cc.Fb().copy(psi4.core.Matrix.from_array(res_container.Fa))
        wfn_cc.Cb().copy(psi4.core.Matrix.from_array(res_container.Ca))
        wfn_cc.set_energy(res_container.energy) 
        
        # RHF reference also if func != HF
        ref_wfn_type = 'rhf'

        print("Reference wfn(CC) : %s\n" % ref_wfn_type)

        psi4.set_options({'cc_type': args.cc_type,
                        'cachelevel': 0,
                        'reference': ref_wfn_type,
                        'df_basis_cc' : args.df_basis_cc,
                        'freeze_core' : False,
                        'mp2_type': 'conv',
                        'MP2_AMPS_PRINT' : True,
                         'maxiter' : args.cc_maxit})


        print("Starting ccsd calculation")
    #%%% wfn_in_dft(frag_cc,wfn_cc,cc_outfile,ene_sup,FnT_ene,wf_type='ccsd') %%%%
        if args.cc_native:
           ene_wf_dft = ccsd_native.wfn_in_dft(res_container,wfn_cc,args.cc_outfile,ene_ref_sup,scf_e_test,wf_type='ccsd')
        else:
           ene_wf_dft = ccsd_conv.wfn_in_dft(res_container,wfn_cc,ene_ref_sup,scf_e_test,args.numpy_mem)
        print ("Energy : WF-in-%s energy : % 4.12f" %  (args.func.upper(),ene_wf_dft) )
        # hardcoded ccsd 
