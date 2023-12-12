import psi4
import argparse
import os
import sys
import re
import time
import numpy as np

sys.path.insert(0, "../common")
modpaths = os.environ.get('RTUTIL_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from util import Molecule

# scf_type, df_guess, basis_spec, geom_file, func, e(d)_convergence : arguments psi4.set_options()
# acc_param : determines the type of scf acceleration scheme

def get_ene_wfn(func_name,molecule,return_wfn=False,print_cube=False):

    t = time.process_time()
    
    if not isinstance(func_name,str):
         result = psi4.energy('scf',dft_functional=func_name, \
                            molecule=molecule, return_wfn=return_wfn)
    else:
         result = psi4.energy(func_name, molecule=molecule, return_wfn=return_wfn)

    t2 = time.process_time()     
    print('Total time for psi4.energy() : %.3f seconds \n\n' % (t2-t))

    if print_cube:
        # check result contain the wfn
        if len(result) <2:
            raise Exception("return_wfn should be set True\n")
        psi4.cubeprop(result[1])
    
    if return_wfn:
        return result[0],result[1]
    else:
        return result

def initialize(scf_type, df_guess, basis_spec, puream, geom_file, func, acc_param, e_conv, d_conv, use_lv,\
                                       cc_type=None, cc_maxiter=100, debug = False, supermol=False, core_guess=False,\
                                       df_basis_scf='def2-universal-jkfit', df_basis_cc='def2-universal-jkfit'):
    if use_lv:
        print("Using P = mu*[SDS]\n")
    # check compatibility scf_type && cc_type
    # cc_type = 'conv' requires scf_type = 'direct'
    # cc_type = 'cd/df' requires scf_type = 'cd/[mem_df;disk_df;df]) accordingly
    print("scf_type : %s\n" % scf_type)
    if cc_type is not None:
        cc_flag = True
        # check
        if cc_type not  in scf_type:
            if not (scf_type.upper() == 'DIRECT' and (cc_type == 'conv' or cc_type == 'fno') ):
                raise TypeError('check cc_type and scf_type')
        
    else:
        cc_flag = False




    fgeom = geom_file
    func_l = func
    #func_h=args.func1
    #print("High Level functional : %s\n" % func_h)
    #print("Low Level functional : %s\n" % func_l)
    print(".. using basis .. : %s\n" % basis_spec )


    moltot = Molecule(fgeom,label=False)

    psi4.set_memory('4 GB')




    #append some options
    moltot.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    
    if debug:
        moltot.display_xyz()


    totsys_mol = psi4.geometry(moltot.geometry())
    totsys_mol.print_out()


    
    L=[4.5,4.5,4.5]
    Dx=[0.2,0.2,0.2]

    #if func == 'hf':
    #    reference = 'rhf'
    #else:
    #    reference = 'rks'

    psi4.core.set_output_file('single_ppsi4.out', False)
    psi4.set_options({
                      'puream': puream,
                      'DF_SCF_GUESS': df_guess,
                      'scf_type': scf_type,
                     # 'reference' : reference,
                      'df_basis_scf': df_basis_scf,
                      'df_ints_io' : 'save',       # ?
                      'dft_radial_scheme' : 'becke',
                       #'dft_radial_points': 80,
                      'dft_spherical_points' : 434,
                      'cubeprop_tasks': ['density'],
                      'e_convergence': 1.0e-8,
                      'd_convergence': 1.0e-8})

    # -->USE basis_helper <--
    #as argument e.g: basis_spec = "sto-3g; H1:def2-svp; H2:3-21G"   # dictionary like string
    # manipulate the string with the basis
    basis_sub_str = basis_spec.split(";")
    basis_str = "assign " + str(basis_sub_str[0])+"\n"

    if len(basis_sub_str) >1:
       for elm in basis_sub_str[1:]:
           tmp= elm.split(":")
           basis_str += "assign " + str(tmp[0]) + " " +str(tmp[1]) +"\n"

    #https://github.com/psi4/psi4/blob/master/samples/python/mints9/input.py
    psi4.basis_helper(basis_str,
                     name='mybas')

    job_nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
    psi4.set_num_threads(job_nthreads)

    #if cc_flag:
    #    psi4.set_options({'cc_type': cc_type,
    #                    'cachelevel': 0,
    #                    'df_basis_cc' : df_basis_cc,
    #                    'df_ints_io' : 'save',
    #                    'freeze_core' : False,
    #                    'mp2_type': 'conv',
    #                    'MP2_AMPS_PRINT' : True,
    #                     'maxiter' : cc_maxiter})

    #build a basis set object
    basis_mol = psi4.core.BasisSet.build(totsys_mol, 'ORBITAL',\
            psi4.core.get_global_option('basis')) # or set the basis from input
    #if not cc_flag:
    #    psi4.core.clean()
    num_frag = totsys_mol.nfragments()
    
    if num_frag <2:
        raise Exception("no fragments defined\n")

    # a list of tuples
    frag_list = totsys_mol.get_fragments()
    print("Frag list\n")
    print(frag_list)

    frag_mol = [] # a container for the molecular fragment objets
    wfn_list =[] #container for wfn objects
    
    ### user-defined funtional ###
    if func == 'svwn5': 
       func_dict = {
            "name": "SVWN5",
             "x_functionals": {
                "LDA_X": {1.}
            },
            "c_functionals": {
                "LDA_C_VWN": {1.}
            }
       }
       func = func_dict

    ###
    totsys_mol.activate_all_fragments()
    for idx in range(1,num_frag+1):
      frag_mol.append(totsys_mol.extract_subsets(idx))
      ene_idx,wfn_idx = get_ene_wfn(func, molecule=frag_mol[idx-1], return_wfn=True)
      wfn_list.append(wfn_idx)
    
    frags_container = None
    #get mints, ovapm, T,V
    
    # basis_mol
    mints = psi4.core.MintsHelper(basis_mol)
    ovapm = np.array(mints.ao_overlap())
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    H = T+V

    #consistency check
    ndocc_super = 0 # get the total number of doubly occupied MO in mol
    for num,wfn in enumerate(wfn_list):
        ndocc_super +=wfn.nalpha()
    print("doubly occ in mol(super) %i\n" % ndocc_super)

    from base_embedding import RHF_embedding_base
    # acc_param

    substr = acc_param.split(";")
    if len(substr) !=3:
       raise Exception("Wrong SCF_OPTS input.\n")
    acc_opts = [substr[0],float(substr[1]),substr[2]]

    # test jk 
    from base_embedding import get_JK
    frag_container = []
    #start for block
    #import pdb; pdb.set_trace()
    for num,wfn in enumerate(wfn_list):
        #RHF_embedding_base: __init__(..)
        tmp = RHF_embedding_base(totsys_mol,wfn.nalpha(),ndocc_super, func,\
                                   num+1, supermol=supermol,flag_lv=use_lv)
        #-> initialize(..)
        tmp.initialize(ovapm,basis_mol,wfn.basisset(), H, np.array(wfn.Ca()),\
                   acc_opts,target=scf_type,debug=debug,muval=1.0e3)
        # TEST
        #jk_frag = get_JK(scf_type,tmp.mol(),wfn.basisset())
        #tmp.set_jk(jk_frag)

    
        # temp container
        frag_container.append(tmp)
    # end for block
        
    if num_frag != len(frag_container):
        raise Exception("check fragment number\n")
    
    if core_guess:
    
        Cocc_container = frag_container[0].Cocc_gather(frag_container[1:])
    
        for elm in frag_container:
           elm.core_guess(Cocc_container)

    frag_act = frag_container[0]
    fnt_list = [frag_act,frag_container[1:].copy()] 
   
    return fnt_list,totsys_mol,wfn_list
############################################################################

if __name__ == "__main__":

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-g","--geom", help="Specify geometry, with '--' separated fragments, charges and mult.", required=True, 
            type=str, default="XYZ")
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
            default=False, action="store_true")

    parser.add_argument("-o1","--obs1", help="Specify the orbital basis using dictinary-like syntax", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("--puream", help="Pure AM basis option on", required=False,
            default=False, action="store_true")
    parser.add_argument("--e_conv", help="Convergence energy threshold",required=False,
            default=1.0e-7, type = float)
    parser.add_argument("--d_conv", help="Convergence density threshold",
            default=1.0e-6, type = float)

    parser.add_argument("-f","--func", help="Specify the low level theory functional", required=False, 
            type=str, default="blyp")

    parser.add_argument("--scf_type", help="Specify the scf type: direct or df (for now)", required=False, 
            type=str, default='DIRECT')

    parser.add_argument("--df_guess", help="Set True SCF_DF_GUESS in ground state SCF", required=False, 
            type=bool, default=False)
    parser.add_argument("--scf_opts", help="Select SCF acceleration options : accel_scheme;maxvec;type (default : diis;6; None )", required=False,
            default="_diis;6; None", type=str)

    
    args = parser.parse_args()
    frags_container, psi4mol, wfnlist = initialize(args.scf_type, args.df_guess, args.obs1, args.puream,\
                                                        args.geom, args.func, args.scf_opts, args.e_conv,args.d_conv,use_lv=False,debug=True)

    # test Fock,ovap,density block extraction
    # garther the occupied MO (in a composite matrix)
    frozen_frag = []
    frag_1 = frags_container[0]
    print("I am frag : %i\n" % frag_1.whoIam())
    for el in frags_container[1]:
       frozen_frag.append(el)
    Csup = frag_1.Cocc_gather(frozen_frag)
    Dsup = np.matmul(Csup,Csup.T)
    #proj will correspond to the operator to project-out frag(frozen) density
    Fock,proj = frag_1.get_Fock(Csup,full_Fock=True)
    ovap_full = frag_1.full_ovapm()
    ovap_1 = frag_1.S()
    # principal subblock
    ovap_d0 = frag_1.extract_subb(ovap=(ovap_full,'d0'))[0]
    print("Check ovap subblock : SUCCESS ... %s\n" % np.allclose(ovap_d0,ovap_1))
