import psi4
import argparse
import os
import sys
import re
sys.path.insert(0, "../common")
from util import Molecule

def initialize(scf_type,obs1,obs2,puream,geomA,geomB,func,cc_type,cc_maxiter,e_conv,d_conv,\
                charge,chargeA,multA,chargeB,multB):
    
    # check compatibility scf_type && cc_type
    # cc_type = 'conv' requires scf_type = 'direct'
    # cc_type = 'cd/df' requires scf_type = 'cd/[mem_df;disk_df;df]) accordingly
    if cc_type is not None:
        cc_flag = True
        # check
        if cc_type not  in scf_type:
            if not (scf_type == 'direct' and cc_type == 'conv'):
                raise TypeError('check cc_type and scf_type')
        
    else:
        cc_flag = False

    acc_bset = obs1
    gen_bset = obs2



    fgeomA = geomA
    fgeomB = geomB
    func_l = func
    #func_h=args.func1
    #print("High Level functional : %s\n" % func_h)
    #print("Low Level functional : %s\n" % func_l)
    print("Low Level basis : %s\n" % gen_bset)
    print("High Level basis : %s\n" % acc_bset)


    moltot = Molecule(fgeomA,label=True)
    moltot.set_charge(charge)

    psi4.set_memory('2 GB')

    speclist = moltot.labels()


    molB = Molecule(fgeomB)
    #moltot has been augmented by molecule B geometry
    moltot.append(molB.geometry())

    #append some options, we can also include total charge (q) and multiplicity (S) of the total system
    moltot.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    moltot.display_xyz()

    molA = Molecule(fgeomA)
    molA.set_charge(chargeA,multA)
    molB.set_charge(chargeB,multB)

    molA.display_xyz()
    molB.display_xyz()

    molobj=psi4.geometry(moltot.geometry())
    molobj.print_out()


    psi4.core.IO.set_default_namespace("molobj")
    def basisspec_psi4_yo__anonymous775(mol,role):
            mol.set_basis_all_atoms(gen_bset, role=role)
            for k in speclist:
              mol.set_basis_by_label(k, acc_bset,role=role)
            return {}


    #the basis set object for the complex: a composite basis set for rt applications
    psi4.qcdb.libmintsbasisset.basishorde['USERDEFINED'] = basisspec_psi4_yo__anonymous775

    L=[4.5,4.5,4.5]
    Dx=[0.2,0.2,0.2]

    psi4.set_options({'basis': 'userdefined',
                      'puream': puream,
                      'DF_SCF_GUESS': 'False',
                      'scf_type': scf_type,
                      'dft_radial_scheme' : 'becke',
                      'dft_radial_points': 80,
                      'dft_spherical_points' : 974,
                      #'cubeprop_tasks': ['orbitals'],
                      #'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                      'CUBIC_GRID_OVERAGE' : L,
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'CUBIC_GRID_SPACING' : Dx,
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})

    job_nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
    psi4.set_num_threads(job_nthreads)

    #eb, wfn = psi4.energy('scf', return_wfn=True)
    mol_wfn = psi4.core.Wavefunction.build( \
                        molobj,psi4.core.get_global_option('basis'))
    ene,wfn=psi4.energy(func_l ,return_wfn=True)
    print("Relaxed super DFT ene: %.8f" %ene)

    #the composite basis set
    bset=mol_wfn.basisset()
    psi4.core.clean()
    
    #set the molecular object corresponding to the fragment B
    mLow=psi4.geometry(molB.geometry() +"symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com")
    psi4.set_options({'basis': gen_bset,
                      'puream': puream,
                      'scf_type': scf_type,
                      'dft_radial_scheme' : 'becke',
                      'dft_radial_points': 80,
                      'dft_spherical_points' : 974,
                      #'cubeprop_tasks': ['orbitals'],
                      #'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                      'CUBIC_GRID_OVERAGE' : L,
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'CUBIC_GRID_SPACING' : Dx,
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    eneB,isoB = psi4.energy(func_l,return_wfn=True)
    psi4.core.clean()

    #set the molecular object corresponding to the fragment A

    mHigh=psi4.geometry(molA.geometry() +"symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com")
    psi4.set_options({'basis': acc_bset,
                      'puream': puream,
                      'scf_type': scf_type,
                      'dft_radial_scheme' : 'becke',
                      'dft_radial_points': 80,
                      'dft_spherical_points' : 974,
                      #'cubeprop_tasks': ['orbitals'],
                      #'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                      'CUBIC_GRID_OVERAGE' : L,
                      'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                      'CUBIC_GRID_SPACING' : Dx,
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    # set more options
    if cc_flag:
        psi4.set_options({'cc_type': cc_type,
                        'df_ints_io' : 'save',
                        'freeze_core' : False,
                        'mp2_type': 'conv',
                        'MP2_AMPS_PRINT' : True,
                         'maxiter' : cc_maxiter})

    eneA,isoA = psi4.energy(func_l,return_wfn=True)
    
    if not cc_flag:
        psi4.core.clean()


    bsetAA=psi4.core.BasisSet.build(mHigh,'ORBITAL',acc_bset,puream=-1)
    bsetBB=psi4.core.BasisSet.build(mLow,'ORBITAL',gen_bset,puream=-1)
    return bset,bsetAA,bsetBB,molobj,wfn,ene,isoA,isoB
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
    parser.add_argument("--e_conv", help="Convergence energy threshold",required=False,
            default=1.0e-7, type = float)
    parser.add_argument("--d_conv", help="Convergence density threshold",
            default=1.0e-6, type = float)

    parser.add_argument("-f","--func", help="Specify the low level theory functional", required=False, 
            type=str, default="blyp")

    parser.add_argument("--scf_type", help="Specify the scf type: direct or df (for now)", required=False, 
            type=str, default='DIRECT')


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
    
    args = parser.parse_args()
    bset,bsetAA,bsetBB,supermol,wfnAB,ene_sup,isoA,isoB = initialize(args.scf_type,args.obs1,\
                                                    args.obs2,args.puream,args.geomA,args.geomB,args.func,args.e_conv,args.d_conv,\
                                                    args.charge,args.chargeA,args.multA,args.chargeB,args.multB)
