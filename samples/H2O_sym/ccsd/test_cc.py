import psi4
import numpy as np

numpy_mem = 4
totmol=psi4.geometry("""
O -1.4626 0.0000 0.0000 
H -1.7312 0.9302 0.0000 
H -0.4844 0.0275 0.0000 
O 1.6626 0.0000 0.0000 
H 1.9312 0.9302 0.0000 
H 0.6844 0.0275 0.0000 
 
   0 1
   symmetry c1
   no_com  
   no_reorient
""")
psi4.set_options({'basis': 'cc-pvtz',
                  'puream': 'True',
                  'scf_type': 'direct', # direct by default
                  'cc_type': 'conv',  # conv (conventional) by default
                  'df_ints_io' : 'save',
                  'freeze_core' : False,
                  'mp2_type': 'conv',
                  'MP2_AMPS_PRINT' : True,
                  'maxiter' : 50,
                  #'dft_radial_scheme' : 'becke',
                  #'dft_radial_points': 80,
                  #'dft_spherical_points' : 974,
                  'cubeprop_tasks': ['density','orbitals'],
                  'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                  'CUBIC_GRID_OVERAGE' : [4.5,4.5,4.5],
                  'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                  'e_convergence': 1e-7,
                  'd_convergence': 1e-8,
                  'r_convergence': 1e-5})
# ref = hf
#ref_ene, ref_wfn = psi4.energy('scf', return_wfn = True)
ref_ene, ref_wfn = psi4.energy('scf', return_wfn = True)
ene_cc = psi4.energy('ccsd',ref_wfn=ref_wfn)
