import psi4 
import numpy as np
from scipy.linalg import fractional_matrix_power as mpower
import argparse


def rt_iter(wfn_obj,scf_common,loewdin):
    Ca = np.array(wfn_obj.Ca()) #used as orthonormal basis
    Cocc = np.array(wfn_obj.Ca_subset('AO','OCC'))
    ovap = scf_common.ovap

    if loewdin:
        try: 
            Vminus =mpower(ovap,-0.5)
        except scipy.linalg.LinAlgError:
            print("error in scipy.linalg fmp\n")
            
        try: 
            Vplus = mpower(ovap,0.5)
        except scipy.linalg.LinAlgError:
            print("error in scipy.linalg fmp\n")

        C_orth = np.matmul(Vplus,Cocc)    
    else:
       C_orth = np.eye(nbf_adduct)
       C_orth = C_orth[:,:ndocc]
       Vminus = Ca

    print("check Occ MO matrix in MO basis")
    C_test = np.matmul(Vminus,C_orth)
    print( np.allclose(Cocc,C_test ))
    print("setup done\n")

    import rtutil

    from base_embedding  import F_builder
    test = F_builder(scf_common)

    fock, proj, ene_test=test.get_Fock(Cocc,return_ene=True)
    FOCK = np.array(wfn_adduct.Fa())
    print("max diff(F[0] -F[-inf] : %.4e\n" % np.max(np.abs(fock-FOCK)))
    #print(np.allclose(fock,FOCK,atol=1.0e-7), np.max(np.abs(FOCK-fock)))

    #########################################################################
    print("CHECK:Build Fock from Density\n")
    Dmat_test = np.matmul(C_test,np.conjugate(C_test.T)) 
    print("check D: %s\n" %  np.allclose(Dmat_test,np.array(wfn_adduct.Da())) )
    fock_from_D, proj, ene_from_D=test.get_Fock(None,Dmat=Dmat_test,return_ene=True)

    print("max diff(F[0] -F[-inf] : %.4e\n" % np.max(np.abs(fock_from_D-FOCK)))
    print(np.allclose(fock,FOCK,atol=1.0e-7), np.max(np.abs(FOCK-fock_from_D)))

    dip_list = []
    print("start RT\n")
    rt_iter=20
    #set an initial density
    Dmat_ao = Dmat_test
    #set numerical kick
    pulse_opts = {'Fmax' : 2.0e-5, 'w' : 0., 's' : 0., 'imp_type' : 'kick'}

    for n_iter in range(rt_iter):
      # the 'actual' Fock
      fock_from_D, proj, ene_rt=test.get_Fock(None,Dmat=Dmat_ao,return_ene=True)
      
      pulse,fockmd,Cocc_rt = rtutil.mepc(C_orth,test,fock_from_D,FOCK,n_iter,0.1,dipole[2].np,\
                                                    Vminus,ovap,pulse_opts,debug=True)
      Dmat_rt = np.matmul(Cocc_rt,np.conjugate(Cocc_rt.T))
      Dmat_ao = np.matmul(Vminus,np.matmul(Dmat_rt,Vminus.T))
      dip_list.append(np.trace(np.matmul(Dmat_ao,dipole[2].np)).real)
      C_orth =Cocc_rt
      FOCK = fockmd

    t = np.linspace(0,0.1*(rt_iter-1),rt_iter)

    np.savetxt("dipole.tmp",np.c_[t,2.0*np.array(dip_list)])
    #print( np.max(np.abs(fockmd-fock_rt)) )
    #print(np.allclose(fock_rt,fock))








if __name__ == '__main__':

    


    
    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--loewdin", help="Set orthogonalized AO basis for RT ", required=False,
            default=False, action="store_true")
    args = parser.parse_args()
   

    mol1 = psi4.geometry("""
        LI          -0.000015401788    -1.357298417454     0.000000000000
            1,1
            symmetry c1
            no_com
            no_reorient
    """)

    mol_w=psi4.geometry(""" 
        O            0.000007733047     0.461865077395     0.000000000000
        H           -0.778328333938     1.059348634557     0.000000000000
        H            0.778312824867     1.059390049612     0.000000000000
            0,1
            symmetry c1
            no_com
            no_reorient
    """)

    mol = psi4.geometry("""

        LI          -0.000015401788    -1.357298417454     0.000000000000
        1,1
        O            0.000007733047     0.461865077395     0.000000000000
        H           -0.778328333938     1.059348634557     0.000000000000
        H            0.778312824867     1.059390049612     0.000000000000
            symmetry c1
            no_com
            no_reorient
    """)

    mol_h2 = psi4.geometry("""
            H   0.000000000000     0.   0.
            H   0.000000000000    0.    0.72
            0,1
            symmetry c1
            no_com
            no_reorient
    """)
    psi4.core.set_output_file('psi4_out.dat', False)
    psi4.set_options({'basis' : '6-311++G**',
                      'puream' : 'True',
                      'scf_type': 'direct',
                      'df_scf_guess' : 'False',
                      'e_convergence' : 1e-8,
                      'd_convergence' : 1e-8})
    ene_li, wfn_li = psi4.energy("svwn",molecule=mol1,return_wfn=True)
    basis_li = wfn_li.basisset()
    nbf_li = basis_li.nbf()
    ndocc_li = wfn_li.nalpha()
    Cocc_li = np.array(wfn_li.Ca_subset('AO','OCC'))
    ##################


    #ene_w, wfn_w = psi4.energy("svwn",molecule=mol_w,return_wfn=True)


    ##################

    ene_adduct, wfn_adduct = psi4.energy("svwn",molecule=mol_h2,return_wfn=True)
    ndocc = wfn_adduct.nalpha()
    basis_adduct = wfn_adduct.basisset()

    mints = psi4.core.MintsHelper(basis_adduct)
    nbf_adduct = basis_adduct.nbf() 

    # jk class
    jk = psi4.core.JK.build(basis_adduct)
    jk.set_memory(int(4.0e9))  # 1GB
    jk.set_do_wK(False)
    jk.initialize()
    #

    T = np.array(mints.ao_kinetic())
    V = np.array(mints.ao_potential())
    ovap = np.array(mints.ao_overlap())

    dipole = mints.ao_dipole()
    H = T+V

    #set a supermol calculation nbf_frag == nbf_adduct                       
    # doubly occupied MO of the fragment-> complete system

    import sys
    sys.path.insert(0,'../FnT')

    from base_embedding import fock_common

    data_common = fock_common

    data_common.nbf = nbf_adduct
    data_common.nbf_tot = nbf_adduct
    data_common.docc_num =  ndocc
    data_common.Hcore = H
    data_common.ovap = ovap
    data_common.sup_basis =basis_adduct
    data_common.mono_basis = None
    data_common.jk = jk
    data_common.jk_mono = None
    data_common.func_name = "svwn"

    rt_iter(wfn_adduct,data_common,args.loewdin)
