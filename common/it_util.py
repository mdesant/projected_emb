import numpy as np
import sys
import os
modpaths = os.environ.get('RTUTIL_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

import ortho
##################################################################
def make_orthog_MOs(MOcoeff,out_file,verbose=False):
      test_ortho = np.matmul(np.conjugate(MOcoeff.T),MOcoeff)
      err = np.allclose(test_ortho,np.eye(test_ortho.shape[1]),atol=1.0e-12)
      if (not err):
         #orthogonalize it
         if verbose:
            out_file.write("Max deviation: %.8e\n" % np.max(np.abs(test_ortho-np.eye(test_ortho.shape[1]))))
            C_ortho, testerr = ortho.modified_gram_schmidt(MOcoeff,test=verbose)
            out_file.write("MOcoeff orth: %s\n" % testerr)
         else:
            C_ortho = ortho.modified_gram_schmidt(MOcoeff,test=verbose)
         MOcoeff = np.copy(C_ortho)
      return MOcoeff
##################################################################
##################################################################

def exp_opmat(mat,dt):
    #first find eigenvectors and eigenvalues of F mat
    try:
       w,v=np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        print("Error in numpy.linalg.eigh of inputted matrix")
        return None

    diag=np.exp(-1.j*w*dt)

    dmat=np.diagflat(diag)

    # for a general matrix Diag = M^(-1) A M
    # M is v
    #try:
    #   v_i=np.linalg.inv(v)
    #except np.linalg.LinAlgError:
    #   return None

    # transform back
    #tmp = np.matmul(dmat,v_i)
    tmp = np.matmul(dmat,np.conjugate(v.T))

    #in an orthonrmal basis v_inv = v.H

    mat_exp = np.matmul(v,tmp)

    return mat_exp

##################################################################
# Vminus = C(0) | S^{-0.5}
def prop_mmut(fock_ti,MOcoeff,C_midb,delta_t,Vminus,fout=sys.stderr):
    # if at t_i no  previous midpoint density/Coeff is available
    # we evolve backward in order to obtain D(t_i-1/2)
 
    if MOcoeff is not None:
       
      if not isinstance(MOcoeff,(np.ndarray)):
         raise TypeError("input must be a numpy.ndarray")
      #get the MO coefficients on the mo basis (MOs at t=0)
      C_ti_mo = MOcoeff # will be used only if C_midb is None, see below

    elif (Mocoeff is None) and (C_midb is None):
      raise Exception("input C coeff not provided\n")


    #transform fock_ti in the MO ref basis
    fock_ti_mo=np.matmul(np.conjugate(Vminus.T),np.matmul(fock_ti,Vminus))
    #calculate u
    #perform some test
    u=exp_opmat(fock_ti_mo,delta_t)
    # avoid printing 
    if np.isreal(delta_t):
       test=np.matmul(u,np.conjugate(u.T))
       #print('U is unitary? %s' % (np.allclose(test,np.eye(u.shape[0]))))
       if (not np.allclose(test,np.eye(u.shape[0]))):
           Id=np.eye(u.shape[0])
           diff_u=test-Id
           norm_diff=np.linalg.norm(diff_u,'fro')
           fout.write('from fock_mid:U is not unitary, |UU^-1 -I| %.8f' % norm_diff)
    #calculate the new u operator ( for the half-interval propagation)
    u2=exp_opmat(fock_ti_mo,delta_t/2.00)
    if C_midb is None:
        #fout.write("\n")
        #fout.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        #fout.write("@ C_backw is None   @\n")
        #fout.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        #calculate C(i-1/2) ->C_midb  (the suffix mo has been dropped: every matrix involved the
        #evolution step is in the mo basis)
        C_midb = np.matmul(np.conjugate(u2.T),C_ti_mo)
    #end of the if condition
    
    if C_midb is not None:
       if not isinstance(C_midb,(np.ndarray)):
          raise TypeError("input must be a numpy.ndarray")
    #orthogonalize C_midb if needed
    C_midb = make_orthog_MOs(C_midb,fout,False)
    #fout.write('trace of D_midb(%i) : %.8f\n' % (i,np.trace(np.matmul(C_midb,np.conjugate(C_midb.T))).real))
    
    # evolve on the entire time step 
    C_midf=np.matmul(u,C_midb)
    #orthogonalize C_midf if needed
    C_midf = make_orthog_MOs(C_midf,fout,False)

    #get C_ti_dt_mo using u2
    C_ti_dt_mo = np.matmul(u2,C_midf)
    #make the orbital orthogonal before transforming
    C_ti_dt_mo = make_orthog_MOs(C_ti_dt_mo,fout,False)
    # back transform on the AO basis
    C_ti_dt = np.matmul(Vminus,C_ti_dt_mo)
    return C_ti_dt,C_midf
