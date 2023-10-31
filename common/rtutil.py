## exponental midpont+predictor-corrector to propagate electron density
import sys
import numpy as np

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

def mepc(Corth_ti,fock_builder,fock_mid_ti_backwd,i,delta_t,dipole,\
                        Vminus,ovap,imp_opts, do_proj=False, maxiter= 10 ,fout=sys.stderr,debug=False):
    t_arg=np.float_(i)*np.float_(delta_t)
    
    if imp_opts is None:
        pulse = 0.
    else:    
        func = funcswitcher.get(imp_opts['imp_type'], lambda: kick)
    
        pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                            imp_opts['t0'], imp_opts['s'])

    #Dp_ti is in the propgation (orthonormal) basis

    #transform in the AO basis
    C_ti= np.matmul(Vminus,Corth_ti)
    D_ti = np.matmul(C_ti,np.conjugate(C_ti.T)) 
    k=1
    
    fock_mtx,proj,ene=fock_builder.get_Fock(None,Dmat=D_ti,return_ene=True)#the frag_id is meaningless here
    #DEBUG
    #ExcAAhigh_i=0.0
    #ExcAAlow_i=0.0
    
    #add -pulse*dipole
    fock_ti_ao = fock_mtx - (dipole*pulse)
    if do_proj:
        fock_ti_ao +=proj

    #if i==0:
    #    print('F(0) equal to F_ref: %s' % np.allclose(fock_ti_ao,fock_mid_ti_backwd))
    
    #initialize dens_test !useless

    # set guess for initial fock matrix
    fock_guess = 2.00*fock_ti_ao - fock_mid_ti_backwd
    if debug:
            print("max diff(F[i] -F[i-1] : %.4e\n" % np.max(np.abs(fock_ti_ao-fock_mid_ti_backwd)))
    #if i==0:
    #   print('Fock_guess for i =0 is Fock_0: %s' % np.allclose(fock_guess,fock_ti_ao))
    #transform fock_guess in MO basis
    while True:
        fockp_guess=np.matmul(np.conjugate(Vminus.T),np.matmul(fock_guess,Vminus))
        u=exp_opmat(fockp_guess,delta_t)
        #u=scipy.linalg.expm(-1.j*fockp_guess*delta_t) ! alternative routine
        test=np.matmul(u,np.conjugate(u.T))
    #print('U is unitary? %s' % (np.allclose(test,np.eye(u.shape[0]))))
        if (not np.allclose(test,np.eye(u.shape[0]))):
            Id=np.eye(u.shape[0])
            diff_u=test-Id
            norm_diff=np.linalg.norm(diff_u,'fro')
            print('from fock_mid:U deviates from unitarity, |UU^-1 -I| %.8f' % norm_diff)
    #evolve Dp_ti using u and obtain Dp_ti_dt (i.e Dp(ti+dt)). u i s built from the guess fock
    #density in the orthonormal basis
        Corth_ti_dt=np.matmul(u,Corth_ti)
    #backtrasform Dp_ti_dt
        C_ti_dt=np.matmul(Vminus,Corth_ti_dt)
        D_ti_dt = np.matmul(C_ti_dt,np.conjugate(C_ti_dt.T))
    #build the correspondig Fock : fock_ti+dt
        
        #DEBUG
        #dum1,dum2,fock_mtx=get_Fock(D_ti_dt,H,I,func_l,basisset)
        fock_mtx,proj = fock_builder.get_Fock(None,Dmat=D_ti_dt,return_ene=False)
        
        #print('fockmtx s in loop max diff: %.12e\n' % np.max(tfock_mtx-fock_mtx))
        #update t_arg+=delta_t

        if imp_opts is None:
            pulse_dt = 0.
        else:    
            pulse_dt = func(imp_opts['Fmax'], imp_opts['w'], t_arg+delta_t,\
                            imp_opts['t0'], imp_opts['s'])
        fock_ti_dt_ao=fock_mtx -(dipole*pulse_dt)
        if do_proj:
            fock_ti_dt_ao +=proj

        if debug:
            print("max diff(F[i+1] -F[i] : %.4e\n" % np.max(np.abs(fock_ti_dt_ao-fock_ti_ao)))
       
        fock_inter= 0.5*fock_ti_ao + 0.5*fock_ti_dt_ao
    #update fock_guess
        fock_guess=np.copy(fock_inter)
        if k >1:
        #test on the norm: compare the density at current step and previous step
        #calc frobenius of the difference D_ti_dt_mo_new-D_ti_dt_mo
            diff=D_ti_dt-dens_test
            norm_f=np.linalg.norm(diff,'fro')
            if norm_f<(1e-6):
                tr_dt=np.trace(np.matmul(ovap,D_ti_dt))
                fout.write('converged after %i interpolations\n' % (k-1))
                fout.write('i is: %d\n' % i)
                fout.write('norm is: %.12f\n' % norm_f)
                fout.write('Trace(D)(t+dt) : %.8f\n' % tr_dt.real)
                break
        dens_test=np.copy(D_ti_dt)
        k+=1
        if k > maxiter:
         fout.write('norm is: %.12f\n' % norm_f)
         #raise Exception("Numember of iterations exceeded maxit = %i)" % maxiter)
    # return energy components , the Fock, the forward midpoint fock and the evolved density matrix 
    return ene,pulse,fock_ti_ao,fock_inter,Corth_ti_dt
