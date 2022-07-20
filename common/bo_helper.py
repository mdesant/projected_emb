import psi4
import numpy as np
from pkg_resources import parse_version

def test_Fock(Dbo, Hcore, I,U,func_l, basisset):
    #na is the number of basis function of subsys A
    # Hcore if given BO basis

    #Dbo is transformed to the AO basis
    D=np.matmul(U,np.matmul(Dbo,U.T))
    
    # Build J matrix for the low level theory
    J = np.einsum('pqrs,rs->pq', I, D)

    # Build Vxc matrix for the low level theory
    #D must be a psi4.core.Matrix object not a numpy.narray
    restricted = True
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
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
    #compute the corresponding XC energy (low level)
    Exc= potential.quadrature_values()["FUNCTIONAL"]

    #notation: G is the 2 electron operator. AA tag is present when referring to AA subblock of D
    #Glow = 2.0*J+V -> Glowtilde = U Glow U^T

    Glowtilde=np.matmul(U.T,(np.matmul((2.0*J+V),U)))
   
    Ftilde = psi4.core.Matrix.from_array(Glowtilde)
    Ftilde.add(psi4.core.Matrix.from_array(Hcore))
    Coul=2.00*np.trace(np.matmul(D,J))
    return Coul,Exc,Ftilde

def get_BOFock(Dbo, Hcore, I,U, func_h, func_l, basisset,bsetH,exmodel=0):
    #na is the number of basis function of subsys A
    na=bsetH.nbf()
    # Hcore if given BO basis

    #Dbo is transformed to the AO basis
    D=np.matmul(U,np.matmul(Dbo,U.T))
    
    # Build J matrix for the low level theory
    J = np.einsum('pqrs,rs->pq', I, D)

    # Build Vxc matrix for the low level theory
    #D must be a psi4.core.Matrix object not a numpy.narray
    restricted = True
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
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
    #compute the corresponding XC energy (low level)
    Exclow= potential.quadrature_values()["FUNCTIONAL"]
    if sup.is_x_hybrid():
      #
      raise Exception("Low level theory functional is Hybrid?\n")
      alpha = sup.x_alpha()
      Klow = np.einsum('prqs,rs->pq', I, D)
      V.add(psi4.core.Matrix.from_array(-alpha*Klow))
      Exclow += -alpha*np.trace(np.matmul(D,Klow))

    #notation: G is the 2 electron operator. AA tag is present when referring to AA subblock of D
    #Glow = 2.0*J+V -> Glowtilde = U Glow U^T

    Glowtilde=np.matmul(U.T,(np.matmul((2.0*J+V),U)))
   
    Ftilde = psi4.core.Matrix.from_array(Glowtilde)
    Ftilde.add(psi4.core.Matrix.from_array(Hcore))


    #calculate VxcAAhigh and VxcAAlow using Dbo[:na,:na] (the AA D subblock in BO)
    #D must be a psi4.core.Matrix object not a numpy.narray

    #VcxAAlow
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    vname = "RV"
    if not restricted:
        vname = "UV"
    potential=psi4.core.VBase.build(bsetH,sup,vname)
    Dm=psi4.core.Matrix.from_array(Dbo.np[:na,:na])
    potential.initialize()
    potential.set_D([Dm])
    VxcAAlow=psi4.core.Matrix(na,na)
    potential.compute_V([VxcAAlow])
    potential.finalize()
    # compute the low level XC energy  on Dbo AA
    ExcAAlow= potential.quadrature_values()["FUNCTIONAL"]
    if sup.is_x_hybrid():
      #
      raise Exception("Low level theory functional is Hybrid?\n")
      alpha = sup.x_alpha()
      KAAlow = np.einsum('prqs,rs->pq', I[:na,:na,:na,:na], Dbo.np[:na,:na])
      if exmodel==1:
          KAA1 = np.einsum('prqs,rs->pq', I[:na,na:,:na,na:], D[na:,na:])
          KAAlow += KAA1
      VxcAAlow.add(psi4.core.Matrix.from_array(-alpha*KAAlow))
      ExcAAlow += -alpha*np.trace(np.matmul(Dbo.np[:na,:na],KAAlow))
    
    #VcxAAhigh
    if func_h=='hf':
       K = np.einsum('prqs,rs->pq', I[:na,:na,:na,:na], Dbo.np[:na,:na]) #assuming Exc0 model?
       #Exc_ex0 =  -np.trace(np.matmul(Dbo.np[:na,:na],K))
       #print("EX0 energy: %.10e\n" % Exc_ex0)
       #exchange model 1
       if exmodel==1:
           K1 = np.einsum('prqs,rs->pq', I[:na,na:,:na,na:], D[na:,na:])
           #Exc_ex1 =  -np.trace(np.matmul(Dbo.np[:na,:na],K1))
           #print("EX1 energy: %.10e\n" % Exc_ex1)
           K += K1

       VxcAAhigh=psi4.core.Matrix.from_array(-K)
       VxcAAhigh.subtract(VxcAAlow)
       ExcAAhigh = -np.trace(np.matmul(Dbo.np[:na,:na],K))

    else:
       if parse_version(psi4.__version__) >= parse_version('1.3a1'):
               build_superfunctional = psi4.driver.dft.build_superfunctional
       else:
               build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
       sup = build_superfunctional(func_h, restricted)[0]
       sup.set_deriv(2)
       sup.allocate()
       vname = "RV"
       if not restricted:
           vname = "UV"
       potential=psi4.core.VBase.build(bsetH,sup,vname)
       Dm=psi4.core.Matrix.from_array(Dbo.np[:na,:na])
       potential.initialize()
       potential.set_D([Dm])
       nbf=D.shape[0]
       VxcAAhigh=psi4.core.Matrix(na,na)
       potential.compute_V([VxcAAhigh])
       potential.finalize()
       # compute the high level XC energy  on Dbo AA
       ExcAAhigh= potential.quadrature_values()["FUNCTIONAL"]
       
       # we subtract VxcAAlow  from VxcAAhigh
       VxcAAhigh.subtract(VxcAAlow)
       
       if sup.is_x_hybrid():
         alpha = sup.x_alpha()
         K = np.einsum('prqs,rs->pq', I[:na,:na,:na,:na], Dbo.np[:na,:na])
         if exmodel==1:
           K1 = np.einsum('prqs,rs->pq', I[:na,na:,:na,na:], D[na:,na:])
           K += K1
         VxcAAhigh.add(psi4.core.Matrix.from_array(-alpha*K))
         ExcAAhigh += -alpha*np.trace(np.matmul(Dbo.np[:na,:na],K))
    
    tmp=np.zeros((nbf,nbf))
    tmp[:na,:na] =  np.asarray(VxcAAhigh)
    Ftilde.add(psi4.core.Matrix.from_array(tmp))
    Eh=2.00*np.trace(np.matmul(D,J))
    return Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde

def get_BOFockRT(Dbo, Hcore, I,U, func_h, func_l, basisset,bsetH,exmodel=0):
    #na is the number of basis function of subsys A
    na=bsetH.nbf()
    # Hcore if given BO basis

    #Dbo is transformed to the AO basis
    D=np.matmul(U,np.matmul(Dbo,U.T))
    
    # Build J matrix for the low level theory
    J = np.einsum('pqrs,rs->pq', I, D)

    # Build Vxc matrix for the low level theory
    #D must be a psi4.core.Matrix object not a numpy.narray
    restricted = True
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
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
    #compute the corresponding XC energy (low level)
    Exclow= potential.quadrature_values()["FUNCTIONAL"]

    #notation: G is the 2 electron operator. AA tag is present when referring to AA subblock of D
    #Glow = 2.0*J+V -> Glowtilde = U Glow U^T

    Glowtilde=np.matmul(U.T,(np.matmul((J*np.float_(2.0)+V),U)))
   
    Ftilde = Glowtilde + Hcore


    #calculate VxcAAhigh and VxcAAlow using Dbo[:na,:na] (the AA D subblock in BO)
    #D must be a psi4.core.Matrix object not a numpy.narray

    #VcxAAlow
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    vname = "RV"
    if not restricted:
        vname = "UV"
    potential=psi4.core.VBase.build(bsetH,sup,vname)
    Dm=psi4.core.Matrix.from_array(Dbo[:na,:na].real)
    potential.initialize()
    potential.set_D([Dm])
    VxcAAlow=psi4.core.Matrix(na,na)
    potential.compute_V([VxcAAlow])
    potential.finalize()
    # compute the low level XC energy  on Dbo AA
    ExcAAlow= potential.quadrature_values()["FUNCTIONAL"]
    
    #VcxAAhigh
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_h, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    vname = "RV"
    if not restricted:
        vname = "UV"
    potential=psi4.core.VBase.build(bsetH,sup,vname)
    Dm=psi4.core.Matrix.from_array(Dbo[:na,:na].real)
    potential.initialize()
    potential.set_D([Dm])
    nbf=D.shape[0]
    VxcAAhigh=psi4.core.Matrix(na,na)
    potential.compute_V([VxcAAhigh])
    potential.finalize()
    # compute the high level XC energy  on Dbo AA
    ExcAAhigh= potential.quadrature_values()["FUNCTIONAL"]
    
    # we subtract VxcAAlow  from VxcAAhigh
    VxcAAhigh.subtract(VxcAAlow)
    
    if sup.is_x_hybrid():
      alpha = sup.x_alpha()
      K = np.einsum('prqs,rs->pq', I[:na,:na,:na,:na], Dbo[:na,:na])
      if exmodel==1:
          K1 = np.einsum('prqs,rs->pq', I[:na,na:,:na,na:], D[na:,na:])
          K += K1
      ExcAAhigh += -alpha*np.trace(np.matmul(Dbo[:na,:na],K))
      VxcAAhigh = np.asarray(VxcAAhigh)-alpha*K
    #print("K is complex obj: %s\n" % np.iscomplexobj(K)) 
    #print("VxcAAhigh is complex obj: %s\n" % np.iscomplexobj(VxcAAhigh.np)) 
    Ftilde[:na,:na]+=VxcAAhigh
    Eh=2.00*np.trace(np.matmul(D,J))
    return Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde

def get_BOFock_JK(Dbo, Cocc, Hcore, jk,U, func_h, func_l, basisset,bsetH,na,exmodel=0):
    #na is the number of basis function of subsys A
    # Hcore if given BO basis

    #Dbo is transformed to the AO basis
    D=np.matmul(U,np.matmul(Dbo,U.T))
    
    #Cocc is transformed to the AO basis
    CoccAO=np.matmul(U,Cocc)
    
    # Build J matrix for the low level theory using JK class

    jk.C_left_add(psi4.core.Matrix.from_array(CoccAO))
    jk.compute()
    jk.C_clear()
    J=np.array(jk.J()[0]) #copy into J
    # Build Vxc matrix for the low level theory
    #D must be a psi4.core.Matrix object not a numpy.narray
    restricted = True
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
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
    #compute the corresponding XC energy (low level)
    Exclow= potential.quadrature_values()["FUNCTIONAL"]

    if sup.is_x_hybrid():
      #
      #raise Exception("Low level theory functional is Hybrid?\n")
      alpha = sup.x_alpha()
      Klow = np.array(jk.K()[0])
      V.add(psi4.core.Matrix.from_array(-alpha*Klow))
      Exclow += -alpha*np.trace(np.matmul(D,Klow))
    #notation: G is the 2 electron operator. AA tag is present when referring to AA subblock of D
    #Glow = 2.0*J+V -> Glowtilde = U Glow U^T

    Glowtilde=np.matmul(U.T,(np.matmul((2.0*J+V),U)))
   
    Ftilde = psi4.core.Matrix.from_array(Glowtilde)
    Ftilde.add(psi4.core.Matrix.from_array(Hcore))


    #calculate VxcAAhigh and VxcAAlow using Dbo[:na,:na] (the AA D subblock in BO)
    #D must be a psi4.core.Matrix object not a numpy.narray

    #VcxAAlow
    if parse_version(psi4.__version__) >= parse_version('1.3a1'):
            build_superfunctional = psi4.driver.dft.build_superfunctional
    else:
            build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    sup = build_superfunctional(func_l, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    vname = "RV"
    if not restricted:
        vname = "UV"
    potential=psi4.core.VBase.build(bsetH,sup,vname)
    Dm=psi4.core.Matrix.from_array(Dbo.np[:na,:na])
    potential.initialize()
    potential.set_D([Dm])
    VxcAAlow=psi4.core.Matrix(na,na)
    potential.compute_V([VxcAAlow])
    potential.finalize()
    # compute the low level XC energy  on Dbo AA
    ExcAAlow= potential.quadrature_values()["FUNCTIONAL"]
    if sup.is_x_hybrid():
      #
      #raise Exception("Low level theory functional is Hybrid?\n")
      alpha = sup.x_alpha()
      Cocc_A=np.zeros_like(Cocc)
      Cocc_A[:na,:]=np.asarray(Cocc)[:na,:]
      
      jk.C_left_add(psi4.core.Matrix.from_array(Cocc_A))
      jk.compute()
      jk.C_clear()  
      
      KAAlow = np.asarray(jk.K()[0])[:na,:na]
      if exmodel==1:
          Cocc_B=np.zeros_like(Cocc)
          Cocc_B[na:,:]=np.asarray(CoccAO)[na:,:] #see J. Chem. Theory Comput. 2017, 13, 1605-1615
      
          jk.C_left_add(psi4.core.Matrix.from_array(Cocc_B))
          jk.compute()
          jk.C_clear()  
      
          KAA1 = np.array(jk.K()[0])[:na,:na]
          KAAlow += KAA1
      VxcAAlow.add(psi4.core.Matrix.from_array(-alpha*KAAlow))
      ExcAAlow += -alpha*np.trace(np.matmul(Dbo.np[:na,:na],KAAlow))
    
    #VcxAAhigh
    if func_h=='hf':
       Cocc_A=np.zeros_like(Cocc)
       Cocc_A[:na,:]=np.asarray(Cocc)[:na,:]
       
       #DEBUG
       #check=np.matmul(Cocc_A,Cocc_A.T)
       #print("Dbo[:na,:na] & Dbo[Cocc[:na,:]] : %s\n" % np.allclose(Dbo.np[:na,:na],check[:na,:na]))
       
       jk.C_left_add(psi4.core.Matrix.from_array(Cocc_A))
       jk.compute()
       jk.C_clear()  
       K = np.array(jk.K()[0])[:na,:na]    #assuming Exc0 model?
       #Exc_ex0 =  -np.trace(np.matmul(Dbo.np[:na,:na],K))
       #print("ExceAAhigh EX0 mod: %.10e\n" % Exc_ex0)
       #exchange model 1
       if exmodel==1:
           Cocc_B=np.zeros_like(Cocc)
           Cocc_B[na:,:]=np.asarray(CoccAO)[na:,:] #see J. Chem. Theory Comput. 2017, 13, 1605-1615
           
           #DEBUG
           #check=np.matmul(Cocc_B,Cocc_B.T)
           #print("Dbo[na:,na:] & Dbo[Cocc[na:,:]] : %s\n" % np.allclose(Dbo.np[na:,na:],check[na:,na:]))
           
           jk.C_left_add(psi4.core.Matrix.from_array(Cocc_B))
           jk.compute()
           jk.C_clear()  
       
           K1 = np.array(jk.K()[0])[:na,:na]
           #Exc_ex1 =  -np.trace(np.matmul(Dbo.np[:na,:na],K1))
           #print("EX1 energy: %.10e\n" % Exc_ex1)
           K += K1
       VxcAAhigh=psi4.core.Matrix.from_array(-K)
       VxcAAhigh.subtract(VxcAAlow)
       ExcAAhigh = -np.trace(np.matmul(Dbo.np[:na,:na],K))
    
    else:
       if parse_version(psi4.__version__) >= parse_version('1.3a1'):
               build_superfunctional = psi4.driver.dft.build_superfunctional
       else:
               build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
       sup = build_superfunctional(func_h, restricted)[0]
       sup.set_deriv(2)
       sup.allocate()
       vname = "RV"
       if not restricted:
           vname = "UV"
       potential=psi4.core.VBase.build(bsetH,sup,vname)
       Dm=psi4.core.Matrix.from_array(Dbo.np[:na,:na])
       potential.initialize()
       potential.set_D([Dm])
       nbf=D.shape[0]
       VxcAAhigh=psi4.core.Matrix(na,na)
       potential.compute_V([VxcAAhigh])
       potential.finalize()
       # compute the high level XC energy  on Dbo AA
       ExcAAhigh= potential.quadrature_values()["FUNCTIONAL"]
    
    # we subtract VxcAAlow  from VxcAAhigh
       VxcAAhigh.subtract(VxcAAlow)
       #if func_h == func_l:  #check
       #    print("G_high and G_low cancel: %s" % (np.allclose(VxcAAhigh.np,np.zeros((na,na)),atol=1.0e-14)))
       if sup.is_x_hybrid():
         alpha = sup.x_alpha()
         Cocc_A=np.zeros_like(Cocc)
         Cocc_A[:na,:]=np.asarray(Cocc)[:na,:]
         
         jk.C_left_add(psi4.core.Matrix.from_array(Cocc_A))
         jk.compute()
         jk.C_clear()  
         
         K = np.array(jk.K()[0])[:na,:na]
         if exmodel==1:
             Cocc_B=np.zeros_like(Cocc)
             Cocc_B[na:,:]=np.asarray(CoccAO)[na:,:] #see J. Chem. Theory Comput. 2017, 13, 1605-1615
             
         
             jk.C_left_add(psi4.core.Matrix.from_array(Cocc_B))
             jk.compute()
             jk.C_clear()  
         
             K1 = np.array(jk.K()[0])[:na,:na]
             K += K1
         VxcAAhigh.add(psi4.core.Matrix.from_array(-alpha*K))
         ExcAAhigh += -alpha*np.trace(np.matmul(Dbo.np[:na,:na],K))
    
    tmp=np.zeros((nbf,nbf))
    tmp[:na,:na] =  np.asarray(VxcAAhigh)
    Ftilde.add(psi4.core.Matrix.from_array(tmp))
    Eh=2.00*np.trace(np.matmul(D,J))
    return Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde

###########################################################################
def get_AOFock_JK(Dmat, Cocc, Hcore, jk, func, basisset):

    
    
    # Build J matrix for the low level theory using JK class

    jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
    jk.compute()
    jk.C_clear()
    J=np.array(jk.J()[0]) #copy into J
    Jene = 2.00*np.trace( np.matmul(J,Dmat) )
    if func == 'hf':
        K=np.array(jk.K()[0]) #copy into K
        fock = Hcore + 2.00*J -K
        # E_x, just to re-use the same variable name
        Exc = -np.trace( np.matmul(K,Dmat))
    else:
        # Build Vxc matrix 
        #D must be a psi4.core.Matrix object not a numpy.narray
        restricted = True
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
                build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
                build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(func, restricted)[0]
        sup.set_deriv(2)
        sup.allocate()
        vname = "RV"
        if not restricted:
            vname = "UV"
        potential=psi4.core.VBase.build(basisset,sup,vname)
        Dm=psi4.core.Matrix.from_array(Dmat.real)
        potential.initialize()
        potential.set_D([Dm])
        nbf=Dmat.shape[0]
        V=psi4.core.Matrix(nbf,nbf)
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
          Exc += -alpha*np.trace(np.matmul(Dmat,K))
 
        fock = Hcore + 2.00*J + V
    return fock,Jene,Exc
