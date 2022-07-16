import numpy as np
# indirect LIST
class LISTi():
    def __init__(self,maxvec):

        self.__list_count = None
        self.__D_out_list = []
        self.__F_out_list = []
        
        self.__dF_list = []
        self.__dD_list = []
        self.__maxvec = maxvec

    def extrapolate(self,transpose=None):
       

       #if self.__scf_counter >=2 :
     
       # Limit of Fock_out list
       LIST_count = len(self.__F_out_list)
       if LIST_count > self.__maxvec:
           # Remove oldest vector
           del self.__D_out_list[0]
           del self.__F_out_list[0]
           del self.__dD_list[0]
           del self.__dF_list[0]
           LIST_count -= 1
       
       #print("Len of LISTi vectors (F_e, D_e): %i,%i"   %  ( len(F_diff_list),len(D_diff_list) ))
     
       # Build error matrix B (Yan Alexander Wang et al; https://doi.org/10.1063/1.3609242 )
       B = np.empty((LIST_count + 1, LIST_count + 1))
       #the lower part of the matrix frame (most external elemement) are filled with -1 and 0
       #[lower left corner], in the above paper is the upper matrix frame to be filled with -1
       B[-1, :] = -1
       B[:, -1] = -1
       B[-1, -1] = 0
       for num1, e1 in enumerate(self.__dD_list):
           for num2, e2 in enumerate(self.__dF_list):
               #if num2 > num1: continue
               val = np.einsum('ij,ij->', e1, e2)
               B[num1, num2] = val
               #if B would be sym
               #B[num2, num1] = val
       # normalize
       B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
       
     
       # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
       resid = np.zeros(LIST_count + 1)
       resid[-1] = -1
     
       # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
       ci = np.linalg.solve(B, resid)
     
       # Calculate new fock matrix as linear
       # combination of previous fock matrices
       F = np.zeros_like(self.__F_out_list[0])
       for num, c in enumerate(ci[:-1]):
           F += c * self.__F_out_list[num]  
       Dm = np.zeros_like(self.__D_out_list[0])
       for num, c in enumerate(ci[:-1]):
           Dm += c * self.__D_out_list[num]   
       

       return F,Dm

    #def Fock_in(self,fmat):
    #    self.__F_in_list.append(fmat)

    def Fock_out(self,fmat):     
        if not isinstance(fmat,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__F_out_list.append(fmat)

    def D_out(self,Dout):     
        if not isinstance(Dout,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__D_out_list.append(Dout)

    def diff_Fock(self,f_diff):
        if not isinstance(f_diff,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__dF_list.append( f_diff )


    def diff_D(self, d_diff):
        if not isinstance(d_diff,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__dD_list.append( d_diff )

    def list_count(self):
        list_count = len(self.__F_out_list)
        if list_count > self.__maxvec:
            list_count += -1
        return list_count
        

###################################
# direct LIST (LISTd)

class LISTd():
    def __init__(self,maxvec):
        self.__list_count = None
        self.__D_out_list = []
        self.__E_out_list = []
        self.__F_out_list = []
        
        self.__dF_list = []
        self.__maxvec = maxvec

    def extrapolate(self,transpose=False):
       
       # Limit of Fock_out list
       LIST_count = len(self.__F_out_list)
       if LIST_count > self.__maxvec:
           # Remove oldest vector
           del self.__D_out_list[0]
           del self.__E_out_list[0]
           del self.__F_out_list[0]
           del self.__dF_list[0]
           LIST_count -= 1
       
       #print("Len of LISTi vectors (F_e, D_e): %i,%i"   %  ( len(F_diff_list),len(D_diff_list) ))

       #### Alternative
       #### Build A in 3 steps
       #### a1_ij = Tr(D^{out}_j,(F^{out}_i - F^{in}_i))
       #a1 = np.zeros( (LIST_count,LIST_count) )
       #a1p = np.zeros( (LIST_count,LIST_count) )

       #for num1, e1 in enumerate(self.__D_out_list):
       #    for num2, e2 in enumerate(self.__dF_list):
       #        val = np.einsum('ij,ij->', e1, e2)
       #        a1[num2, num1] = val
       ### a1p_ij =  - Tr(D^{out}_i,(F^{out}_i - F^{out}_i))
       #for m2 in range(LIST_count):
       # for m1 in range(LIST_count):
       #        val = np.trace(np.matmul(self.__D_out_list[m1],self.__dF_list[m1]))
       #        a1p[m1,m2] = -val
       #partial = a1p + a1
       #for m2 in range(LIST_count):
       # for m1 in range(LIST_count):
               
       #        partial[m1,m2] +=  self.__E_out_list[m1]



       #check
       
       # Build error matrix B (Yan Alexander Wang et al; https://doi.org/10.1063/1.3609242 )
       B = np.empty((LIST_count + 1, LIST_count + 1))
       #the lower part of the matrix frame (most external elemement) are filled with -1 and 0
       #[lower left corner], in the above paper is the upper matrix frame to be filled with -1
       B[-1, :] = 1  #  depending on the paper
       B[:, -1] = 1  # 
       B[-1, -1] = 0
       dim0  = self.__D_out_list[0].shape[0]
       dcontainer = np.zeros((LIST_count,LIST_count,dim0,dim0))
       Acontainer = np.zeros((LIST_count,LIST_count))
       for num1, e1 in enumerate(self.__D_out_list):
           for num2, e2 in enumerate(self.__D_out_list):
             dcontainer[num2,num1] = e2 -e1
       for m2 in range(LIST_count):
        for m1 in range(LIST_count):
               val = np.trace(np.matmul(dcontainer[m2,m1],self.__dF_list[m1]))
               Acontainer[m1,m2] = val

       
       
       for m2 in range(LIST_count):
        for m1 in range(LIST_count):
               
               Acontainer[m1,m2] +=  self.__E_out_list[m1]
       #print("Check LIST matrix :  %s" % np.allclose(partial,Acontainer))
       #print("Max diff LIST mat %.8f " % (np.max(partial-Acontainer)))
       
       #using Acontainer.T we are defining LISTb (better)        
       if transpose :
         B[:LIST_count,:LIST_count]=Acontainer.T
         
       else:
         B[:LIST_count,:LIST_count]=Acontainer
       
       
       # normalize
       #B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
       
     
       # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
       resid = np.zeros(LIST_count + 1)
       resid[-1] = 1 
     
       # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
       ci = np.linalg.solve(B, resid)
     
       # Calculate new fock matrix as linear
       # combination of previous fock matrices
       F = np.zeros_like(self.__F_out_list[0])
       for num, c in enumerate(ci[:-1]):
           F += c * self.__F_out_list[num]  
       Dm = np.zeros_like(self.__D_out_list[0])
       for num, c in enumerate(ci[:-1]):
           Dm += c * self.__D_out_list[num]   
        
       #print("c[-1] = %.8f" % ci[-1])
       return F,Dm

    #def Fock_in(self,fmat):
    #    self.__F_in_list.append(fmat)

    def Fock_out(self,fmat):     
        if not isinstance(fmat,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__F_out_list.append(fmat)
        
    def D_out(self, Dout): 
        if not isinstance(Dout,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        self.__D_out_list.append(Dout)

    def E_out(self, Eout): 
        if not isinstance(Eout,(float)):
                raise TypeError("input must be a float")
        self.__E_out_list.append(Eout)
        
    def diff_Fock(self,f_diff):
        if not isinstance(f_diff,(np.ndarray)):
                raise TypeError("input must be a numpy.ndarray")
        
        self.__dF_list.append( f_diff )

    def list_count(self):
        list_count = len(self.__F_out_list)
        if list_count > self.__maxvec:
            list_count += -1
        return list_count
