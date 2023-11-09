# class for EDIIS/ADIIS
# largerly inspired by DIIS_helper class form psi4numpy/self-Consistent-Field/helper_HF.py

# Helper classes and functions for the SCF folder.

#References:
#- DIIS equations & algorithm from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
#- Orbital rotaion expressions from [Helgaker:2000]
#- Orbital rotaion expressions from [Helgaker:2000]
#"""

#__authors__ = "Daniel G. A. Smith"
#__credits__ = ["Daniel G. A. Smith"]

#__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
#__license__ = "BSD-3-Clause"
#__date__ = "2017-9-30"
#
# AND
# https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/diis.py#L3

import numpy as np
from itertools import product

class EDIIS_helper():
    def __init__(self,max_vec=4,closed_shell=True,engine='e_diis'):
        self.maxvec = max_vec
        self.f_list = []
        self.d_list = []
        self.ene_vec= []
        self.ediis_quadratic = None
        self.adiis_quadratic = None
        self.adiis_linear = None
        self.closed_shell = closed_shell
        self.__engine = engine
        #check engine id : a_diis or e_diis
        if engine != 'e_diis' and engine !='a_diis':
           raise Exception("wrong engine key\n")
        self.__ccoef = None
    def add(self, fock, density, energy):
        """
         Add energy,fock,and density to the corresponding vectors
        """
        self.f_list.append(fock)
        self.d_list.append(density)
        self.ene_vec.append(energy)

    def ediis_populate(self):

     
     num_entries = len(self.f_list)
     self.ediis_quadratic = np.zeros((num_entries, num_entries))
     for i in range(num_entries):
        for item_num in range(len(self.d_list[i])):
            d = self.d_list[i][item_num]
            for j in range(num_entries):
                f = self.f_list[j][item_num]
                self.ediis_quadratic[i][j] +=d.dot(f)
     diag = np.diag(self.ediis_quadratic)
     # D_i F_i + D_j F_j - D_i F_j - D_j F_i; First two terms use broadcasting tricks
     self.ediis_quadratic = diag[:, None] + diag - self.ediis_quadratic - self.ediis_quadratic.T
     self.ediis_quadratic *= -1/2

     if self.closed_shell:
        self.ediis_quadratic *= 2

    #EDIIS fuctional
    def ediis_energy(self, x):
       ediis_linear = np.array([entry for entry in self.ene_vec])
       return np.dot(ediis_linear, x) + np.einsum("i,ij,j->", x, self.ediis_quadratic, x)/2.

    def ediis_gradient(self, x):
       """ Gradient of energy estimate w.r.t. input coefficient """
       ediis_linear = np.array([entry for entry in self.ene_vec])
       return ediis_linear + np.einsum("i,ij->j", x, self.ediis_quadratic)

    def ediis_coefficients(self):
       from scipy.optimize import minimize
       self.ediis_populate()
       result = minimize(self.ediis_energy, np.ones(len(self.f_list)), method="SLSQP",
                         bounds = tuple((0, 1) for i in self.f_list),
                         constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1, "jac": lambda x: np.ones_like(x)}],
                         jac=self.ediis_gradient, tol=5e-6, options={"maxiter": 200})

       if not result.success:
           raise Exception("EDIIS minimization failed")

       return result.x

    def extrapolate(self):
       ## check vectors lenght w.r.t  maxvec
       # Limit size of DIIS vector
       diis_count = len(self.ene_vec)
       
       if diis_count == 0:
           raise Exception("DIIS: No previous vectors.")
       if diis_count == 1:
           return self.f_list[0]   #return the fock
       
       if diis_count > self.maxvec:
           # Remove oldest vector
           del self.ene_vec[0]
           del self.f_list[0]
           del self.d_list[0]
           diis_count -= 1
       
       if self.__engine == 'e_diis':
          coef = self.ediis_coefficients()

       elif self.__engine == 'a_diis':
          coef = self.adiis_coefficients()
       
       # store coeff
       self.__ccoef = coef
       V = np.zeros_like(self.f_list[-1])
       for num,c in enumerate(coef):
           V +=c*self.f_list[num]
       return V   
    
    def get_energy(self):
       coef = self.__ccoef 
       if self.__engine == 'e_diis':
          res = self.ediis_energy(coef)
       elif self.__engine == 'a_diis':
          res = self.adiis_energy(coef)
       return res
   #################################
   # A-DIIS methods

    def adiis_energy(self, x):
        # self.ene_vec[-1]
        return  np.dot(self.adiis_linear, x) + np.einsum("i,ij,j->", x, self.adiis_quadratic, x)/2. #+ self.ene_vec[-1]  

    def adiis_gradient(self, x):
        return self.adiis_linear + np.einsum("i,ij->j", x, self.adiis_quadratic)

    def adiis_coefficients(self):
        from scipy.optimize import minimize
        self.adiis_populate()
        result = minimize(self.adiis_energy, np.ones(len(self.f_list)), method="SLSQP",
                          bounds = tuple((0, 1) for i in self.f_list),
                          constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1, "jac": lambda x: np.ones_like(x)}],
                          jac=self.adiis_gradient, tol=5e-6, options={"maxiter": 200})

        if not result.success:
            raise Exception("ADIIS minimization failed. File a bug, and include your entire input and output files.")

        return result.x

    def adiis_populate(self):
        num_entries=len(self.f_list)
 
        # deltas of density and Focks -> (D_i - D_n) and (F_i -F_n)
        dD = [[] for x in range(num_entries)]
        dF = [[] for x in range(num_entries)]
        for container, array in zip([self.d_list,self.f_list], [dD, dF]):
            for item_num in range(len(container)):
                latest_entry = container[num_entries-1][item_num]  #select last entry by row
                for entry_num in range(num_entries):
                    temp = np.copy(container[entry_num][item_num]) # list are mutable, make a copy
                    temp -=latest_entry
                    array[entry_num].append(temp)
        self.adiis_linear = np.zeros(num_entries)
        latest_fock = []
        for item_num in range(len(self.f_list)):
            latest_fock.append( self.f_list[len(self.f_list)-1][item_num] )
        for i in range(num_entries):
            self.adiis_linear[i] = sum(d.dot(f) for d,f in zip(dD[i],latest_fock) )
        #quadratic
 
        self.adiis_quadratic = np.zeros((num_entries, num_entries))
        for i, j in product(range(num_entries), repeat = 2):
            self.adiis_quadratic[i][j] = sum(d.dot(f) for d, f in zip(dD[i], dF[j]))
        if self.closed_shell:
            self.adiis_quadratic *=2
            self.adiis_linear *= 2



