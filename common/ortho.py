from numpy.linalg import norm
import numpy as np

# Julia implementation of modified gram-schmidt
# function modified_gram_schmidt(matrix)
#     # orthogonalises the columns of the input matrix
#     num_vectors = size(matrix)[2]
#     orth_matrix = copy(matrix)
#     for vec_idx = 1:num_vectors
#         orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
#         for span_base_idx = (vec_idx+1):num_vectors
#             # perform block step
#             orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
#         end
#     end
#     return orth_matrix
# end

def modified_gram_schmidt(matrix,test=False):
   #orthonalize the columns
   num_vectors = matrix.shape[1]
   orth_matrix = np.copy(matrix)
   for vec_idx in range(num_vectors):
       orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
    
       for span_base_idx in range(vec_idx+1,num_vectors):
           #perform block step
           orth_matrix[:, span_base_idx] -= np.conjugate(orth_matrix[:, span_base_idx]).dot(orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]

   if test:
      norm_list = []
      if matrix.shape[0] != num_vectors:
         for col_idx in range(num_vectors):
            tmp_list = []
            for base_idx in range(col_idx+1,num_vectors):
                tmp_list.append(np.conjugate(orth_matrix[:,base_idx]).dot(orth_matrix[:,col_idx]))
            norm_list += tmp_list
         check_vec = np.array(norm_list)
         res = np.allclose(check_vec,np.zeros( check_vec.shape[0] ))
      else: 
         ortho_check = np.matmul(orth_matrix,orth_matrix.T)
         res = np.allclose(np.eye(num_vectors),ortho_check) 

      return orth_matrix,res

   else:
      return orth_matrix

if __name__ == "__main__":
   print("Test a 5x5 matrix")   
   #define a matrix (square)
   matvec = np.random.rand(5,5)
   #check the column vector to be linearly independent
   det = np.linalg.det(matvec) 
   print("det|A| : %.8f" % det)
   
   try:
     mat_inv = np.linalg.inv(matvec)
   except np.linalg.LinAlgError:
     print("Error in numpy.linalg.inv of inputted matrix")
   
   print("Check matrix orthonality ...")
   print(np.allclose(mat_inv,matvec.T))

   if det == 0.0:
     exit()

   print("start orthogonalization ..")
   res,err = modified_gram_schmidt(matvec,test=True)
   det = np.linalg.det(matvec) 
   print("det|G| (orthogonal) : %.8f" % det)
   print("Check matrix orthonality ...")
   #test = np.matmul(res,res.T)
   #print("OO^T = 1 : %s" % np.allclose(np.eye(5),test))
   print("OO^T = 1 : %s" % err)
   try:
     mat_inv = np.linalg.inv(res)
   except np.linalg.LinAlgError:
     print("Error in numpy.linalg.inv of inputted matrix")
   print("O^T = O^-1 : %s" % np.allclose(mat_inv,res.T))
  
   print("Test rectangular matrix:")

   matvec = matvec[:,:4] 
   print("A(%i,%i)" % matvec.shape)
   res,err = modified_gram_schmidt(matvec,test=True)
   print("Orthogonality check : %s" % err)
