import util
import psi4 
molA =  util.Molecule("../samples/H2O_sym/H2O1.xyz")

#set arbitrary charge and multiuplicity '2S+1'

chargeA = 1
multA = 2
molA.set_charge(chargeA,multA)

