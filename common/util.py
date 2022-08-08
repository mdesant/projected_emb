"""
Aux function for input processing
"""
__authors__   =  "Matteo De Santis"
__credits__   =  ["Matteo De Santis"]

__copyright__ = "(c) 2020, MDS"
__license__   = "BSD-3-Clause"
__date__      = "2020-03-01"

import sys
import os

import numpy as np
import psi4
from pkg_resources import parse_version
##################################################################

def set_input(fgeom):
  geomobj = str()
  with open(fgeom,"r") as f:
   numat=f.readline()
   species=f.readline()
   for line in f:
    geomobj +=str(line)
  geomobj += "symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com"
  print("GEOMETRY in angstrom:\n")
  print(geomobj)
  mol =psi4.geometry(geomobj)
  f.close()
  return geomobj, mol,numat,species


class Molecule():

  def __init__(self,fgeom='geom.xyz',label=False):

      self.__geometry = None
      self.__natom = None
      self.__labels = None
      self.__status_on = None
      self.set_geometry(fgeom,label)
      self.get_labels(fgeom,label)

  def set_geometry(self,fgeom,label=False):

      self.__geometry=str()
      with open(fgeom,'r') as data:
         self.__natom = int(data.readline()) # natom is the 'proper' 
                                           # number of atoms in the 
                                           # (active) molecule
         next(data)
         for line in data:
            self.__geometry += str(line)
      if label:
         labeled = str()
         tmp=self.__geometry.split('\n')
         tmp.pop()
         for m in tmp:
             labeled += str(m.split()[0]) +'1' + '   ' + str(m.split()[1])  + '   ' + str(m.split()[2]) + '   ' + str(m.split()[3])+'\n'
         self.__geometry = labeled

  def get_labels(self,fgeom,label):

      self.__labels=[]
      with open(fgeom,'r') as data:
         next(data)
         next(data)
         for line in data:
           tmp = line.split()
           if label :
             self.__labels.append(tmp[0]+'1')
           else:
             self.__labels.append(tmp[0])
         self.__labels = list(dict.fromkeys(self.__labels))

  def get_ghost(self,idx1,idx0=0):
      gdummy = str()
      tmp=self.__geometry.split('\n')
      tmp.pop()
      for m in tmp[idx0:idx1]:
        gdummy +="@"+m.strip()+'\n'
      return gdummy
  def append(self,options):
      # options is a string like "symmetry c1"+"\n" or a 
      # string containing moelcular coordinates
      self.__geometry += options

  def set_charge(self,charge,mult=1):
      self.__geometry+=str(charge)+' '+str(mult) + '\n'

  def display_xyz(self):
      print(self.__geometry)

  def natom(self):
      return self.__natom
  def geometry(self):
      return self.__geometry

  def labels(self):
      return self.__labels
  def core_geom(self,idx1,idx0=0):

         res = str()
         tmp=self.__geometry.split('\n')

         for m in tmp[idx0:idx1]:
            res += m+'\n'
         return res
