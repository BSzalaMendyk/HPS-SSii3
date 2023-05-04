#!/usr/bin/env python
# # LAMMPS to HOOMD conversion
# Original version: Hamed
# Modified to output python runscripts (HPS and KH) for HOOMD: Roshan 
# Updated by Azamat
# Updated by Beata:
#-------- Include HPS-SSii+3 model: 
#---------------- i,i+3 special pair interactions with both 12-10 and 12-6 potential forms
#---------------- i,i+4 exclusions for non-bonded interactions
#-------- Updated for MDanalysis v2.2.0 and HOOMD-blue v2.9.7: all types (particles, bonds, angles...) must be provided as lists not arrays

import re
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import argparse
import MDAnalysis as mda
import hoomd, hoomd.md, gsd, gsd.hoomd
import sys, os


parser = argparse.ArgumentParser('LAMMPS/PARAMBUILDER to HOOMD conversion file')
# required, positional arguments
parser.add_argument('datafile', type=argparse.FileType('r'), help='LAMMPS data file, e.g. in.data')
parser.add_argument('lammpsfile', type=argparse.FileType('r'), help='LAMMPS input file. e.g. test.lmp')
parser.add_argument('paramfile', type=argparse.FileType('r'), help='Parambuilder input file, e.g. param.in, helps to identify the rigid domains of the chain')
#optional arguments
parser.add_argument('-momentfile', type=argparse.FileType('r'), help='Moment of inertia file from parambuilder (-hoomd), e.g. in_moments.txt, it only needs in case of rigid body simulation')
parser.add_argument('-outfile', default='runfile.py', type=argparse.FileType('w'), help='Outfile file, which is HOOMD input file')
parser.add_argument('-integrator', type=int, default=2, help="Choice of integrator/operation (1) NPT (2) Slab + NVT/Langevin (3) Energy min. (default 2) (4) Resizing (5) Energy min. after Resizing : ")
parser.add_argument('-runtime', type=int, default=500000000, help="Simulation runtime, default is 5e8")
parser.add_argument('-temp', type=int, default=300, help="Temperature, default is 300K")
parser.add_argument('-box', type=int, default=150, help="Simulation box size for Resizing option, default is 150 Angstrom")
args = parser.parse_args()

#This conversion script takes care of both flexible and rigid body simulation
#This part checks if the chain is fully flexible, partially rigid or fully rigid
#takes the rigid group ranges from param input file
param_file = args.paramfile
rigid_range_i= []
rigid_range_f= []
with param_file as myfile:
    lines = myfile.readlines()
    for line in lines:
        if re.search(r'rigid_group', line):
            s = line.split()[3]
            s = re.split(':',s)
            rigid_range_i.append(int(s[0])-1)
            rigid_range_f.append(int(s[1]))
print(rigid_range_i)
print(rigid_range_f)


#functions to get the parameters of lj_lambda potential (HPS_URRY)
def lj_lambda(lines, pair_type, pair):
    type1, type2, epsilon, sigma, lam, rcut=[],[], [], [], [], []
    for line in lines:
        if(pair_type==1):
            if(line.split()[0]=="pair_coeff"):
                type1.append(line.split()[1])
                type2.append(line.split()[2])
                epsilon.append(line.split()[3])
                sigma.append(line.split()[4])
                lam.append(line.split()[5])
                rcut.append(line.split()[6])
                
        if(pair_type>1):
            if(line.split()[0]=="pair_coeff" and line.split()[3]==pair):
                type1.append(line.split()[1])
                type2.append(line.split()[2])
                epsilon.append(line.split()[4])
                sigma.append(line.split()[5])
                lam.append(line.split()[6])
                rcut.append(line.split()[7])
    return type1, type2, epsilon, sigma, lam, rcut
def kh(lines, pair_type, pair):
    type1, type2, epsilon, sigma, lam, rcut=[],[], [], [], [], []
    for line in lines:
        if(pair_type==1):
            if(line.split()[0]=="pair_coeff"):
                type1.append(line.split()[1])
                type2.append(line.split()[2])
                epsilon.append(line.split()[3])
                if float(line.split()[3])<=0:
                    lam.append(1)
                elif float(line.split()[3])>0:
                    lam.append(-1)
                sigma.append(line.split()[4])
                rcut.append(line.split()[5])
                
        if(pair_type>1):
            if(line.split()[0]=="pair_coeff" and line.split()[3]==pair):
                type1.append(line.split()[1])
                type2.append(line.split()[2])
                epsilon.append(line.split()[4])
                if float(line.split()[4])<0:
                    lam.append(1)
                elif float(line.split()[4])>0:
                    lam.append(-1)
                sigma.append(line.split()[5])
                rcut.append(line.split()[6])
    return type1, type2, epsilon, sigma, lam, rcut


#functions to get the bond potential parameters
def harmonic(lines, bond_types, bond):
    bond_type,k,r0=[],[], []
    for line in lines:
        if(bond_types==1):
            if(line.split()[0]=="bond_coeff"):
                bond_type.append(line.split()[2])
                k.append(line.split()[3])
                r0.append(line.split()[4])
                
        if(bond_types>1):
            if(line.split()[0]=="bond_coeff" and line.split()[2]==bond):
                bond_type.append(line.split()[2])
                k.append(line.split()[3])
                r0.append(line.split()[4])
    return bond_type,k,r0

#angle function
def angle_pot(lines, angle_types, angle):
    angle_type,theta0=[],[]
    for line in lines:
        if(angle_types==1):
            if(line.split()[0]=="angle_coeff"):
                angle_type.append(line.split()[1])
                theta0.append(line.split()[2])
                
        if(angle_types>1):
            if(line.split()[0]=="angle_coeff" and line.split()[1]==angle):
                angle_type.append(line.split()[1])
                theta0.append(line.split()[2])
    return angle_type,theta0

def dihed_pot(lines, dihed_types, dihed_angle):
    dihed_type,eps_d=[],[]
    for line in lines:
        if(dihed_types==1):
            if(line.split()[0]=="dihedral_coeff"):
                dihed_type.append(line.split()[1])
                eps_d.append(line.split()[-1])

        if(dihed_types>1):
            if(line.split()[0]=="dihedral_coeff" and line.split()[1]==dihedral):
                dihed_type.append(line.split()[1])
                eps_d.append(line.split()[-1])
    #print(dihed_type)
    return dihed_type,eps_d

def ii3_specpair(lines, dihed_types, dihedral):
    ii3_type, eps_ii3, sigm_ii3, lambd_ii3 =[],[],[],[]
    for line in lines:
        if(dihed_types==1):
            if(line.split()[0]=="dihedral_coeff"):
                ii3_type.append(line.split()[1])
                eps_ii3.append(float(line.split()[2]))
                sigm_ii3.append(float(line.split()[3]))
                lambd_ii3.append(float(line.split()[4]))
        if(dihed_types>1):
            if(line.split()[0]=="dihedral_coeff" and line.split()[1]==dihedral):
                ii3_type.append(line.split()[1])
                eps_ii3.append(float(line.split()[2]))
                sigm_ii3.append(float(line.split()[3]))
                lambd_ii3.append(float(line.split()[4]))
    #print(dihed_type)
    return ii3_type, eps_ii3, sigm_ii3, lambd_ii3

#function to fix the pbc
def fix_pbc3d(x,l_box):
    l_box_half=l_box/2.0
    x=x-l_box*np.int_(x/l_box_half)
    return x


#lammps to hoomd conversion factors
length_ratio = 1
en_ratio=1
charge_ratio = 332.0637/80
angle_ratio = 3.14159265359/180 # angle_pot (to convert it into radians)

# reading input files and extract required informtion
#first it takes data file generated via parambuilder and converts it to gsd format for HOOMD
lammpsfile = args.datafile
univ = mda.Universe(lammpsfile, atom_style='id resid type charge x y z', bond_style='id ')
s = gsd.hoomd.Snapshot()

#particles information
types,indices = np.unique(univ.atoms.types, return_index=True) # id of unique types of particles
types=univ.atoms.types[sorted(indices)]
s.particles.types=types.tolist()

atomtype=np.zeros(len(univ.atoms.types))
#chainging the id of particles from original ids to numbers from 0 to n_type-1 
for i in range(0,len(s.particles.types)):
    idx=np.where(univ.atoms.types==s.particles.types[i])
    atomtype[idx]=i
nc=len(univ.residues.resnums)

#particles id start from zero as opposed to lammps: and the first and last particles in the range belong to rigid body. Thus  rigid_range_i should be rigid_range_i-1. However, the upper limit should be similar to lammmps (the way that python defines range)
type_com_rigid = np.where(np.int_(types)==1)
type_com_rigid = types[:type_com_rigid[0][0]]
nrigid_pm=len(type_com_rigid)
nrigid=nrigid_pm*nc
npc=len(np.where(univ.residues.atoms.resindices==0)[0])-nrigid_pm

atomid = np.zeros(len(univ.atoms.types))
for i in range(nrigid_pm):
    idx=np.where(univ.atoms.types==s.particles.types[i])
    atomid[idx]=i

for i in range(nrigid,univ.atoms.n_atoms):
    atomid[i] = int(univ.atoms.types[i]) + nrigid_pm -1

lbox=univ.dimensions[0:3]
s.particles.N = univ.atoms.n_atoms    
s.particles.typeid = np.int_(atomtype)
s.particles.mass = univ.atoms.masses
s.particles.charge = univ.atoms.charges
s.particles.position = univ.atoms.positions * length_ratio
s.particles.body=np.zeros(s.particles.N)-1
s.particles.moment_inertia=np.zeros((s.particles.N,3))
s.particles.orientation=np.zeros((s.particles.N,4))
s.particles.orientation[:,:]=np.array([1, 0., 0., 0.])
simulation_box_coordinate=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

poslist = []
seqlist = []
center_mass = []
#creates list of types of constituent particles if rigid-body simulation
seqfile = open('seqlist.txt','w')
for i in range(nc):
    for j in range(nrigid_pm):
        s.particles.body[i*nrigid_pm+j]=i*nrigid_pm+j
        s.particles.body[nrigid+i*npc+rigid_range_i[j]:nrigid+i*npc+rigid_range_f[j]]=j*nc+i
        rigid_pos_temp=s.particles.position[nrigid+i*npc+rigid_range_i[j]:nrigid+i*npc+rigid_range_f[j]]
        rigid_mass_temp=s.particles.mass[nrigid+i*npc+rigid_range_i[j]:nrigid+i*npc+rigid_range_f[j]]
        rigid_body_com_temp=s.particles.position[nc*j+i*nrigid_pm]
        if i==0:
            #writing the list of positions of constituent particles
            poslist.append(rigid_pos_temp)
            #writing list of the center of mass of central particles
            center_mass.append(rigid_body_com_temp)
            for at in range(nrigid+i*npc+rigid_range_i[j],nrigid+i*npc+rigid_range_f[j]):
                seqfile.write(" "+str(np.int_(atomid)[at]-nrigid_pm+1)+" ")
            seqfile.write("\n")
        fix_pbc3d(rigid_pos_temp,lbox)
        s.particles.orientation[j*nc+i,:]=[1,0,0,0]
#List of positions of constituent particles
poslist = np.array(poslist)
np.save('poslist.npy',poslist)
#Lst of the center of mass of central particles
center_mass = np.array(center_mass)
np.save('center_mass.npy',center_mass)

#List of the center of moment of inertia of central particles, if rigid body simulation
#if flexible, there is no need to write the moment of inertia, it just prints: flexible chain
if len(rigid_range_i)!=0:
    moment_file = args.momentfile
    line_number = -1
    i = -1
    list_of_results = []
    with moment_file as myfile:
        lines = myfile.readlines()
        for line in lines:
            # For each line, check if line contains the string
            line_number += 1
            if 'eigenvalue' in line:
                i += 1
                mom_int = lines[line_number+1]
                x = mom_int.split('    ')
                s.particles.moment_inertia[i] = [int(float(x[1])),int(float(x[2])),int(float(x[3]))]
else:
    print('flexible chain')

#reading and removing the blank line
infile = args.lammpsfile
f=infile
data=f.readlines()
f.close()
lines = (line.rstrip() for line in data)
lines = list((line for line in lines if line))

#getting special bond information from lammps input
for line in lines:
    if(line.rstrip().split()[0]=="bond_style"):
        bond_style=line.rstrip().split()[1:]
        #print(pair_style)
        bond_style=" ".join(bond_style)
bond_style = re.sub(r'\b[0-9,.]+\b\s*', '', bond_style).split()
if 'hybrid' in bond_style:
    bond_style.remove('hybrid')
bond_types=int(len(bond_style))

#bonds information
s.bonds.group=[]
s.bonds.typeid=[]
s.bonds.types=[]
s.bonds.N=0
if 'special' in bond_style:
    xid=np.where((np.array(bond_style))=="special")[0][0]+1
    for bnd in univ.bonds:
        if not bnd.type == str(xid):
            s.bonds.typeid.append(0)
            s.bonds.group.append(bnd.indices)
            s.bonds.N += 1
            if not bnd.type in s.bonds.types:
                s.bonds.types.append(bnd.type)
            print(bnd.indices)
else:
    s.bonds.group=univ.bonds.indices
    s.bonds.typeid=(np.array(univ.bonds._bondtypes,dtype=np.int)-1).tolist()
    s.bonds.types=list(univ.bonds.topDict.dict.keys())
    s.bonds.N=len(univ.bonds._bondtypes)
#print(s.bonds.types)

#angles information
s.angles.group=univ.angles.indices
s.angles.typeid=(np.array(univ.angles._bondtypes,dtype=np.int)-1).tolist()
s.angles.types=list(univ.angles.topDict.dict.keys())
s.angles.N=len(univ.angles._bondtypes)
#print(s.angles.typeid)
#print(univ.angles)
#print(s.angles.types)

#dihedral angles information
s.dihedrals.group=univ.dihedrals.indices
#s.dihedrals.typeid=np.linspace(0,len(univ.dihedrals._bondtypes)-1,len(univ.dihedrals._bondtypes),dtype=np.int)
#s.dihedrals.types=list(univ.dihedrals._bondtypes)
s.dihedrals.typeid=(np.array(univ.dihedrals._bondtypes,dtype=np.int)-1).tolist()
s.dihedrals.types=list(univ.dihedrals.topDict.dict.keys())
s.dihedrals.N=len(univ.dihedrals._bondtypes)
for dih_typ in univ.dihedrals._bondtypes:
    if not dih_typ in s.dihedrals.types:
        s.dihedrals.types.append(dih_typ)
#s.dihedrals.types=s.dihedrals.types.sort()
# print('Dihedral types number: ',s.dihedrals.N)
# print(univ.dihedrals._bondtypes)
# print(univ.dihedrals.topDict['21'].types())
# print(univ.dihedrals.select_bonds('13').to_indices())
# print(univ.dihedrals.select_bonds('31').to_indices())
# print(univ.dihedrals.topDict.dict.keys())
# print(s.dihedrals.types)
# print(s.dihedrals.typeid)


#i,i+3 special pairs information
s.pairs.N = 0
#s.pairs.types = ['ii3']
s.pairs.types = []
s.pairs.typeid = []
#print(univ.atoms.n_residues)
#print(univ.atoms.residues)
#print(s.particles)
s.pairs.group = []
for group,typeid in zip(s.dihedrals.group,s.dihedrals.typeid):
    #atom_sel=univ.select_atoms('resid '+str(res.resid))
    if not 'ii3_'+str(typeid+1) in s.pairs.types:
        s.pairs.types.append('ii3_'+str(typeid+1))
    s.pairs.group.append([group[0],group[-1]])    # MDAnalysis starts indexing from 1
    s.pairs.typeid.append(typeid)
    s.pairs.N += 1
    #for idx in range(atom_sel.n_atoms-3):
    #for atom in atom_sel:
#        print(atom_sel[idx].id)
#         s.pairs.types.append('ii3_'+str(atom_sel[idx].id)+'-'+str(atom_sel[idx].id+3))
#         s.pairs.group.append([atom_sel[idx].id-1,atom_sel[idx].id+2])    # MDAnalysis starts indexing from 1
#         s.pairs.typeid.append(0)
#         s.pairs.N += 1


#box info
s.configuration.dimensions=3
s.configuration.box=[univ.coord.dimensions[0]*length_ratio,univ.coord.dimensions[1]*length_ratio,univ.coord.dimensions[2]*length_ratio,0,0,0]
s.configuration.step=0

#save gsd file
f = gsd.hoomd.open(name='converted.gsd', mode='wb')
f.append(s)
f.close()

#This part write the input file for HOOMD

#converting the pair interactions
outfile=args.outfile
f=outfile
f.write('#importing the libraries\nimport sys, os, numpy as np\nimport hoomd, hoomd.md as md\nfrom hoomd import azplugins\nimport gsd, gsd.hoomd, gsd.pygsd\n')

# RUN PARAMETERS  
runtime = args.runtime
f.write("nsteps = %s #simulation runtime length\n"%runtime)
temp=args.temp
f.write("\ntemp=%s #simulation temperature\n"%temp)
f.write("\nhoomd.context.initialize() #Initialize the execution context\n")
f.write("system = hoomd.init.read_gsd(sys.argv[1]) #Read initial system state from an GSD file.\n")

# CHECK FOR RIGID BODIES, If yes adding md.constrain.rigid to constrain particles in rigid bodies.
if len(rigid_range_i)!=0:
    f.write("\n### RIGID BODIES ###\n")
    f.write("center_mass=np.load('center_mass.npy', allow_pickle=True) #center of mass of center particles\n")
    f.write("poslist=np.load('poslist.npy', allow_pickle=True) #positions of constituent particles\n")
    f.write("seqfile = open('seqlist.txt').readlines() #types of constituent particles\n")
    f.write("rigid = hoomd.md.constrain.rigid() #Constrain particles for each type of central particles\n")
    f.write("for i in range(poslist.shape[0]):\n")
    f.write("   rigid.set_param(system.particles[%d*i].type,positions=poslist[i]-center_mass[i],types=seqfile[i].split(" "))\n"%(nc))
    f.write("rigid.create_bodies(False)\n")

f.write("\nnl = hoomd.md.nlist.cell() #Cell list based neighbor list\n")


def my_abs(value):
    """Returns absolute value without using abs function"""
    if value <= 0:
        return value * -1
    return value * 1

#getting bond information
for line in lines:
    if(line.rstrip().split()[0]=="bond_style"):
        bond_style=line.rstrip().split()[1:]
        #print(pair_style)
        bond_style=" ".join(bond_style)
bond_style = re.sub(r'\b[0-9,.]+\b\s*', '', bond_style).split()
if 'hybrid' in bond_style:
    bond_style.remove('hybrid')
bond_types=int(len(bond_style))
f.write('#### BOND DATA ####\n')
f.write('harmonic = hoomd.md.bond.harmonic() #Harmonic bond potential and its parameters\n')
if("harmonic" in bond_style):
    xid=np.where((np.array(bond_style))=="harmonic")
    bond_type,k,r0=harmonic(lines,bond_types, np.array(bond_style)[xid]) 
    for i in range(0,len(bond_type)):
        f.write("harmonic.bond_coeff.set('%d', k=%lf, r0=%lf)\n"%(xid[0][0]+1,2*float(k[i])*en_ratio/length_ratio/length_ratio,float(r0[i])*length_ratio))
        continue

for line in lines:
    #print(line.rstrip())
    if(line.rstrip().split()[0]=="angle_style"):
        angle_style=line.rstrip().split()[1:]
        #    #print(pair_style)
        angle_style=" ".join(angle_style)
angle_style = re.sub(r'\b[0-9,.]+\b\s*', '', angle_style).split()
angle_types=int(len(angle_style))
#print(angle_style)

f.write("\n")
f.write('#### ANGLE DATA ####\n')

f.write('def bch_angle_pot(theta,e_alpha):\n')
f.write('   gamma = 0.1\n')
f.write('   t_alpha = 1.6\n')
f.write('   t_beta = 2.27\n')
f.write('   k_a = 106.4\n')
f.write('   k_b = 26.3\n')
f.write('   V = -np.log(np.exp(-gamma*(k_a*(theta-t_alpha)**2+e_alpha)) + np.exp(-gamma*k_b*(theta-t_beta)**2))/gamma\n')
f.write('   T = (-2*gamma*k_a*(theta - t_alpha)*np.exp(-gamma*(e_alpha + k_a*(theta - t_alpha)**2)) - 2*gamma*k_b*(theta-t_beta)*np.exp(-gamma*k_b*(theta - t_beta)**2)) / (gamma*(np.exp(-gamma*(e_alpha + k_a*(theta - t_alpha)**2)) + np.exp(-gamma*k_b*(theta - t_beta)**2)))\n')
f.write('   return (V,T)\n')

if("bch" in angle_style):
    xid=np.where((np.array(angle_style))=="bch")
    #print(xid,np.array(angle_style)[xid])
    angle_type,theta0=angle_pot(lines,angle_types, np.array(angle_style)[xid])
    f.write("bch_table = hoomd.md.angle.table(width=1000)\n")
    for i in range(0,len(angle_type)):
        f.write("bch_table.angle_coeff.set('%s', func=bch_angle_pot, coeff=dict(e_alpha=%lf))\n"%(angle_type[i],float(theta0[i])))
        continue
e0=''
e1=''
e2=''
for line in lines:
    #print(line.rstrip())
    if(line.rstrip().split()[0]=="dihedral_style"):
        dihedral_style=line.rstrip().split()[1:]
        dihedral_style = dihedral_style[0:dihedral_style.index('#')]
        if '/lj1210lambda' or '/lambda' in dihedral_style[0]:
            e0,e1,e2 = dihedral_style[1],dihedral_style[2],dihedral_style[3]
            dihedral_style = dihedral_style[0:1]
        #print(pair_style)
        dihedral_style=" ".join(dihedral_style)
dihedral_style = re.sub(r'\b[0-9,.]+\b\s*', '', dihedral_style).split()
dihed_types=int(len(dihedral_style))


f.write("\n")
f.write('#### DIHEDRAL DATA ####\n')
f.write('def dihed_angle_pot(theta,eps_d):\n')
f.write('   ka1=11.4\n')
f.write('   ka2=0.15\n')
f.write('   kb1=1.8\n')
f.write('   kb2=0.65\n')
f.write('   fa1=0.9\n')
f.write('   fa2=1.02\n')
f.write('   fb1=-1.55\n')
f.write('   fb2=-2.5\n')
f.write('   e0 = '+e0+'\n')
f.write('   e1 = '+e1+'\n')
f.write('   e2 = '+e2+'\n')
f.write('   pi=np.pi\n')
f.write('   Ua=np.exp(-ka1*(theta-fa1)**2-eps_d)+np.exp(-ka2*(theta-fa2)**4+e0)+np.exp(-ka2*(theta-fa2+2*pi)**4+e0)\n')
f.write('   Ub=np.exp(-kb1*(theta-fb1)**2+e1+eps_d)+np.exp(-kb1*(theta-fb1-2*pi)**2+e1+eps_d)+np.exp(-kb2*(theta-fb2)**4+e2)+np.exp(-kb2*(theta-fb2-2*pi)**4+e2)\n')
f.write('   V = -np.log(Ua+Ub)\n')
f.write('   T =(-2*ka1*(theta-fa1)*np.exp(-eps_d-ka1*(theta-fa1)**2)-4*ka2*(theta-fa2)**3*np.exp(e0-ka2*(theta-fa2)**4)-4*ka2*(-fa2+theta-2*pi)**3*np.exp(e0-ka2*(-fa2+theta-2*pi)**4)-2*kb1*(theta-fb1)*np.exp(eps_d-kb1*(theta-fb1)**2+e1)-2*kb1*(-fb1+theta-2*pi)*np.exp(eps_d-kb1*(-fb1+theta-2*pi)**2+e1)-4*kb2*(theta-fb2)**3*np.exp(e2-kb2*(theta-fb2)**4)-4*kb2*(-fb2+theta-2*pi)**3*np.exp(e2-kb2*(-fb2+theta-2*pi)**4))/(np.exp(-eps_d-ka1*(theta-fa1)**2)+np.exp(e0-ka2*(theta-fa2)**4)+np.exp(e0-ka2*(-fa2+theta-2*pi)**4)+np.exp(eps_d-kb1*(theta-fb1)**2+e1)+np.exp(eps_d-kb1*(-fb1+theta-2*pi)**2+e1)+np.exp(e2-kb2*(theta-fb2)**4)+np.exp(e2-kb2*(-fb2+theta-2*pi)**4))\n')
f.write('   return (V,T)\n')
f.write("\n")

if("gaussian/ljlambda" in dihedral_style):
    xid=np.where((np.array(dihedral_style))=="gaussian/ljlambda")
    dihed_type,eps_d = dihed_pot(lines,dihed_types, np.array(dihedral_style)[xid])
    f.write("dihed = hoomd.md.dihedral.table(width=1000)\n")
    for i in range(0,len(dihed_type)):
        f.write("dihed.dihedral_coeff.set('%s', func=dihed_angle_pot, coeff=dict(eps_d=%lf))\n"%(dihed_type[i],float(eps_d[i])))
        #continue
    ii3_type, eps_ii3, sigm_ii3, lambd_ii3 = ii3_specpair(lines,dihed_types, np.array(dihedral_style)[xid])
    f.write("\n# SPECIAL PAIR INTERACTION\n")
    f.write("ii3 = hoomd.md.special_pair.ii3_126(name=\"ii3_pairs\")\n")
    for idx in range(len(ii3_type)):
        f.write("ii3.pair_coeff.set(\'ii3_%s\',lambd=%lf, sigma=%lf,eps=%lf) \n"%(ii3_type[idx],lambd_ii3[idx],sigm_ii3[idx],eps_ii3[idx]))
        
if("gaussian/lj1210lambda" in dihedral_style):
    xid=np.where((np.array(dihedral_style))=="gaussian/ljlambda")
    dihed_type,eps_d = dihed_pot(lines,dihed_types, np.array(dihedral_style)[xid])
    f.write("dihed = hoomd.md.dihedral.table(width=1000)\n")
    for i in range(0,len(dihed_type)):
        f.write("dihed.dihedral_coeff.set('%s', func=dihed_angle_pot, coeff=dict(eps_d=%lf))\n"%(dihed_type[i],float(eps_d[i])))
        #continue
    ii3_type, eps_ii3, sigm_ii3, lambd_ii3 = ii3_specpair(lines,dihed_types, np.array(dihedral_style)[xid])
    f.write("\n# SPECIAL PAIR INTERACTION\n")
    f.write("ii3 = hoomd.md.special_pair.ii3_1210(name=\"ii3_pairs\")\n")
    for idx in range(len(ii3_type)):
        f.write("ii3.pair_coeff.set(\'ii3_%s\',lambd=%lf, sigma=%lf,eps=%lf) \n"%(ii3_type[idx],lambd_ii3[idx],sigm_ii3[idx],eps_ii3[idx]))

#getting the pair styles  
for line in lines:
    if(line.rstrip().split()[0]=="pair_style"):
        pair_style=line.rstrip().split()[1:]
        pair_style=" ".join(pair_style)
pair_style = re.sub(r'\b[0-9,.]+\b\s*', '', pair_style).split()
pair_types=len(pair_style)
pair_type=1
f.write("\nnl.reset_exclusions(exclusions=['1-2', '1-3','1-4','body']) #setting the exclusions from short range pair interactions\n")
if 'special' in bond_style:
    xid=np.where((np.array(bond_style))=="special")[0][0]+1
    f.write('#### SPECIAL EXCLUSIONS DATA ####\n')
    lammpsdata=args.datafile.readlines()
    lammpsdata = lammpsdata[lammpsdata.index(' Bonds\n'):lammpsdata.index(' Angles\n')]
    for line in lammpsdata:
        line = line.rstrip().split()
        if len(line) > 1 and line[1]==str(xid):
            f.write("nl.add_exclusion(%d,%d)\n"%(int(line[2])-1,int(line[3])-1))
    f.write('\n')
f.write("nb = azplugins.pair.ashbaugh(r_cut=0, nlist=nl) #Ashbaugh-Hatch potential and its parameters\n")
pairstyles=['ljlambda','kh/cut/coul/debye']
for styl in pairstyles:
    if(styl in pair_style):
        if styl=='ljlambda':
            xid=np.where((np.array(pair_style))==styl)
            type1, type2, epsilon, sigma, lam, rcut=[],[],[],[],[],[]
            type1, type2, epsilon, sigma, lam, rcut=lj_lambda(lines,pair_type, np.array(pair_style)[xid])
            for i in range(0,len(type1)):
                f.write("nb.pair_coeff.set('%s', '%s', epsilon = %lf, sigma = %lf, lam = %lf, r_cut = %lf)\n"%(type1[i], type2[i], float(epsilon[i])* en_ratio, float(sigma[i])* length_ratio, float(lam[i]), float(rcut[i])* length_ratio))
                continue
        if styl=='kh/cut/coul/debye':
            xid=np.where((np.array(pair_style))==styl)
            type1, type2, epsilon, sigma,lam, rcut=[],[],[],[],[],[]
            type1, type2, epsilon, sigma,lam, rcut=kh(lines,pair_type, np.array(pair_style)[xid])
            for i in range(0,len(type1)):
                f.write("nb.pair_coeff.set('%s', '%s', epsilon = %lf, sigma = %lf, lam = %lf, r_cut = %lf)\n"%(type1[i], type2[i], my_abs(float(epsilon[i])* en_ratio), float(sigma[i])* length_ratio, float(lam[i]), float(rcut[i])* length_ratio))
                continue

#electrostatics
type_charge=s.particles.charge[sorted(indices)]
f.write("\n### ELECTROSTATICS ###\n")
f.write("\nyukawa = hoomd.md.pair.yukawa(r_cut=0.0, nlist=nl)\n")
f.write("yukawa.pair_coeff.set(['%s'], ['%s'], epsilon=%lf, kappa=%lf, r_cut=%lf)\n"% ('\', \''.join(map(str,s.particles.types )), '\', \''.join(map(str, s.particles.types)), 0.0,0.0,0.0))
type_charge=np.round(type_charge,2)
for i in range(0, len(s.particles.types)):
    for j in range(0, len(s.particles.types)):
        if(type_charge[i]!=0.0 and type_charge[j]!=0.0):
            f.write("yukawa.pair_coeff.set('%s','%s', epsilon=%lf, kappa=%lf, r_cut=%lf)\n"% (s.particles.types[i], s.particles.types[j], charge_ratio*float(type_charge[i])*float(type_charge[j]) ,0.1,35.0))

# information for intergrator
f.write("\n### MAKE PARTICLE GROUPS ###\n")
f.write("all=hoomd.group.all()\n")
f.write("nonrigid = hoomd.group.nonrigid()\n")
f.write("rigid = hoomd.group.rigid()\n")
f.write("ghost = hoomd.group.rigid_center()\n")
f.write("integrate_group = hoomd.group.union(name='int_grp',a=nonrigid,b=ghost)\n")
f.write("outp_group = hoomd.group.difference(name='outp',a=all, b=ghost)")

# choose integrator 
choice=args.integrator
if choice==1:
    f.write("\n### NPT Integration ###\n")
    f.write("hoomd.md.integrate.mode_standard(dt=0.2045814)\n")
    f.write("integrator = hoomd.md.integrate.npt(group=integrate_group, kT=temp*0.0019872067,tau=500*20.4581492542,P=1.0*0.00023900573,tauP=5000*20.4581492542)\n")
    f.write("hoomd.analyze.log(filename='thermo.log', quantities=['potential_energy','pair_ashbaugh_energy', 'pair_yukawa_energy','bond_harmonic_energy','pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=1000000, overwrite=False, header_prefix='#')\n")
    f.write("hoomd.dump.gsd('prod.gsd',period=1000000, group=all,truncate=True)\n")
    f.write("hoomd.dump.dcd('prod.dcd',period=100000, group=outp_group, overwrite=False)\n")
    f.write("hoomd.run_upto(nsteps)\n")

elif choice==2:
    f.write("\n### Slab extension and NVT Integration ###\n")
    f.write("num_particles=len(system.particles)\n")
    f.write("print(num_particles)\n")
    f.write("position = np.zeros((num_particles,3))\n")
    f.write("#manipulating the snapshot to unwrap the particles in the desired dimension\n")
    f.write("for i in range(num_particles):\n")
    if len(rigid_range_i)!=0:
        f.write("   position[i,0] = system.particles[i].position[0] + system.box.Lx*system.particles[i].image[0]\n")
        f.write("   position[i,1] = system.particles[i].position[1] + system.box.Ly*system.particles[i].image[1]\n")
        f.write("   position[i,2] = system.particles[i].position[2] + system.box.Lz*system.particles[i].image[2]\n")
    else:
        f.write("   position[i,0] = system.particles[i].position[0]\n")
        f.write("   position[i,1] = system.particles[i].position[1]\n")
        f.write("   position[i,2] = system.particles[i].position[2] + system.box.Lz*system.particles[i].image[2]\n")
    f.write("snap = system.take_snapshot(all=True)\n")
    f.write("#updating the particles positions after unwrapping\n")
    f.write("for i in range(num_particles):\n")
    f.write("   snap.particles.position[i] = position[i]\n")
    f.write("# setting the new z box size in the snapshot\n")
    f.write("snap.box = hoomd.data.boxdim(Lx=system.box.Lx, Ly=system.box.Lx, Lz=system.box.Lx*7, xy=0, xz=0, yz=0)\n")
    f.write("system.restore_snapshot(snap)\n")
    
    f.write("hoomd.md.integrate.mode_standard(dt=0.2045814) #time step is 10fs\n")
    f.write("integrator = hoomd.md.integrate.langevin(group=integrate_group, kT=temp*0.0019872067,seed=12341) #Langevin integrator\n")
    f.write("#setting the friction factor for the Langevin integrator [mass/1000ps]\n")
    atomids=[]
    for i,val in enumerate(univ.atoms.types):
        if val not in atomids:
            atomids.append(val)
            f.write("integrator.set_gamma('%s',gamma=%f)\n"%(val,univ.atoms.masses[i]/1000*0.04889615))
    f.write("hoomd.analyze.log(filename='thermo.log', quantities=['potential_energy','pair_ashbaugh_energy', 'pair_yukawa_energy','bond_harmonic_energy', 'angle_table_energy','dihedral_table_energy', 'special_pair_iithree_energy','pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=1000000, overwrite=True, header_prefix='#') #Log a number of calculated quantities to a file.\n")
    f.write("writegsd=hoomd.dump.gsd('restart.gsd',period=1000000, group=all,overwrite=True,truncate=True,dynamic=['property', 'momentum']) #writes the simulation snapshot for restarting purposes\n")
    f.write("hoomd.dump.gsd('prod.gsd',period=100000, group=outp_group,overwrite=True,truncate=True,dynamic=['property', 'momentum']) #writes the last state of the simulation\n")
    f.write("hoomd.dump.dcd('prod.dcd',period=100000, group=outp_group, overwrite=True) #writes the trajectory of the simulation\n")
    f.write("hoomd.run_upto(nsteps) #Runs the simulation up to a given time step number.\n")
    f.write("writegsd.write_restart()\n")

elif choice==3:
    f.write("\n### NVT Integration only ###\n")
    f.write("hoomd.md.integrate.mode_standard(dt=0.2045814) #time step is 10fs\n")
    f.write("integrator = hoomd.md.integrate.langevin(group=integrate_group, kT=temp*0.0019872067,seed=12341) #Langevin integrator\n")
    f.write("#setting the friction factor for the Langevin integrator [mass/1000ps]\n")
    atomids=[]
    for i,val in enumerate(univ.atoms.types):
        if val not in atomids:
            atomids.append(val)
            f.write("integrator.set_gamma('%s',gamma=%f)\n"%(val,univ.atoms.masses[i]/1000/0.020455814925))
    f.write("hoomd.analyze.log(filename='thermo.log', quantities=['potential_energy','pair_ashbaugh_energy', 'pair_yukawa_energy','bond_harmonic_energy', 'angle_table_energy','dihedral_table_energy', 'special_pair_iithree_energy','pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=1000000, overwrite=True, header_prefix='#') #Log a number of calculated quantities to a file.\n")
    f.write("writegsd=hoomd.dump.gsd('restart.gsd',period=1000000, group=all,overwrite=True,truncate=True,dynamic=['property', 'momentum']) #writes the simulation snapshot for restarting purposes\n")
    f.write("hoomd.dump.gsd('prod.gsd',period=1000000, group=outp_group,overwrite=True,truncate=True,dynamic=['property', 'momentum']) #writes the last state of the simulation\n")
    f.write("hoomd.dump.dcd('prod.dcd',period=100000, group=outp_group, overwrite=True) #writes the trajectory of the simulation\n")
    f.write("hoomd.run_upto(nsteps) #Runs the simulation up to a given time step number.\n")
    f.write("writegsd.write_restart()\n")

elif choice==4:
    boxsize = args.box
    f.write("\n### NVT Integration + Resizing ###\n")
    #f.write("resize_steps=30000\n")
    f.write("hoomd.md.integrate.mode_standard(dt= 0.2045814/10) #time step is 1fs\n")
    f.write("hoomd.update.box_resize(L=hoomd.variant.linear_interp([(0,system.box.Lx),(nsteps-1,%s)]),scale_particles=True) #Linearly reducing the box size to %s Angstrom\n"%(boxsize,boxsize))
    f.write("integrator = hoomd.md.integrate.langevin(group=integrate_group, kT=temp*0.0019872067,seed=12341) #Langevin integrator\n")
    f.write("#setting the friction factor for the Langevin integrator [mass/1000ps]\n")
    atomids=[]
    for i,val in enumerate(univ.atoms.types):
        if val not in atomids:
            atomids.append(val)
            f.write("integrator.set_gamma('%s',gamma=%f)\n"%(val,univ.atoms.masses[i]/1000*0.04889615))
    f.write("hoomd.analyze.log(filename='thermo.log', quantities=['potential_energy','pair_ashbaugh_energy', 'pair_yukawa_energy','bond_harmonic_energy', 'angle_table_energy','dihedral_table_energy', 'special_pair_iithree_energy','pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=1000, overwrite=False, header_prefix='#') #Log a number of calculated quantities to a file.\n")
    f.write("writegsd=hoomd.dump.gsd('resize.gsd',period=1000, group=all,overwrite=True,truncate=True, dynamic=['property', 'momentum']) #writes the last state of the simulation\n")
    f.write("#hoomd.dump.dcd('prod.dcd',period=100000, group=outp_group, overwrite=False)\n")
    f.write("hoomd.run_upto(nsteps) #Runs the simulation up to a given time step number.\n")
    f.write("writegsd.write_restart()\n")
    f.write("#resetting the timestep of the snapshot\n")
    f.write("lastframe = gsd.hoomd.HOOMDTrajectory(gsd.pygsd.GSDFile(open('resize.gsd','rb')))[-1]\n")
    f.write("lastframe.configuration.step = 0\n")
    f.write("newgsdfile=gsd.hoomd.open('resized_box.gsd','wb')\n")
    f.write("newgsdfile.append(lastframe)\n")
    f.write("newgsdfile.close()\n")
    

elif choice==5:
    f.write("\n### Energy min after Resizing ###\n")
    #f.write("min_steps=10000\n")
    f.write("hoomd.md.integrate.mode_standard(dt= 0.2045814) #time step is 10fs\n")
    f.write("fire = hoomd.md.integrate.mode_minimize_fire(dt= 0.2045814,ftol=0.1,Etol=1e-05) #energy minimization with fire option\n")
    f.write("integrator = hoomd.md.integrate.langevin(group=integrate_group, kT=temp*0.0019872067,seed=12341) #Langevin integrator\n")
    f.write("#setting the friction factor for the Langevin integrator [mass/1000ps]\n")
    atomids=[]
    for i,val in enumerate(univ.atoms.types):
        if val not in atomids:
            atomids.append(val)
            f.write("integrator.set_gamma('%s',gamma=%f)\n"%(val,univ.atoms.masses[i]/1000*0.04889615))
    f.write("hoomd.analyze.log(filename='thermo.log', quantities=['potential_energy','pair_ashbaugh_energy', 'pair_yukawa_energy','bond_harmonic_energy', 'angle_table_energy','dihedral_table_energy', 'special_pair_iithree_energy','pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=1000, overwrite=False, header_prefix='#') #Log a number of calculated quantities to a file.\n")
    f.write("writegsd=hoomd.dump.gsd('box_min.gsd',period=1000, group=all,overwrite=True,truncate=True, dynamic=['property', 'momentum']) #writes the last state of the simulation\n")
    #f.write("#hoomd.dump.dcd('prod.dcd',period=100000, group=outp_group, overwrite=False)\n")
    f.write("hoomd.run_upto(nsteps) #Runs the simulation up to a given time step number.\n")
    f.write("writegsd.write_restart()\n")
    f.write("#resetting the timestep of the snapshot\n")
    f.write("lastframe = gsd.hoomd.HOOMDTrajectory(gsd.pygsd.GSDFile(open('box_min.gsd','rb')))[-1]\n")
    f.write("lastframe.configuration.step = 0\n")
    f.write("newgsdfile=gsd.hoomd.open('Sys_Minimized_box.gsd','wb')\n")
    f.write("newgsdfile.append(lastframe)\n")
    f.write("newgsdfile.close()\n")

f.close()
