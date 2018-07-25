#!/usr/bin/python -u

import numpy as np
import openbabel
import glob
from optparse import OptionParser
import fppy
from itertools import combinations


def lcons2lat(cons):
    (a, b, c, alpha, beta, gamma) = cons

    bc2 = b**2 + c**2 - 2 * b * c * np.cos(alpha)

    h1 = a
    h2 = b * np.cos(gamma)
    h3 = b * np.sin(gamma)
    h4 = c * np.cos(beta)
    h5 = ((h2 - h4)**2 + h3**2 + c**2 - h4**2 - bc2) / (2 * h3)
    h6 = np.sqrt(c**2 - h4**2 - h5**2)

    lattice = [[h1, 0., 0.], [h2, h3, 0.], [h4, h5, h6]]
    return np.array(lattice, float)


def getStruct(ciffile):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "fract")
    ccif = cleancif(ciffile)
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, ccif)
    outmol = obConversion.WriteString(mol)
    buff = outmol.splitlines()
    latparameters = np.array(buff[1].split(), float)
    latparameters[3] = latparameters[3] / 180. * np.pi
    latparameters[4] = latparameters[4] / 180. * np.pi
    latparameters[5] = latparameters[5] / 180. * np.pi
    atomtmp = [x.split() for x in buff[2:-1]]
    symbs = [x[0] for x in atomtmp]
    atoms = [x[1:] for x in atomtmp]
    atoms = np.array(atoms, float)
    lat = lcons2lat(latparameters)
    # print(buff)
    # print(latparameters)
    # print(atoms)
    # print (symbs)
    return lat, atoms, symbs


def check_occupacy(catominfo_, iocu):
    retlist = []
    for x in catominfo_:
        xx = x.split()[iocu]
        if float(xx.split('(')[0]) < 1.0:
            retlist.append(x.split()[0])
    return retlist


def cleancif(ciffile):
    buff = []
    with open(ciffile) as f:
        for line in f:
            buff.append(line)
    cifstring = ''.join(buff)
    # print cifstring
    cifstring_ = cifstring.split('loop_')
    n = len(cifstring_)
    for i in range(n):
        item = cifstring_[i]
        if '_atom_site_occupancy' in item:
            atominfo = item
            iatominfo = i
    # print atominfo
    # print i
    catominfo = atominfo.strip().split('\n')
    # print catominfo.split('\n')
    # print catominfo
    iocu = catominfo.index('_atom_site_occupancy')
    # print iocu
    # print catominfo[iocu]
    irstart = 0
    for x in catominfo:
        if '_atom' in x:
            irstart += 1
    # rstlist = catominfo[:irstart] + check_occupacy(catominfo[irstart:], iocu)
    # # print rstlist
    # rstlistn = '\n'.join(rstlist)
    # cifstring_[iatominfo] = rstlistn
    # ccif = 'loop_\n'.join(cifstring_)
    rstlist = check_occupacy(catominfo[irstart:], iocu)
    kick = []
    for x in rstlist:
        for i in range(len(buff)):
            if x in buff[i].split():
                kick.append(i)
    ccif_ = []
    for i in range(len(buff)):
        if i not in kick:
            ccif_.append(buff[i])
    ccif = ''.join(ccif_)
    return ccif


def calBonds(lat, atoms, symbs):
    nat = len(atoms)
    bonds = []
    for iat in range(nat):
        ri = np.dot(atoms[iat], lat)
        for jat in range(nat):
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for iz in (-1, 0, 1):
                        trans = np.array([ix, iy, iz], float)
                        redj = atoms[jat] + trans
                        rj = np.dot(redj, lat)
                        rij = ri - rj
                        dij = np.sqrt(np.dot(rij, rij))
                        bonds.append([iat, jat, symbs[iat], symbs[jat], dij])
    return bonds


def secBonds(bonds, bondtype, lmin, lmax):
    ba = bondtype.split('-')[0]
    bb = bondtype.split('-')[1]
    oband = []
    for x in bonds:
        if x[2] is ba and x[3] is bb:
            if x[4] <= lmax and x[4] >= lmin:
                oband.append(x[4])
    return oband


def select_C_cluster(lat, atoms, symbs):
    nat = len(atoms)
    atoms_c = []
    for i in range(nat):
        if symbs[i] == 'C':
            atoms_c.append(atoms[i])
    nat_c = len(atoms_c)
    C_cluster = []
    for iat in range(nat_c):
        select_c1 = []
        ri = np.dot(atoms_c[iat], lat)
        for jat in range(nat_c):
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for iz in (-1, 0, 1):
                        trans = np.array([ix, iy, iz], float)
                        redj = atoms_c[jat] + trans
                        rj = np.dot(redj, lat)
                        rij = ri - rj
                        dij = np.sqrt(np.dot(rij, rij))
                        if dij < 3.0:
                            select_c1.append(rj)
        if len(select_c1) >= 6:
            C_cluster.append(select_c1)
    return C_cluster


def detecC6(C_cluster, fpc6):
    types = np.array([1,1,1,1,1,1], int)
    znucl = np.array([6], int)
    for scluster in C_cluster:
        nsc = len(scluster)
        comb = combinations(range(nsc), 6)
        for x in list(comb):
            toc = []
            for i in x:
                toc.append(scluster[i])
            toc = np.array(toc)
            fptmp = fppy.fp_nonperiodic(toc, types, znucl)
            fptmp = np.array(fptmp, float)
            dd = fpc6 - fptmp
            fpd = np.dot(dd, dd)
            if fpd < 0.01:
                return True
    return False


def calfpc6():
    buff = []
    with open('C6') as f:
        for line in f:
            buff.append(line.split())
    lat = np.array(buff[2:5], float)
    typ = np.array(buff[6], int)
    nat = sum(typ)
    pos = np.array(buff[8:nat+8], float)
    rxyz = np.dot(pos, lat)
    types = []
    for i in range(len(typ)):
        types += [i + 1] * typ[i]
    types = np.array(types, int)
    znucl = [6]
    znucl = np.array(znucl, int) 
    fp = fppy.fp_nonperiodic(rxyz, types, znucl)
    fp = np.array(fp, float)
    return fp


def main():
    parser = OptionParser()
    parser.set_defaults(cifdir=None,
                        bondtype="C-C",
                        lmin=1.2,
                        lmax=1.5,
                        n=-1)
    parser.add_option("-c", dest="ciflists", type="string")
    parser.add_option("-b", dest="bondtype", type="string")
    parser.add_option("-l", dest="lmin", type="float")
    parser.add_option("-u", dest="lmax", type="float")
    parser.add_option("-n", dest="n", type="int")
    (options, args) = parser.parse_args()
    # cifiles = glob.glob(options.cifdir + '/*.cif')
    cifiles = []
    with open(options.ciflists) as f:
        for line in f:
            cifiles.append(line.strip())

    fpc6 = calfpc6()

    for c, cifile in enumerate(cifiles[:options.n]):
        print '#--> ', c, cifile
        lat, atoms, symbs = getStruct(cifile)
        if options.bondtype == 'C6':
            C_cluster = select_C_cluster(lat, atoms, symbs)
            if detecC6(C_cluster, fpc6):
                print ('-----------------------')
                print ('%40s ...C6...' % cifile)
                print ('-----------------------')
        else:
            bonds = calBonds(lat, atoms, symbs)
            oband = secBonds(bonds, options.bondtype, options.lmin, options.lmax)
            if len(oband) > 0:
                print ('-----------------------')
                print ('%40s ...OK...' % cifile)
                print ('-----------------------')
                for obb in oband:
                    print (options.bondtype, obb)
                print ('-----------------------')


if __name__ == '__main__':
    main()
    # print cleancif('7.cif')
