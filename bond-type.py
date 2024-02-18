# %%
import numpy, scipy

import pyscf
from pyscf import gto, scf

def euler_anlge_to_matrix(n=[0, 0, 1], theta=0.0):

    norm = numpy.linalg.norm(n)
    n = n / norm
    
    nx, ny, nz = n
    # Build the rotation matrix in real space
    rxx = nx * nx * (1 - numpy.cos(theta)) + numpy.cos(theta)
    rxy = nx * ny * (1 - numpy.cos(theta)) + nz * numpy.sin(theta)
    rxz = nx * nz * (1 - numpy.cos(theta)) - ny * numpy.sin(theta)

    ryx = nx * ny * (1 - numpy.cos(theta)) - nz * numpy.sin(theta)
    ryy = ny * ny * (1 - numpy.cos(theta)) + numpy.cos(theta)
    ryz = ny * nz * (1 - numpy.cos(theta)) + nx * numpy.sin(theta)

    rzx = nx * nz * (1 - numpy.cos(theta)) + ny * numpy.sin(theta)
    rzy = ny * nz * (1 - numpy.cos(theta)) - nx * numpy.sin(theta)
    rzz = nz * nz * (1 - numpy.cos(theta)) + numpy.cos(theta)

    return numpy.array([[rxx, rxy, rxz], [ryx, ryy, ryz], [rzx, rzy, rzz]])

def rotate_bond_order_by_z_axis(m, n_vector, theta, rho, ix1=None, ix2=None):
    # Build the rotation matrix in real space

    rot = euler_anlge_to_matrix(n=n_vector, theta=theta)

    m_rot = m.copy()
    m_rot.atom = list(zip([m.atom_symbol(ia) for ia in range(m.natm)], numpy.dot(m.atom_coords(), rot)))
    m_rot.unit = 'Bohr'
    m_rot.build()

    u    = m.ao_rotation_matrix(rot)
    ovlp = gto.intor_cross('int1e_ovlp', m, m_rot)

    ovlp_u = numpy.dot(ovlp, u)
    ps = numpy.einsum('ij,jk->ik', rho, ovlp_u)

    bond_order = numpy.einsum('ij,ji->', ps[ix1], ps[ix2])
    return bond_order

def get_trans_cord(mol, atom1, atom2):
    atoms = mol.atom.split('\n')
    natom = len(atoms) - 2
    ligand_atom = []
    ligand_cord = numpy.zeros((natom,3))
    for iatom in range(natom):
        ligand_cord[iatom,:] = [float(i) for i in atoms[iatom+1][1:].split(' ') if i != '' ]
        iatom_label = [i for i in atoms[iatom+1].split(' ') if i != '' ][0]
        ligand_atom.append(iatom_label)
    
    center = 0.5 * (ligand_cord[atom1,:] + ligand_cord[atom2,:])
    for iatom in range(natom):
        ligand_cord[iatom,:] -= center

    m_trans = mol.copy()
    m_trans.atom = list(zip([mol.atom_symbol(ia) for ia in range(mol.natm)], ligand_cord))
    m_trans.build()

    n_vector = ligand_cord[atom1,:] - ligand_cord[atom2,:]
    return m_trans, n_vector

c2h4 = pyscf.gto.Mole()
c2h4.build(
    atom='''
C                 -1.35837200    0.89876226   -4.95388865
H                 -0.82520826   -0.02894266   -4.95388865
H                 -2.42837200    0.89876226   -4.95388865
C                 -0.68309770    2.07373955   -4.95388865
H                 -1.21626144    3.00144447   -4.95388865
H                  0.38690230    2.07373955   -4.95388865
''',
    verbose=0, basis='sto3g'
)
atom1 = 0
atom2 = 3
newmol, n_vector = get_trans_cord(mol=c2h4, atom1=atom1, atom2=atom2)

rhf = scf.RHF(newmol)
rhf.kernel()

# %%
ind_c1 = newmol.search_ao_label('0 C')
ind_c2 = newmol.search_ao_label('3 C')

ix1 = numpy.ix_(ind_c1, ind_c2)
ix2 = numpy.ix_(ind_c2, ind_c1)

ps = numpy.einsum('ij,jk->ik', rhf.make_rdm1(), rhf.get_ovlp())
bond_order = numpy.einsum('ij,ji->', ps[ix1], ps[ix2])
print("bond_order = %6.4f" % bond_order)

# %%
from matplotlib import pyplot
fig, axs = pyplot.subplots(2, 1)

tt = numpy.linspace(0, 2*numpy.pi, 100)
bo = [rotate_bond_order_by_z_axis(newmol, n_vector, t, rhf.make_rdm1(), ix1, ix2) for t in tt]

boft = numpy.fft.fft(bo)
freq = numpy.fft.fftfreq(len(bo), tt[1]-tt[0]) * numpy.pi

ax = axs[0]
ax.plot(tt, bo)
ax.set_xlim(0, 2*numpy.pi)
ax.text(0.05, 0.1, r"$b(\theta)$", transform=ax.transAxes, size=20)
ax.set_xlabel(r"$\theta$", size=20)

ax = axs[1]
ax.plot(freq, boft.real / len(bo), label='real', linestyle='', marker='o')
# ax.plot(freq, boft.imag / len(bo), label='imag', linestyle='', marker='o')
ax.set_xlim(-4, 4)
ax.set_ylim(0.0, bond_order*1.1)
ax.set_xlabel(r"$\omega$", size=20)
ax.text(0.05, 0.1, r"$\tilde{b}(\omega)$", transform=ax.transAxes, size=20)

fig.tight_layout()
fig.savefig('fft')

# %%
sigma_order = 0.0
pi_order = 0.0

for f, b in zip(freq, boft):

    if abs(f) < 1e-3:
        sigma_order += b.real / len(bo)
    elif abs(abs(f) - 1.0) < 1e-1:
        pi_order += b.real / len(bo)

print("sigma_order = %6.4f" % sigma_order)
print("pi_order    = %6.4f" % pi_order)