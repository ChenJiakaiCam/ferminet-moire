# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ground state config for Sc-H-H... chain with OBC with FermiNet and pseudopotentials."""

from ferminet import base_config
from ferminet.utils import system
import ml_collections
import pyscf

def _set_geometry(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Constructs a 1D chain with Sc on the left, followed by H atoms."""

  # Number of Hydrogen atoms
  n_hydrogens = cfg.system.n_hydrogens

  # Distance between neighbouring atoms
  a = cfg.system.bond_length

  cfg.system.molecule = []

  # 1. Place the Scandium atom at the origin
  cfg.system.molecule.append(
      system.Atom(symbol='Sc', coords=(0.0, 0.0, 0.0), units='bohr')
  )

  # 2. Place the Hydrogen atoms extending to the right
  for i in range(1, n_hydrogens + 1):
    x = i * a
    cfg.system.molecule.append(
        system.Atom(symbol='H', coords=(x, 0.0, 0.0), units='bohr')
    )

  # One Sc contributes 11 valence electrons (with the pseudopotential).
  # One H contributes 1 electron.
  nelec = 11 + n_hydrogens
  
  # For 9 H's, this gives 20 total electrons -> (10 spin-up, 10 spin-down)
  cfg.system.electrons = (nelec // 2, nelec - nelec // 2)

  return cfg


def finalize(cfg):
  """Sets the molecule, nuclear charge, and electrons for the atoms.

  Args:
    cfg: ml_collections.ConfigDict after all argument parsing.

  Returns:
    ml_collections.ConfigDict with the nuclear charge for the atom in
    cfg.system.molecule and cfg.system.charge appropriately set.
  """

  # Create a pyscf Mole object with pseudopotentials to be used for
  # pretraining and updating the config for consistency
  cfg = _set_geometry(cfg)
  
  mol = pyscf.gto.Mole()
  mol.atom = [[atom.symbol, atom.coords] for atom in cfg.system.molecule]

  atoms = list(set([atom.symbol for atom in cfg.system.molecule]))
  pseudo_atoms = cfg.system.pp.symbols if cfg.system.use_pp else []
  mol.basis = {
      atom:
      cfg.system.pp.basis if atom in pseudo_atoms else 'cc-pvdz'
      for atom in atoms
  }
  mol.ecp = {
      atom: cfg.system.pp.type
      for atom in atoms if atom in pseudo_atoms
  }

  mol.charge = 0
  mol.spin = 0
  mol.unit = 'bohr'
  mol.build()

  cfg.system.pyscf_mol = mol

  return cfg


def get_config():
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  
  cfg.system.update({
      'bond_length': 3.78,   # Distance between adjacent atoms (bohr)
      'n_hydrogens': 9,      # Number of H atoms following the Sc atom
  })

  cfg.system.use_pp = True  # Enable pseudopotentials
  # In base_config.default(), the default setting for cfg.system.pp.basis is explicitly 'ccecp-cc-pVDZ'
  cfg.system.pp.symbols = ['Sc']  # Indicate which atoms to apply PP to
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  
  cfg.pretrain.iterations = 10_000
  cfg.optim.reset_if_nan = True
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize

  return cfg