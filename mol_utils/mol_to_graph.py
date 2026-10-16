from typing import Any, List
from collections import OrderedDict

import torch
import numpy as np

x_map = OrderedDict([
    # 'atomic_num':
    # list(range(0, 119)),
    ('chirality',[
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ]),
    ('degree', list(range(0, 11))),
    ('formal_charge', list(range(-5, 10))),
    ('num_hs',list(range(0, 9))),
    ('num_radical_electrons', list(range(0, 5))),
    ('hybridization', [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ]),
    ('is_aromatic', [False, True]),
    ('is_in_ring', [False, True])
])

e_map = OrderedDict([
    ('bond_type', [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ]),
    ('stereo', [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ]),
    ('is_conjugated', [False, True]),
])


ATOM_CODES = {}
metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))


atom_classes = [
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (15, 'P'),
    (16, 'S'),
    (34, 'Se'),
    ## halogen
    ([9, 17, 35, 53], 'halogen'),
    ## halogen
    (metals, 'metal')
]


NUM_ATOM_CLASSES = len(atom_classes)
for code, (atom, name) in enumerate(atom_classes):
    if type(atom) is list:
        for a in atom:
            ATOM_CODES[a] = code
    else:
        ATOM_CODES[atom] = code

def map_atom_code(AtomicNum):
    try:
        return ATOM_CODES[AtomicNum] + 1
    except KeyError:
        return 0
    


def one_hot_encode_features(features, feature_lengths):
    num_samples, num_features = features.shape
    one_hot_encoded = []

    for i in range(num_features):
        feature_column = features[:, i]
        num_classes = feature_lengths[list(feature_lengths.keys())[i]]
        one_hot = torch.eye(num_classes)[feature_column]
        one_hot_encoded.append(one_hot)

    return torch.concat(one_hot_encoded, dim=1)


x_map_length = {key: len(value) for key, value in x_map.items()}
e_map_length = {key: len(value) for key, value in e_map.items()}

def from_rdmol(mol: Any, one_hot=True) -> tuple[torch.Tensor, ...]:
    """Encode RDKit atoms and bonds in the checkpoint-compatible order."""
    from rdkit import Chem

    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    atom_ids: List[int] = []
    atomic_nums: List[int] = []

    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        AtomicNum = atom.GetAtomicNum()
        atom_ids.append(map_atom_code(AtomicNum))
        atomic_nums.append(AtomicNum)

        row.append(x_map['chirality'].index(str(atom.GetChiralTag())) if str(atom.GetChiralTag()) in x_map['chirality'] else 0)
        row.append(x_map['degree'].index(atom.GetTotalDegree()) if atom.GetTotalDegree() in x_map['degree'] else len(x_map['degree']) - 1)
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()) if atom.GetFormalCharge() in x_map['formal_charge'] else len(x_map['formal_charge']) - 1)
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()) if atom.GetTotalNumHs() in x_map['num_hs'] else len(x_map['num_hs']) - 1)
        row.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()) if atom.GetNumRadicalElectrons() in x_map['num_radical_electrons'] else len(x_map['num_radical_electrons']) - 1)
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())) if str(atom.GetHybridization()) in x_map['hybridization'] else 0)
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)
        
    atom_ids = torch.tensor(atom_ids,dtype=torch.long).view(-1, 1)
    atomic_nums = torch.tensor(atomic_nums,dtype=torch.long).view(-1, 1) 
    x = torch.tensor(xs, dtype=torch.long).view(-1, 8)
    
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if one_hot:
        x = one_hot_encode_features(x,x_map_length)
        edge_attr = one_hot_encode_features(edge_attr,e_map_length)
    else:
        x = x.float()
        edge_attr = edge_attr.float()
    return atom_ids, atomic_nums, x, edge_index, edge_attr
