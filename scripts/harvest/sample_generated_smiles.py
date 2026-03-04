#!/usr/bin/env python3
"""Sample generated SMILES based on validity and similarity."""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.DataStructs import ExplicitBitVect
from tqdm import tqdm

from harvest.chem import smiles_to_mol, mol_to_weight, mol_to_morgan_fp, tanimoto


MORGAN_NUM_BITS = int(os.environ.get("MORGAN_NUM_BITS", 2048))
MORGAN_RADIUS = int(os.environ.get("MORGAN_RADIUS", 2))
SIMILARITY_THRESHOLD_TC = float(os.environ.get("SIMILARITY_THRESHOLD_TC", 0.8))
SIMILARITY_THRESHOLD_PW = float(os.environ.get("SIMILARITY_THRESHOLD_PW", 0.01))


RDLogger.DisableLog("rdApp.*")


def cli() -> argparse.Namespace:
    """
    Defines command line interface for script.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True, help="path to CSV file containing reference SMILES")
    parser.add_argument("--reference-smiles-col", type=str, default="smiles", help="name of column containing reference SMILES (default: SMILES)")
    parser.add_argument("--generated", type=str, required=True, help="path to CSV file containing generated SMILES")
    parser.add_argument("--generated-smiles-col", type=str, default="smiles", help="name of column containing generated SMILES (default: SMILES)")
    parser.add_argument("--outdir", type=str, required=True, help="directory to save output files")
    return parser.parse_args()


@dataclass(frozen=True)
class Compound:
    """
    Data class representing a chemical compound.
    
    :cvar smiles: SMILES string representation of the compound
    :cvar mol: RDKit Mol object representing the compound
    :cvar weight: molecular weight of the compound
    :cvar morgan_fp: Morgan fingerprint of the compound as an ExplicitBitVect
    """

    smiles: str
    mol: Chem.Mol
    weight: float
    morgan_fp: ExplicitBitVect


def empirical_p_two_sided(w: float, ref_sorted: np.ndarray) -> float:
    """
    Two-sided empirical p-value: how extreme is w relative to ref distribution?

    :param w: value to compare against reference distribution
    :param ref_sorted: sorted reference values
    :return: two-sided empirical p-value for w relative to ref_sorted
    .. note:: p = 2 * min(F(w), 1 - F(w)), where F is the empirical CDF.
    """
    n = ref_sorted.size

    # Count of reference values <= w
    k = np.searchsorted(ref_sorted, w, side="right")
    F = k / n
    p = 2.0 * min(F, 1.0 - F)

    # Avoid returning exactly 0 (finite-sample)
    return max(p, 1.0 / n)


def main() -> None:
    """
    Main function.
    """
    args = cli()

    os.makedirs(args.outdir, exist_ok=True)

    picked_org: list[Compound] = []
    picked_gen: list[Compound] = []

    # Read and parse reference SMILES into memory
    with open(args.reference, "r") as f:
        header = f.readline().strip().split(",")
        smiles_col_idx = header.index(args.reference_smiles_col)

        if smiles_col_idx == -1:
            raise ValueError(f"SMILES column '{args.reference_smiles_col}' not found in reference CSV header: {header}")
        
        for line in tqdm(f):
            smiles = line.strip().split(",")[smiles_col_idx]
            
            mol = smiles_to_mol(smiles)
            if mol is None:
                continue
            weight = mol_to_weight(mol)
            morgan_fp = mol_to_morgan_fp(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_NUM_BITS)
            picked_org.append(Compound(smiles, mol, weight, morgan_fp))
    
    picked_org_weights = np.sort([c.weight for c in picked_org])

    # Loop over generated SMILES and pick only dissimilar ones that have similar weight to the reference compounds
    with open(args.generated, "r") as f:
        header = f.readline().strip().split(",")
        smiles_col_idx = header.index(args.generated_smiles_col)

        if smiles_col_idx == -1:
            raise ValueError(f"SMILES column '{args.generated_smiles_col}' not found in generated CSV header: {header}")

        for line in tqdm(f):
            smiles = line.strip().split(",")[smiles_col_idx]
            
            mol = smiles_to_mol(smiles)
            if mol is None:
                continue
            weight = mol_to_weight(mol)
            morgan_fp = mol_to_morgan_fp(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_NUM_BITS)

            # Check if weight is similar to reference distribution
            p_weight = empirical_p_two_sided(weight, picked_org_weights)
            if p_weight < SIMILARITY_THRESHOLD_PW:
                continue
            
            # Check if molecule is dissimilar to all reference and already picked generated compounds
            is_dissimilar = True

            for ref in picked_org:
                sim = tanimoto(morgan_fp, ref.morgan_fp)
                if sim >= SIMILARITY_THRESHOLD_TC:
                    is_dissimilar = False
                    break
                
            for gen in picked_gen:
                sim = tanimoto(morgan_fp, gen.morgan_fp)
                if sim >= SIMILARITY_THRESHOLD_TC:
                    is_dissimilar = False
                    break
            
            if is_dissimilar:
                picked_gen.append(Compound(smiles, mol, weight, morgan_fp))

    # Plot weight distributions of reference and picked generated compounds
    org_w = np.array([c.weight for c in picked_org], dtype=np.float32)
    gen_w = np.array([c.weight for c in picked_gen], dtype=np.float32)

    all_w = np.concatenate([org_w, gen_w])
    w_min, w_max = all_w.min(), all_w.max()

    bins = np.linspace(w_min, w_max, 60)

    plt.figure()
    plt.hist(org_w, bins=bins, density=True, histtype="step", linewidth=2, label=f"reference (n={len(picked_org)})")
    plt.hist(gen_w, bins=bins, density=True, histtype="step", linewidth=2, label=f"picked generated (n={len(picked_gen)})")

    plt.xlabel("molecular weight (Da)")
    plt.ylabel("density")
    plt.xlim(0, w_max)
    plt.title("weight density distributions")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(args.outdir, "weight_distributions.png"), dpi=300)
    plt.close()

    # Combine reference and picked generated compounds into a single CSV file
    with open(os.path.join(args.outdir, "picked_compounds.csv"), "w") as f:
        f.write("smiles,source,weight\n")
        for c in picked_org:
            f.write(f"{c.smiles},reference,{c.weight:.4f}\n")
        for c in picked_gen:
            f.write(f"{c.smiles},picked_generated,{c.weight:.4f}\n")


if __name__ == "__main__":
    main()
