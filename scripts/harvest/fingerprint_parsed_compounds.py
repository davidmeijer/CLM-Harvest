#!/usr/bin/env python3
"""Fingerprint compounds parsed by RetroMol."""

import argparse
import json
import os
import yaml
import math
from typing import Generator, Any

import numpy as np
from tqdm import tqdm

from retromol.model.result import Result


def cli() -> argparse.Namespace:
    """
    Command line interface for fingerprinting parsed compounds.

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="directory to save output files")
    parser.add_argument("--jsonl", type=str, required=True, help="path to JSONL file containing RetroMol-parsed compounds")
    parser.add_argument("--matching-rules", type=str, default="matching rules file used for RetroMol parsing")
    return parser.parse_args()



def iter_jsonl(path: str) -> Generator[dict[str, Any], None, None]:
    """
    Generator that yields JSON objects from a JSONL file.
    
    :param path: path to JSONL file
    :yield: JSON object from each line of the file
    """
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def result_to_monomer_dists(r: Result) -> list[list[tuple[str, float]]]:
    a = r.linear_readout.assembly_graph
    dists: list[list[tuple[str, float]]] = []
    for n in a.g.nodes:
        ident = a.g.nodes[n].get("identity", None)
        if ident is None:
            continue
        # Currently deterministic, always 1.0 for the identified monomer
        dists.append([(ident.name, 1.0)])
    return dists


def make_fp_residues(r: Result, assignment: list[str]) -> np.ndarray:
    """
    Create residue feature vector for a Result object based on matching rules.
    
    :param r: Result object containing parsed compound information
    :param assignment: list of feature names corresponding to matching rules
    :return: numpy array representing the residue features of the fingerprint
    :raises ValueError: if an identity from the monomer distributions is not found in the assignment list
    """
    fp_residues = np.zeros(len(assignment), dtype=np.float32)
    monomer_dists = result_to_monomer_dists(r)
    for d in monomer_dists:
        for ident, prob in d:
            if ident in assignment:
                idx = assignment.index(ident)
                fp_residues[idx] += prob
            else:
                raise ValueError(f"Identity '{ident}' not found in assignment list!")
    fp_residues /= (np.linalg.norm(fp_residues) + 1e-8)  # normalize to unit vector (avoid division by zero)
    return fp_residues


def pack_fp(fp_residues: np.ndarray, fp_edges: np.ndarray, n_ident: int, cov: float) -> np.ndarray:
    """
    Pack fingerprint components into a single numpy array.
    
    :param fp_residues: numpy array representing residue features of the fingerprint
    :param fp_edges: numpy array representing edge features of the fingerprint
    :param n_ident: number of identified monomers in the compound
    :param cov: coverage of the compound by identified monomers
    :return: concatenated and normalized fingerprint as a numpy array
    """
    cov = float(max(0.0, min(1.0, cov)))
    n_ident = float(math.log1p(n_ident))
    return np.concatenate([fp_residues.astype(np.float32), fp_edges.astype(np.float32), np.array([n_ident, cov], dtype=np.float32)])


def fingerprint_result(r: Result, assignment: list[str]) -> np.ndarray:
    """
    Compute fingerprint for a given Result object based on matching rules.
    
    :param r: Result object containing parsed compound information
    :param assignment: list of feature names corresponding to matching rules
    :return: numpy array representing the fingerprint of the compound
    """
    fp_residues = make_fp_residues(r, assignment)
    fp_edges = np.zeros(256, dtype=np.float32)
    monomer_dists = result_to_monomer_dists(r)
    n_ident = len(monomer_dists)
    cov = r.calculate_coverage()
    return pack_fp(fp_residues, fp_edges, n_ident, cov)


def main() -> None:
    """
    Main function for fingerprinting parsed compounds.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    # Setup fingerprint design based on matching rules
    with open(args.matching_rules, "r") as f:
        matching_rules = yaml.safe_load(f)
    assignment = [r["name"] for r in matching_rules]

    # TODO: assign bag of monomers (dereplicated) to specific spots; train model on these fingerprints
    # TODO: do same as above but add 2/3-mers of monomers from the assembly graph in a hashed sub-fingerprint

    with open(os.path.join(args.outdir, "dataset.txt"), "w") as out_f:
        # Write header: smiles, every fingerpint feature, n_ident, cov
        header = ["smiles"] + [f"fp_{i}" for i in range(len(assignment))] + ["n_ident", "cov"]
        out_f.write(",".join(header) + "\n")

        for obj in tqdm(iter_jsonl(args.jsonl)):
            r = Result.from_dict(obj)
            smiles = r.submission.smiles
            fp = fingerprint_result(r, assignment)
            fp_str = ",".join([str(x) for x in fp])
            out_f.write(f"{smiles},{fp_str}\n")


if __name__ == "__main__":
    main()
