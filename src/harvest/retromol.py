"""Module for parsing compounds with RetroMol."""


def cmd_run_retromol(
    input_path: str,
    out_dir: str
) -> None:
    """
    Command to run RetroMol retrosynthesis algorithm on a set of input compounds.

    :param input_path: path to input file containing SMILES strings to run retrosynthesis on
    :param out_dir: directory to save output results
    .. note:: assumes that input file has one SMILES string per line, without a header
    """
    raise NotImplementedError("RetroMol retrosynthesis command not yet implemented")
