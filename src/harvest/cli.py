"""Command line interface for Harvest."""

import argparse
from datetime import time
import os
import sys
import shlex
import subprocess

from harvest.version import __version__
from harvest.sample import cmd_sample_unconditional


_SLURM_FLAGS_WITH_VALUE = {
    "--part",
    "--cpus",
    "--mem",
    "--time",
    "--gres",
    "--job-name",
}
_SLURM_FLAGS_BOOL = {"--slurm"}


def _strip_slurm_flags(argv: list[str]) -> list[str]:
    """
    Return copy of argv with slurm options removed.

    :param argv: list of command line arguments
    :return: cleaned list of command line arguments
    .. note:: this is what is passed to the inner CLI when using slurm submission
    """
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in _SLURM_FLAGS_BOOL:
            i += 1
            continue

        if arg in _SLURM_FLAGS_WITH_VALUE:
            # Skip flag + its value
            i += 2
            continue

        cleaned.append(arg)
        i += 1

    return cleaned


def _submit_via_slurm(slurm_args: argparse.Namespace, cli_argv: list[str]) -> None:
    """
    Submit the current Harvest command to Slurm using sbatch.

    :param slurm_args: parsed command line arguments (including slurm options)
    :param cli-argv: list of command line arguments to pass to the inner Harvest CLI
    """
    python = sys.executable

    # Get output directory from CLI args
    output_dir = os.path.abspath(slurm_args.out)

    # Ensure log directory exists
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Slurm settings (CLI wins over env, env wins over defaults)
    partition   = slurm_args.part       or os.environ.get("PARTITION", "skinniderlab")
    cpus        = slurm_args.cpus       or int(os.environ.get("CPUS", "8"))
    mem         = slurm_args.mem        or os.environ.get("MEM", "16G")
    time        = slurm_args.time       or os.environ.get("TIME", "24:00:00")
    gres        = slurm_args.gres       or os.environ.get("GRES", "gpu:1")
    job_name    = slurm_args.job_name   or os.environ.get("JOB_NAME", "harvest_job")

    # The command that will run on the node: python -m harvest.cli <args...>
    inner_cmd = [python, "-m", "harvest.cli", *cli_argv]
    inner_cmd_str = shlex.join(inner_cmd)

    sbatch_cmd = [
        "sbatch",
        "-J", job_name,
        "-p", partition,
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "-o", os.path.join(output_dir, "logs", "harvest_%x_%j.out"),
        "-e", os.path.join(output_dir, "logs", "harvest_%x_%j.err"),
    ]

    if gres:
        sbatch_cmd.append(f"--gres={gres}")

    # Wrap inner command so wet get some basic info + timing
    wrap_script = f"""set -euo pipefail
echo "Node: $(hostname)"
echo "Using Python: {python}"
echo "CPUs: ${{OMP_NUM_THREADS:-{cpus}}}; Mem limit: {mem}"
/usr/bin/time -v {inner_cmd_str}
"""
    
    sbatch_cmd.extend(["--export", f"ALL,OMP_NUM_THREADS={cpus},MKL_NUM_THREADS={cpus},PYTHONUNBUFFERED=1"])
    sbatch_cmd.append("--wrap", wrap_script)

    if slurm_args.dry_run:
        print("[DRY RUN] Would submit Harvest job to slurm with:")
        print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
        print("\n[DRY RUN] Full --wrap script:\n")
        print(wrap_script)
        return
    
    print("Submitting Harvest job to Slurm:")
    print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
    subprocess.run(sbatch_cmd, check=True)


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command line interface for Harvest.

    :param argv: list of command line arguments (defaults to sys.argv[1:])
    :return: the parsed command line arguments
    """
    if argv is None:
        argv = sys.argv[1:]

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=str, required=True, help="directory to save output results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"Harvest {__version__}", help="show the version number and exit")

    # Global Slurm options (only meaningful when --slurm is set)
    parser.add_argument("--slurm", action="store_true", help="submit this Harvest command as a slurm job instead of running it locally")
    parser.add_argument("--part", default=None, help="slurm partition (queue) name")
    parser.add_argument("--cpus", type=int, default=None, help="number of CPUs per task for slurm")
    parser.add_argument("--mem", default=None, help="memory request for slurm (e.g., 16G)")
    parser.add_argument("--time", type=str, default=None, help="time limit for slurm (e.g., 24:00:00)")
    parser.add_argument("--gres", type=str, default=None, help="slurm generic resources (e.g., gpu:1)")
    parser.add_argument("--job-name", type=str, default=None, help="slurm job name")
    parser.add_argument("--dry-run", action="store_true", help="if set, only print the slurm submission command without actually submitting")

    sub = parser.add_subparsers(dest="cmd", required=True)

    psu = sub.add_parser("sample-unconditional", parents=[common], help="sample unconditional CLM")
    psu.add_argument("--model-dir", type=str, required=True, help="path to dir trained CLM")
    psu.add_argument("--device", type=str, default="cpu", help="device to run sampling on (e.g., 'cuda:0' or 'cpu')")
    psu.add_argument("--num-samples", type=int, default=1000, help="number of samples to take")
    psu.set_defaults(func=lambda args: cmd_sample_unconditional(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        device=args.device,
        nsamples=args.num_samples,
    ))

    args = parser.parse_args(argv)

    if getattr(args, "slurm", False):
        # Rebuild CLi argv without the slurm-only flags
        cli_argv = _strip_slurm_flags(argv)
        _submit_via_slurm(args, cli_argv)
    else:
        args.func(args)


def main(argv: list[str] | None = None) -> None:
    """
    Main entry point for the Harvest CLI.
    """
    cli(argv)


if __name__ == "__main__":
    main()
