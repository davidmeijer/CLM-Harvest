"""Train a CLM on an unconditional or conditional dataset."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


def _find_workflow_dir() -> Path | None:
    """
    Locate the repository workflow directory that contains the Snakefile.

    :returns: Path to workflow directory if found, else None
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "workflow" / "Snakefile"
        if candidate.is_file():
            return candidate.parent

    return None


def _resolve_workflow_paths(
    configfile: str,
    workflow_dir: str | None,
    snakefile: str | None,
) -> tuple[Path, Path, Path]:
    """
    Resolve and validate paths for Snakemake workflow.
    
    :param configfile: path to Snakemake config YAML
    :param workflow_dir: path to workflow directory (optional)
    :param snakefile: path to Snakefile (optional)
    :returns: tuple of (config_path, workflow_dir, snakefile_path)
    :raises FileNotFoundError: if any of the required files/directories are not found
    """
    config_path = Path(configfile).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    wf_dir: Path | None = None
    if workflow_dir:
        wf_dir = Path(workflow_dir).expanduser().resolve()

    sf_path: Path | None = None
    if snakefile:
        sf_path = Path(snakefile).expanduser().resolve()

    if sf_path is None:
        if wf_dir is None:
            wf_dir = _find_workflow_dir()
        if wf_dir is None:
            raise FileNotFoundError(
                "Could not locate workflow/Snakefile. Provide --workflow-dir or --snakefile."
            )
        sf_path = wf_dir / "Snakefile"
    elif wf_dir is None:
        wf_dir = sf_path.parent

    if not sf_path.is_file():
        raise FileNotFoundError(f"Snakefile not found: {sf_path}")
    if wf_dir is None or not wf_dir.is_dir():
        raise FileNotFoundError(f"Workflow directory not found: {wf_dir}")

    return config_path, wf_dir, sf_path


def cmd_train_model(
    configfile: str,
    workflow_dir: str | None = None,
    snakefile: str | None = None,
    jobs: int | None = None,
    latency_wait: int | None = None,
    rerun_incomplete: bool = False,
    default_resources: list[str] | None = None,
    snakemake_args: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    Run the Snakemake training workflow from any working directory.

    :param configfile: path to Snakemake config YAML
    :param workflow_dir: path to workflow directory (defaults to repo workflow/)
    :param snakefile: path to Snakefile (defaults to <workflow_dir>/Snakefile)
    :param jobs: Snakemake --jobs value
    :param latency_wait: Snakemake --latency-wait value
    :param rerun_incomplete: Snakemake --rerun-incomplete flag
    :param default_resources: list of --default-resources values (e.g., ["slurm_partition=skinniderlab"])
    :param snakemake_args: additional args to pass through to Snakemake
    :param dry_run: if True, print the command without executing
    """
    config_path, wf_dir, sf_path = _resolve_workflow_paths(
        configfile=configfile,
        workflow_dir=workflow_dir,
        snakefile=snakefile,
    )

    cmd = [
        "snakemake",
        "--snakefile",
        str(sf_path),
        "--directory",
        str(wf_dir),
        "--configfile",
        str(config_path),
    ]

    if jobs is not None:
        cmd.extend(["--jobs", str(jobs)])
    if latency_wait is not None:
        cmd.extend(["--latency-wait", str(latency_wait)])
    if rerun_incomplete:
        cmd.append("--rerun-incomplete")
    if default_resources:
        cmd.append("--default-resources")
        cmd.extend(default_resources)
    if snakemake_args:
        cmd.extend(snakemake_args)

    if dry_run:
        print("[DRY RUN] Would run:")
        print(" ", shlex.join(cmd))
        return

    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, check=True)
