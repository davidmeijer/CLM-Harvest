"""Module for sampling CLMs."""

from collections.abc import Iterator
import argparse
import inspect
import os

import torch
from tqdm import tqdm
from clm.models import RNN

from harvest.loader import prep_clm


def _sample_device_kwargs(sample_fn, device: torch.device) -> dict:
    """
    Return device-related kwargs supported by the CLM sample method.

    This guards against version differences where `sample` may accept
    `device`, `use_cuda`, or `cuda`, or nothing at all.
    """
    try:
        params = inspect.signature(sample_fn).parameters
    except (TypeError, ValueError):
        return {}

    if "device" in params:
        return {"device": device}
    if "use_cuda" in params:
        return {"use_cuda": device.type == "cuda"}
    if "cuda" in params:
        return {"cuda": device.type == "cuda"}
    return {}


def _sync_model_device(model: torch.nn.Module, device: torch.device) -> None:
    """
    Best-effort sync for models that track a `device` attribute internally.
    """
    if hasattr(model, "device"):
        try:
            model.device = device
        except Exception:
            pass


def sample_unconditional_clm(
    model: RNN,
    num_samples: int,
    *,
    batch_size: int = 1024,
    max_len: int = 250,
    device: str | torch.device
) -> Iterator[str]:
    """
    Sample unconditional generative model.
    
    :param model: CLM model to sample from
    :param num_samples: number of samples to generate
    :param batch_size: batch size for sampling
    :param max_len: maximum length of generated sequences
    :param device: device to run sampling on
    :return: generated SMILES strings
    """
    model.eval()  # loading sets to eval, but ensure in eval mode

    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    _sync_model_device(model, device)
    device_kwargs = _sample_device_kwargs(model.sample, device)

    with torch.inference_mode():
        remaining = num_samples

        while remaining > 0:
            this_batch = min(batch_size, remaining)

            batch_smiles = model.sample(
                n_sequences=this_batch,
                max_len=max_len,
                return_smiles=True,
                return_losses=False,
                descriptors=None,
                **device_kwargs,
            )
            
            # Stream generated SMILES strings
            for s in batch_smiles:
                yield s

            remaining -= this_batch


def cmd_sample_unconditional(
    model_dir: str,
    out_dir: str,
    device: str | torch.device,
    nsamples: int,
) -> None:
    """
    Command to sample compounds using an unconditional CLM.

    :param model_dir: path to directory containing trained CLM model(s)
    :param out_dir: directory to save output results
    :param device: device to run sampling on (e.g., 'cuda:0' or 'cpu')
    :param nsamples: total number of samples to generate
    :return: None
    """
    model_configs = prep_clm(model_dir=model_dir)

    # Make sure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Determine device
    device = torch.device(device)
    
    # Determine number of samples per model
    num_models = len(model_configs)
    samples_per_model = nsamples // num_models
    remainder = nsamples % num_models

    # We are going to stream generated SMILES to output file
    sampled = 0
    output_path = os.path.join(out_dir, "generated_smiles.csv")
    with open(output_path, "w") as out_f:
        # write header
        out_f.write("smiles\n")

        for model_idx, model_config in enumerate(tqdm(model_configs, desc="Models", leave=False)):
            
            # Load model for sampling
            model = model_config.load_model(device=device)

            # Determine number of samples for this model
            num_samples = samples_per_model + (remainder if model_idx == num_models - 1 else 0)

            # Generate samples and write to output file
            for s in tqdm(sample_unconditional_clm(model, num_samples, device=device), desc="Samples", leave=False):
                out_f.write(s + "\n")
                sampled += 1

                # Flush every 1000 samples
                if sampled % 1000 == 0:
                    out_f.flush()
