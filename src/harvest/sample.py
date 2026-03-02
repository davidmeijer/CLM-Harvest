"""Module for sampling CLMs."""

from collections.abc import Iterator
import argparse
import inspect
import os

import torch
import numpy as np
from tqdm import tqdm
from clm.models import RNN, ConditionalRNN

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


def sample_conditional_clm(
    model: ConditionalRNN,
    conditions: np.ndarray,
    num_samples: int,
    *,
    batch_size: int = 1024,
    max_len: int = 250,
    device: str | torch.device,
    corrupt: bool = False,
) -> Iterator[str]:
    """
    Sample conditional generative model.

    :param model: conditional CLM model to sample from
    :param conditions: array of conditional descriptors
    :param num_samples: number of samples to generate
    :param batch_size: batch size for sampling
    :param max_len: maximum length of generated sequences
    :param device: device to run sampling on
    :param corrupt: whether to apply corruption to the generated samples
    :return: generated SMILES strings
    """
    model.eval()  # loading sets to eval, but ensure in eval mode

    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    _sync_model_device(model, device)
    device_kwargs = _sample_device_kwargs(model.sample, device)

    # 1 x D base tensor
    base = torch.tensor(conditions, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        remaining = num_samples

        while remaining > 0:
            this_batch = min(batch_size, remaining)
            print(f"Sampling batch of {this_batch} compounds...")

            # this_batch x D
            desc = base.expand(this_batch, -1).contiguous()
            if corrupt:
                # desc = corrupt_descriptors(desc, 0.6, 1)
                raise NotImplementedError("Corruption not implemented yet")

            batch_smiles = model.sample(
                descriptors=desc,
                max_len=max_len,
                return_smiles=True,
                return_losses=False,
                **device_kwargs,
            )
            
            # Stream generated SMILES strings
            for s in batch_smiles:
                yield s

            remaining -= this_batch


def cmd_generate_unconditional(args: argparse.Namespace) -> None:
    """
    Command to generate compounds using an unconditional CLM.

    :param args: Parsed command line arguments
    """
    model_configs = prep_clm(model_dir=args.model)

    # Make sure output dir exists
    os.makedirs(args.out, exist_ok=True)

    # Determine device
    device = torch.device(args.device)
    
    # Determine number of samples per model
    num_models = len(model_configs)
    samples_per_model = args.nsamples // num_models
    remainder = args.nsamples % num_models

    # We are going to stream generated SMILES to output file
    sampled = 0
    output_path = os.path.join(args.out, "generated_smiles.smi")
    with open(output_path, "w") as out_f:
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


def cmd_generate_conditional(args: argparse.Namespace) -> None:
    """
    Command to generate compounds using a Harvest conditional CLM.

    :param args: Parsed command line arguments
    """
    model_configs = prep_clm(model_dir=args.model)
    up_to = args.samp if args.samp is not None else float('inf')

    # Make sure output dir exists
    os.makedirs(args.out, exist_ok=True)

    # Determine device
    device = torch.device(args.device)

    # Determine number of samples per model
    num_models = len(model_configs)
    samples_per_model = args.nsamples // num_models
    remainder = args.nsamples % num_models

    # Load all models once
    models = []
    for model_idx, model_config in enumerate(model_configs):
        model = model_config.load_model(device=device)
        models.append(model)

    output_path = os.path.join(args.out, "generated_smiles_conditional.txt")
    sampled = 0

    with open(output_path, "w") as out_f, open(args.cond_fp, "r") as cond_f:
        # Skip header
        next(cond_f)
        out_f.write("identifier,model,smiles\n")

        for line_idx, line in enumerate(tqdm(cond_f, desc="Fingerprints")):
            identifier, *descr_strs = line.strip().split(",")
            conditions = np.array([float(x) for x in descr_strs], dtype=np.int8)

            for model_idx, model in enumerate(models):
                # Determine number of samples for this model
                num_samples = samples_per_model + (remainder if model_idx == num_models - 1 else 0)

                # Sample conditional CLM
                for s in tqdm(sample_conditional_clm(model, conditions, num_samples, device=device), desc="Samples", leave=False):
                    out_f.write(f"{identifier},model_{model_idx+1},{s}\n")
                    sampled += 1

                    # Flush every 1000 samples
                    if sampled % 1000 == 0:
                        out_f.flush()

            if line_idx + 1 >= up_to:
                break
