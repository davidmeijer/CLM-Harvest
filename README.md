[![Tests](https://github.com/skinniderlab/CLM/actions/workflows/tests.yml/badge.svg)](https://github.com/skinniderlab/CLM/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/skinniderlab/CLM/badge.svg?branch=master)](https://coveralls.io/github/skinniderlab/CLM?branch=master)

## CLM-Harvest

This repository contains Python source code required to train and evaluate chemical language models for generating natural compound structures.

The CLM code is forked from [https://github.com/skinniderlab/CLM](https://github.com/skinniderlab/CLM).

Complete documentation for the CLM is available at [https://skinniderlab.github.io/CLM/](https://skinniderlab.github.io/CLM/)

## Overview

Harvest contains of an unconditional and a conditional CLM.
The unconditional CLM is a model that generates natural product-like compounds similar to real natural products.

The conditional CLM is able to generate natural product(-like) compounds based on biosynthetic cues extracted from sequence data.

A conditional Harvest model is trained on synthetic natural products, retrobiosynthesized using [RetroMol](https://github.com/moltools/retromol).
At inference time, a conditional Harvest model can ingest [antiSMASH](https://antismash.secondarymetabolites.org/#!/start) generated output file to generate natural product(-like) compounds based on the biosynthetic cues contained in the GenBank file.

## Installation

To install the CLM-Harvest package, clone this repository and install the package using Conda:

```bash
cond create -f environment.yml
conda activate clm-harvest
```

One installed, models can be trained and evaluated using the `harvest` command line interface (CLI).

## Usage

Under development. See `harvest --help` for available commands and options.
