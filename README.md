# Digital Musicology (DH-401): Expressive Timing in Performance

This repository contains our solution for [the first assignment](https://hackmd.io/@RFMItzZmQbaIqDdVZ0DovA/Hka1NuOpp) of Digital Musicology (DH-401) course. The assignment consists of two tasks: (A) creating a timing mapping between symbolic time and performance attributes (tempo,velocity) to make MIDI sound more expressive, (B) analyzing the dataset and its distributions.

We used [Aligned Scores and Performances (ASAP) dataset](https://github.com/fosfrancesco/asap-dataset) for the assignment.

## Installation

Follow this steps to reproduce our work:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:

   ```bash
   pre-commit install
   ```

3. Download dataset:

   ```bash
   mkdir data
   cd data
   git clone https://github.com/fosfrancesco/asap-dataset.git
   ```

## Project Structure

All main functions are written in the `src` directory. The `experiments.ipynb` file in the root directory provides all the scripts to reproduce figures, tables, etc. from our report.

## Authors

The project was done by:

- Petr Grinberg
- Marco Bondaschi
- Ismaïl Sahbane
- Ben Erik Kriesel
