# Harvard-McLean Dialogue Analysis

## Usage

```
usage: main.py [-h] [-d] [--liwc]

optional arguments:
  -h, --help   show this help message and exit
  -d, --debug  include debugging messages
  --liwc       export transcripts for LIWC analysis
```

## Environment

This project is written in Python 2.7. An exhaustive list of dependencies can be
found in `environment.yml`.

To create a new [conda](https://github.com/conda/conda) environment with all
requirements, run the following command.

```bash
conda env create -f environment.yml
```

Additionally, install the requirements for
[deep_disfluency](https://github.com/dsg-bielefeld/deep_disfluency)
using the following command.

```bash
pip install -r deep_disfluency/requirements.txt
```

To update from an old version of the environment, run the following command.

```bash
conda env update environment.yml
```

Activate/deactivate this environment as usual.

```bash
conda activate hm_dialogue
conda deactivate hm_dialogue
```

## Scripts

### rename_transcripts.py

Renames new transcript files from Justin to a common format. For example,
`MULTISENSE_HXXCM_onsiteInterview_audioHeadset_S1+S2_visit3.txt` becomes
`170208_MS0034.txt`.

### align_speakers.py

Provides an interface for tagging speaker IDs (e.g., S1) with speaker names
(e.g., Participant).

## References

Used in the following publications:

> A. Vail, E. Liebson, J. Baker, L.-P. Morency. Toward Objective, Multifaceted Characterization of Psychotic Disorders: Lexical, Structural, and Disfluency Markers of Spoken Language. Proceedings of the Twentieth International Conference on Multimodal Interaction (ICMI 2018), Boulder, Colorado, 2018. https://doi.org/10.1145/3242969.3243020

> J. Girard*, A. Vail*, E. Liebenthal, K. Brown, C. Kilciksiz, L. Pennant, E. Liebson, D. Öngür, L.-P. Morency, J. Baker. Computational Analysis of Spoken Language in Acute Psychosis and Mania. Schizophrenia Research, 2021. (*equal contribution) https://doi.org/10.1016/j.schres.2021.06.040
