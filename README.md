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
