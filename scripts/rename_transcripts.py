"""Converts each file in IN_DIRECTORY to a common filename format in
OUT_DIRECTORY."""

import os
import re
import shutil

IN_DIRECTORY = os.path.join("_data", "raw_transcripts")
OUT_DIRECTORY = os.path.join("_data", "renamed_transcripts")

# Format of input transcript IDs
REGEX_MATCH = re.compile(r"MULTISENSE\_([A-Z0-9]+)\_.*\_visit(\d+)\.txt")

# Aligns each participant key, visit number to common format
# 2024/09/10 --- replaced hardcoded values with random representative values
ALIGNMENT = {
    # fmt: off
    "NSS4O": {
        "1": "240521_MS0059",
        "2": "240602_MS0059",
        "3": "240619_MS0059",
        "4": "240629_MS0059",
        "5": "240731_MS0059",
    },
    "BPP3Q": {
        "1": "240803_MS0046",
        "2": "240810_MS0046",
        "3": "240811_MS0046"
    },
    "KN00L": {
        "1": "240602_MS0053",
        "2": "240619_MS0053"
    },
    # fmt: on
}


def rename_transcripts():
    """Converts each transcript file to a common filename format."""

    if not os.path.exists(OUT_DIRECTORY):
        os.makedirs(OUT_DIRECTORY)

    for filename in os.listdir(IN_DIRECTORY):
        match = REGEX_MATCH.match(filename)
        new_filename = ALIGNMENT[match.group(1)][match.group(2)]

        old_filepath = os.path.join(IN_DIRECTORY, filename)
        new_filepath = os.path.join(OUT_DIRECTORY, new_filename + ".txt")
        shutil.copyfile(old_filepath, new_filepath)


if __name__ == "__main__":
    rename_transcripts()
