"""An annotation script that prompts the user to align speaker IDs to speaker
names, e.g. S1 to Participant."""

import collections
import os
import sys

IN_DIRECTORY = os.path.join('_data', 'renamed_transcripts')
OUT_DIRECTORY = os.path.join('_data', 'aligned_transcripts')

Turn = collections.namedtuple('Turn', ['speaker', 'timestamp', 'text'])


def save_transcript(filename, turns, speaker_lookup):
    """Saves the new aligned transcript to file.

    Args:
        filename (str): The filename (not the filepath) of the current session.
        turns (list of Turns): A list of Turn objects representing the turns
            taken in the current session.
        speaker_lookup (dict): A dictionary of type str:str representing the
            lookup table to align speaker IDs ('S1', 'S2', etc.) to speaker
            names ('Participant', 'Clinician', etc.).

    """
    out_fp = os.path.join(OUT_DIRECTORY, filename)
    with open(out_fp, 'w') as outfile:
        for turn in turns:
            outfile.write('{}\t{}\t{}\n'.format(speaker_lookup[turn.speaker],
                                                turn.timestamp,
                                                turn.text))


def speaker_map(speaker_id, speaker):
    """Maps speaker IDs to speaker names.

    Args:
        speaker_id (str): A single digit number (1, 2, or 3) indicating the
            shorthand index of a speaker as given by the user during input.
        speaker (str): A string e.g., 'S1' indicating which speaker ID in the
            transcript is being indexed.

    Returns:
        str: The name of the speaker; could be original e.g., 'S1' if
            unidentified by the user.
    """
    if speaker_id == '1':
        speaker = 'Participant'
    elif speaker_id == '2':
        speaker = 'Clinician'
    elif speaker_id == '3':
        speaker = 'RA'
    return speaker


def get_speaker_lookup(turns):
    """Prompts the user with sample speaker turns to determine alignment
    between speaker IDs and speaker names.

    Args:
        turns (list of Turns): A list of Turn objects representing the turns
            taken in the current session.

    Returns:
        dict: A dictionary of type str:str representing the
            lookup table to align speaker IDs ('S1', 'S2', etc.) to speaker
            names ('Participant', 'Clinician', etc.).

    """
    speakers = set([t.speaker for t in turns])
    sys.stdout.write('Found {} speakers...\n\n'.format(len(speakers)))

    speaker_lookup = dict()
    for speaker in speakers:

        speaker_turns = [t.text for t in turns if t.speaker == speaker]
        sys.stdout.write('\n'.join(speaker_turns[:10]) + '\n')

        speaker_id = raw_input('Who is this? '
                               '[1: Participant, 2: Clinician, 3: RA] ')

        speaker_lookup[speaker] = speaker_map(speaker_id, speaker)

        sys.stdout.write('\n')

    sys.stdout.write('Alignment:\n' + str(speaker_lookup) + '\n\n-----\n\n')

    return speaker_lookup


def clean_lines(lines):
    """Takes all lines from transcripts and returns the list of utterances.

    Args:
        lines (list of str): All lines from a transcript file (including lines
            such as [silence] and empty lines).

    Returns:
        list of str: All lines from a transcript file that contain utterances.
    """
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 0 and l[0] == 'S']
    return lines


def read_turns(filename):
    """Loads a session from file and compiles a list of Turn objects
    representing the session.

    Args:
        filename (str): The filename (not the filepath) of the current session.

    Returns:
        list of Turns: A list of Turn objects representing the turns
            taken in the current session.

    """

    sys.stdout.write('Reading transcript {}...\n'.format(filename))

    filepath = os.path.join(IN_DIRECTORY, filename)
    with open(filepath) as infile:
        lines = clean_lines(infile.readlines())

    turns = []
    for line in lines:
        line = line.split()
        turn = Turn(line[0], line[1], ' '.join(line[2:]))
        turns.append(turn)

    return turns


def main():
    """Main pipeline."""

    if not os.path.exists(OUT_DIRECTORY):
        os.makedirs(OUT_DIRECTORY)

    for filename in os.listdir(IN_DIRECTORY):
        out_fp = os.path.join(OUT_DIRECTORY, filename)
        if not os.path.exists(out_fp):
            turns = read_turns(filename)
            speaker_lookup = get_speaker_lookup(turns)
            save_transcript(filename, turns, speaker_lookup)

if __name__ == '__main__':
    main()
