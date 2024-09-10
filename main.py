"""Main pipeline for analysis."""
import argparse
import os

import construction
import analysis.unifeature
import analysis.liwc_repair_scores
import analysis.predictive

from utils.log import init_logger, log, debug
import utils.reader.harvard

# directory to output LIWC-ready transcripts
LIWC_OUT_PATH = os.path.join('_data', 'liwc_transcripts')

# location of the HM transcripts to analyze
HM_ROOT = os.path.join('_data', 'aligned_transcripts')


def export_for_liwc():
    """Exports LIWC-ready plaintext files."""
    if not os.path.exists(LIWC_OUT_PATH):
        os.makedirs(LIWC_OUT_PATH)
    for transcript in utils.reader.harvard.get_transcripts(HM_ROOT):
        lines = transcript.export_lines()
        filepath = os.path.join(LIWC_OUT_PATH, transcript.session + '.txt')
        with open(filepath, 'w') as outfile:
            outfile.write('\n'.join(transcript.export_lines()))
        debug('Exported {} lines from session {}.'.format(len(lines),
                                                          transcript.session))
    log('Finished exporting for LIWC analysis.')


def parse_args():
    """Iniitalizes and reads command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug",
                        help="include debugging messages",
                        action='store_true')
    parser.add_argument("--liwc",
                        help="export transcripts for LIWC analysis",
                        action='store_true')

    return parser.parse_args()


def main():
    """Main pipeline."""
    args = parse_args()
    init_logger(args)

    if args.liwc:
        export_for_liwc()

    else:

        transcripts = construction.construct_transcripts(HM_ROOT)

        analysis.unifeature.test_pos_neg(transcripts)

        # unifeature analysis
        analysis.unifeature.regression_liwc_scores(transcripts)
        analysis.unifeature.regression_ppl_scores(transcripts)
        analysis.unifeature.regression_repair_scores(transcripts)

        # dual-feature analysis
        analysis.liwc_repair_scores.glms(transcripts)

        # predictive analysis
        analysis.predictive.run_predictive_models(transcripts)


if __name__ == '__main__':
    main()
