"""Reconstructed frozen-source evaluation; no target adaptation or default run."""
import argparse,json
from pathlib import Path
from evaluation_runtime import run

DIRECTIONS=frozenset({('ffpp','celebdf_v2'),('ffpp','wilddeepfake'),('celebdf_v2','ffpp'),('celebdf_v2','wilddeepfake')})


def validate_direction(config):
    if (config['source_dataset_id'],config['target_dataset_id']) not in DIRECTIONS:
        raise ValueError('Only the four declared cross-dataset directions are supported')


def evaluate_cross_dataset(config):
    validate_direction(config)
    return run(config,[('clean',None)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--execute-evaluation',action='store_true')
    a=p.parse_args();c=json.loads(Path(a.config).read_text());validate_direction(c)
    if not a.execute_evaluation:raise ValueError('Explicit --execute-evaluation required; scientific gates remain external')
    evaluate_cross_dataset(c)

if __name__=='__main__':main()
