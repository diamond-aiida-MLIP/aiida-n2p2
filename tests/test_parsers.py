"""Parser tests using golden Al fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

from aiida.plugins import ParserFactory

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'


def _md5_bytes(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()


def test_scale_parser_output_matches_reference(
    scale_calcjob_node,
    scale_retrieved_folder,
    patch_parser_retrieved,
    regression_reference,
):
    """nnpScaleParser must reproduce scaling.data byte-for-byte."""
    parser = ParserFactory('n2p2.scale')(scale_calcjob_node)
    patch_parser_retrieved(parser, scale_retrieved_folder)
    result = parser.parse()

    assert result.status == 0, f'Scale parser failed: {result.message}'

    parsed_md5 = _md5_bytes(parser.outputs.scale.get_content())
    expected_md5 = regression_reference['scaling']['md5']

    assert parsed_md5 == expected_md5, (
        f'Regression failure: parsed scaling.data md5={parsed_md5}, '
        f'expected {expected_md5} (Al dump-949).'
    )


def test_train_parser_selects_best_weights(
    train_calcjob_node,
    train_retrieved_folder,
    patch_parser_retrieved,
    patch_train_atomic_number,
    regression_reference,
):
    """nnpTrainParser must pick epoch 182 and the matching weights file."""
    patch_train_atomic_number(train_calcjob_node)
    parser = ParserFactory('n2p2.train')(train_calcjob_node)
    patch_parser_retrieved(parser, train_retrieved_folder)
    result = parser.parse()

    assert result.status == 0, f'Train parser failed: {result.message}'

    parsed_md5 = _md5_bytes(parser.outputs.weights.get_content())
    training = regression_reference['training']

    assert parsed_md5 == training['best_weights_md5'], (
        f'Regression failure: parser selected weights md5={parsed_md5}, '
        f'expected {training["best_weights_md5"]} from '
        f'{training["best_weights_file"]} (epoch {training["best_epoch"]}).'
    )
