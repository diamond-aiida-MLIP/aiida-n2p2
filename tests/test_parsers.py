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


def test_scale_parser_recovers_missing_output_from_remote(
    scale_calcjob_node,
    patch_parser_retrieved,
    monkeypatch,
    tmp_path,
):
    """If retrieve is premature, the parser should poll the remote folder."""
    from aiida.orm import FolderData
    from aiida_n2p2.parsers.scaling import nnpScaleParser

    empty_retrieved_dir = tmp_path / 'empty_retrieved'
    empty_retrieved_dir.mkdir()
    (empty_retrieved_dir / 'scale.log').write_text('started\n', encoding='utf-8')
    empty_retrieved = FolderData()
    empty_retrieved.put_object_from_tree(str(empty_retrieved_dir))

    recovered = FIXTURES_AL / 'scaling.data'

    def _fake_fetch(node, filename, logger=None, **kwargs):
        destination = tmp_path / filename
        destination.write_bytes(recovered.read_bytes())
        return destination

    monkeypatch.setattr(
        'aiida_n2p2.parsers.scaling.fetch_remote_output_file',
        _fake_fetch,
    )

    parser = nnpScaleParser(scale_calcjob_node)
    patch_parser_retrieved(parser, empty_retrieved)
    result = parser.parse()

    assert result.status == 0, f'Scale parser failed: {result.message}'
    assert parser.outputs.scale.filename == 'scaling.data'


def test_train_parser_selects_best_weights(
    train_calcjob_node,
    train_retrieved_folder,
    train_retrieved_temporary_dir,
    patch_parser_retrieved,
    patch_train_atomic_number,
    regression_reference,
):
    """nnpTrainParser must pick epoch 182 and the matching weights file."""
    patch_train_atomic_number(train_calcjob_node)
    parser = ParserFactory('n2p2.train')(train_calcjob_node)
    patch_parser_retrieved(parser, train_retrieved_folder)
    result = parser.parse(
        retrieved_temporary_folder=str(train_retrieved_temporary_dir)
    )

    assert result.status == 0, f'Train parser failed: {result.message}'

    parsed_md5 = _md5_bytes(parser.outputs.weights.get_content())
    training = regression_reference['training']

    assert parsed_md5 == training['best_weights_md5'], (
        f'Regression failure: parser selected weights md5={parsed_md5}, '
        f'expected {training["best_weights_md5"]} from '
        f'{training["best_weights_file"]} (epoch {training["best_epoch"]}).'
    )

    last_md5 = _md5_bytes(parser.outputs.last_weights.get_content())
    assert last_md5 == training['last_weights_md5']
    summary = parser.outputs.training_summary.get_dict()
    assert summary['last_epoch'] == training['last_epoch']
    assert summary['best_epoch'] == training['best_epoch']
    assert summary['target_epochs'] == 200
    assert summary['training_completed'] is True
    assert summary['epochs_remaining'] == 0
