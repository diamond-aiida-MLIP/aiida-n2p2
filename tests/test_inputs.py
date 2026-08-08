"""Tests for N2p2Dataset and input preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_n2p2.data.dataset import N2p2Dataset
from aiida_n2p2.data.parameters import N2p2Parameters
from aiida_n2p2.utils.input_nn import keyword_states, parse_input_nn, render_input_nn

FIXTURES = Path(__file__).parent / 'fixtures' / 'al'
EXAMPLE_NN = Path(__file__).parent.parent / 'examples' / '1.Al' / 'input.nn'


def test_dataset_reads_structure_count():
    dataset = N2p2Dataset.from_file(FIXTURES / 'minimal_input.data')
    assert dataset.base.attributes.get('n_structures') == 2
    assert dataset.base.attributes.get('md5')
    single = dataset.get_singlefile()
    assert single.filename == 'input.data'
    assert 'begin' in single.get_content()


def test_parameters_roundtrip_preserves_active_keywords():
    original = parse_input_nn(EXAMPLE_NN)
    rendered = render_input_nn(original)
    reparsed = parse_input_nn(rendered)
    assert keyword_states(original)['epochs'] == keyword_states(reparsed)['epochs']
    assert keyword_states(original)['force_weight'] == keyword_states(reparsed)['force_weight']


def test_parameters_apply_overrides_and_enable_lists():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.apply_overrides(
        {
            'epochs': 500,
            'force_weight': 12.5,
            '__enable__': ['use_old_weights_short'],
            '__disable__': ['memorize_symfunc_results'],
        }
    )
    keywords = params.base.attributes.get('keywords')
    assert keywords['epochs']['value'] == '500'
    assert keywords['use_old_weights_short']['state'] == 'active_flag'
    assert keywords['memorize_symfunc_results']['state'] == 'disabled'


def test_parameters_singlefile_matches_render():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    original = EXAMPLE_NN.read_bytes()
    single = params.get_singlefile()
    assert single.filename == 'input.nn'
    assert single.get_content().encode('utf-8') == original


def test_parameters_set_epochs_preserves_formatting():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.set('epochs', '777')
    rendered = params.render()
    assert 'epochs                          777' in rendered
    assert '# These keywords are (almost) always required.' in rendered


def test_prepare_for_restart_enables_flag_without_changing_template_file():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.prepare_for_restart()
    rendered = params.render()
    assert '\nuse_old_weights_short' in rendered or rendered.lstrip().startswith('use_old_weights_short')
    assert keyword_states(parse_input_nn(rendered))['use_old_weights_short']['state'] == 'active_flag'
