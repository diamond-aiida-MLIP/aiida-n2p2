"""Tests for input.nn parsing and rendering."""

from pathlib import Path

from aiida_n2p2.data.parameters import N2p2Parameters
from aiida_n2p2.utils.input_nn import enable_keyword, keyword_states, parse_input_nn

EXAMPLE_NN = Path(__file__).parent.parent / 'examples' / '1.Al' / 'input.nn'


def test_parse_input_nn_finds_commented_restart_flag():
    lines = parse_input_nn(EXAMPLE_NN)
    states = keyword_states(lines)
    assert states['use_old_weights_short']['state'] == 'disabled'
    assert states['epochs']['state'] == 'active'
    assert states['epochs']['value'] == '200'


def test_enable_restart_flag():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.enable('use_old_weights_short')
    rendered = params.render()
    assert 'use_old_weights_short' in rendered
    assert '#use_old_weights_short' not in rendered.split('\n')[0:60]


def test_prepare_for_restart_extends_epochs():
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.prepare_for_restart(additional_epochs=100)
    keywords = params.get_attribute('keywords')
    assert keywords['use_old_weights_short']['state'] == 'active_flag'
    assert keywords['epochs']['value'] == '300'
