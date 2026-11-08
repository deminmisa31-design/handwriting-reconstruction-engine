"""Regression tests for HRE 0.1 token composition patch.

Run in the patched HRE repository with: python -m pytest -q
Tests do not require user photos; real-photo validation is a separate stage.
"""
import pytest
from hre.compose import CoverageError, plan_text


class MiniStore:
    def __init__(self, definitions):
        self._sources = []
        self._samples = []
        for i, item in enumerate(definitions):
            text, lang, kind, synthetic = item
            sid = f'source{i}'
            self._sources.append({'id': sid, 'synthetic': synthetic})
            self._samples.append({
                'id': f'sample{i:02d}', 'source_id': sid,
                'text': text, 'language': lang, 'kind': kind,
                'status': 'verified', 'revision': 1,
                'metrics': {'stroke_width_median': 3.0},
                'entry': None, 'exit': None, 'baseline': 10.0,
            })

    def sources(self):
        return self._sources

    def samples(self):
        return self._samples


def words(*items):
    return MiniStore([(*item, False) for item in items])


def test_russian_word_with_separately_verified_comma():
    s = words(('Привет', 'ru', 'word'), (',', 'ru', 'character'))
    plan = plan_text(s, 'Привет, Привет', 'ru')
    assert ''.join(t['text'] for t in plan['tokens']) == 'Привет, Привет'
    assert [(t['text'], t['start'], t['end']) for t in plan['tokens']
            if t['kind'] == 'sample'] == [
                ('Привет', 0, 6), (',', 6, 7), ('Привет', 8, 14)]
    assert not plan['tokens'][1]['join_previous']
    assert plan['tokens'][0]['group'] == plan['tokens'][1]['group']


def test_mixed_russian_english_in_same_line():
    s = words(('Привет', 'ru', 'word'), (',', 'ru', 'character'),
              ('hello', 'en', 'word'), ('!', 'en', 'character'))
    plan = plan_text(s, 'Привет, hello!', 'auto')
    assert plan['language'] == 'auto'
    assert ''.join(t['text'] for t in plan['tokens']) == 'Привет, hello!'
    assert [x['text'] for x in plan['tokens'] if x['kind'] == 'sample'] == ['Привет', ',', 'hello', '!']


def test_mixed_script_without_space():
    s = words(('Hello', 'en', 'word'), ('мир', 'ru', 'word'), ('!', 'ru', 'character'))
    plan = plan_text(s, 'Helloмир!', 'auto')
    assert ''.join(t['text'] for t in plan['tokens']) == 'Helloмир!'
    assert [t['start'] for t in plan['tokens']] == [0, 5, 8]


def test_missing_punctuation_is_error_not_font_fallback():
    s = words(('Hello', 'en', 'word'))
    with pytest.raises(CoverageError) as ex:
        plan_text(s, 'Hello!', 'en')
    assert any(x['text'] == '!' for x in ex.value.missing)


def test_synthetic_not_accepted_in_strict_mode():
    s = MiniStore([('Hello', 'en', 'word', False), ('!', 'en', 'character', True)])
    with pytest.raises(CoverageError):
        plan_text(s, 'Hello!', 'en')
    assert ''.join(t['text'] for t in plan_text(s, 'Hello!', 'en', mode='experimental-synthesis')['tokens']) == 'Hello!'


def test_compatibility_whole_word_has_priority():
    s = words(('Hello!', 'en', 'word'), ('Hello', 'en', 'word'), ('!', 'en', 'character'))
    samples = [x for x in plan_text(s, 'Hello!', 'en')['tokens'] if x['kind'] == 'sample']
    assert len(samples) == 1 and samples[0]['text'] == 'Hello!'


def test_spaces_newlines_and_unicode_positions_are_exact():
    s = words(('Ёж', 'ru', 'word'), ('.', 'ru', 'character'), ('ok', 'en', 'word'))
    text = 'Ёж.  ok\nЁж.'
    plan = plan_text(s, text, 'auto')
    assert ''.join(t['text'] for t in plan['tokens']) == text
    assert [(t['start'], t['end']) for t in plan['tokens'] if t['kind'] == 'sample'] == [
        (0, 2), (2, 3), (5, 7), (8, 10), (10, 11)]


def test_fixed_language_refuses_wrong_script_instead_of_mislabelling():
    s = words(('Hello', 'en', 'word'), ('мир', 'ru', 'word'))
    with pytest.raises(CoverageError) as ex:
        plan_text(s, 'Hello мир', 'en')
    assert any(e['text'] == 'мир' for e in ex.value.missing)


def test_rejects_unexpected_controls():
    s = words(('Привет', 'ru', 'word'))
    with pytest.raises(ValueError): plan_text(s, 'Привет\t', 'auto')
