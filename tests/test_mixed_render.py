import io
import json
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest

from hre.compose import plan_text, render, CoverageError
from hre.store import Store
from hre.validation import validate_manifest
from hre.web import create_app
from test_mixed_script import words


def test_verified_compound_word_with_trailing_punctuation():
    s=words(("don't",'en','word'),(',', 'en','character'))
    assert [t['text'] for t in plan_text(s,"don't,",'en')['tokens']]==["don't",',']


def test_neutral_digits_do_not_require_cursive_anchors():
    s=words(('2','ru','character'),('6','ru','character'),(',', 'ru','character'))
    p=plan_text(s,'26,','ru')
    assert [t['text'] for t in p['tokens']]==['2','6',',']
    assert not any(t['join_previous'] for t in p['tokens'])


def test_decomposed_unicode_word_stays_exact():
    s=words(('Cafe\u0301','en','word'),('!','en','character'))
    assert [t['text'] for t in plan_text(s,'Cafe\u0301!','en')['tokens']]==['Cafe\u0301','!']


def test_auto_does_not_select_mislabelled_latin_word():
    s=words(('hello','ru','word'),('hello','en','word'))
    assert plan_text(s,'hello','auto')['tokens'][0]['sample_id']=='sample01'


@pytest.mark.parametrize('text,expected',[("don't",['do',"n't"]),("don't,",['do',"n't",','])])
def test_fragment_contraction_remains_composable(text,expected):
    s=words(('do','en','fragment'),("n't",'en','fragment'),(',', 'en','character'))
    assert [t['text'] for t in plan_text(s,text,'en')['tokens']]==expected


def test_word_sample_with_punctuation_wins_over_letter_fallback():
    s=words(('Hello','en','word'),('H','en','character'),('e','en','character'),
            ('l','en','character'),('o','en','character'),('!','en','character'))
    assert [t['text'] for t in plan_text(s,'Hello!','en')['tokens']]==['Hello','!']


@pytest.fixture
def mixed_library(tmp_path):
    db=Store(tmp_path/'library')
    im=Image.new('RGB',(140,90),'white');d=ImageDraw.Draw(im)
    d.line([(20,60),(30,25),(40,60),(50,25),(60,60)],fill=(30,55,150),width=3)
    raw=io.BytesIO();im.save(raw,format='PNG');s=db.import_bytes(raw.getvalue(),'SYNTHETIC-TEST.png',synthetic=True)
    for text,language in [('Привет','ru'),(',', 'ru'),('hello','en'),('!','en')]:
        a=db.add_sample(s['id'],[15,20,65,65],text=text,language=language,
                        kind='word' if text.isalpha() else 'character',baseline=36)
        db.update_sample(a['id'],status='verified',reviewer='synthetic-fixture')
    return db


def test_full_mixed_render_preserves_geometry_and_provenance(mixed_library,tmp_path):
    db=mixed_library;path=tmp_path/'mixed.png';text='Привет, hello!\nhello!'
    m=render(db,text,'auto',path,width=400,height=300,mode='experimental-synthesis')
    assert validate_manifest(m,db,path)['passed']
    assert [p['text'] for p in m['placements']]==['Привет',',','hello','!','hello','!']
    assert len({p['group'] for p in m['placements'][:2]})==1
    assert all(p['x']+p['width']<=400 for p in m['placements'])
    with pytest.raises(CoverageError):render(db,text,'auto',path,width=400,height=300)


def test_mixed_render_api_and_cli(mixed_library,tmp_path):
    db=mixed_library;c=create_app(db).test_client();text='Привет, hello!'
    assert c.post('/api/plan',json={'text':text,'language':'auto','mode':'experimental-synthesis'}).status_code==200
    assert c.post('/api/render',json={'text':text,'language':'auto','mode':'experimental-synthesis'}).status_code==200
    f=tmp_path/'text.txt';f.write_text(text,'utf-8');out=tmp_path/'cli.png'
    p=subprocess.run([sys.executable,'-m','hre','--library',str(db.root),'render','--text',str(f),
                      '--language','auto','--mode','experimental-synthesis','--output',str(out)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(out.with_suffix('.json').read_text('utf-8'))['text']==text
