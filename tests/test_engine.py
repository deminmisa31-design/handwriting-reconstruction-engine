import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from hre.store import Store
from hre.imaging import extract_ink, segment, metrics, perspective, mesh_warp, composite_linear
from hre.compose import plan_text, render, CoverageError
from hre.web import create_app


@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'тетрадь.png'
    im = Image.new('RGB', (320, 180), 'white')
    d = ImageDraw.Draw(im)
    for x in (25, 50, 100, 130):
        d.line([(x, 35), (x+8, 65), (x+15, 40)], fill=(35, 50, 140), width=3)
        d.line([(x, 110), (x+8, 140), (x+15, 115)], fill=(35, 50, 140), width=3)
    im.save(p)
    return p


@pytest.fixture
def db(tmp_path):
    return Store(tmp_path / 'library')


def sample(db, source, text='test', language='en', kind='word', **kw):
    s = db.import_file(source)
    item = db.add_sample(s['id'], [20, 25, 150, 75], text=text, language=language, kind=kind, **kw)
    return db.update_sample(item['id'], status='verified', reviewer='test-author')


def test_import_is_immutable_deduplicated_and_auditable(db, source):
    before = source.read_bytes()
    a = db.import_file(source)
    assert a['sha256'] == hashlib.sha256(before).hexdigest()
    assert a['width'] == 320 and a['original_name'] == 'тетрадь.png'
    assert db.import_file(source)['id'] == a['id']
    assert source.read_bytes() == before
    assert db.file(a['original']).read_bytes() == before


def test_exif_orientation_preserved_in_metadata(db, tmp_path):
    p = tmp_path/'rotated.jpg'
    exif = Image.Exif(); exif[274] = 6
    Image.new('RGB', (40, 20), 'white').save(p, exif=exif)
    s = db.import_file(p)
    assert (s['width'], s['height'], s['exif_orientation']) == (20, 40, 6)


def test_verification_requires_transcription_and_reviewer(db, source):
    s = db.import_file(source)
    a = db.add_sample(s['id'], [20,25,150,75])
    assert a['status'] == 'candidate'
    with pytest.raises(ValueError): db.update_sample(a['id'], status='verified', reviewer='me')
    with pytest.raises(ValueError): db.update_sample(a['id'], text='word', status='verified')
    a = db.update_sample(a['id'], text='word', status='verified', reviewer='me')
    assert a['status'] == 'verified'
    assert db.update_sample(a['id'], text='changed')['status'] == 'transcribed'
    assert len(db.audit(a['id'])) >= 3


def test_crop_bounds_and_reanalysis_preserve_manual(db, source):
    s = db.import_file(source)
    with pytest.raises(ValueError): db.add_sample(s['id'], [-1,0,999,12])
    with pytest.raises(ValueError): db.add_sample(s['id'], [0,0,0,10])
    a = sample(db, source)
    db.analyze(s['id']); db.analyze(s['id'])
    assert db.sample(a['id'])['status'] == 'verified'
    items = db.samples()
    assert len({(i['source_id'], tuple(i['bbox']), i['origin']) for i in items}) == len(items)


def test_zip_cannot_escape_and_images_import(db, source):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('../escape.png', source.read_bytes())
    with pytest.raises(ValueError): db.import_zip(stream.getvalue())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z: z.writestr('inside/тест.png', source.read_bytes())
    assert len(db.import_zip(stream.getvalue())) == 1


def test_alpha_segmentation_and_stroke_metrics(source):
    rgb = np.array(Image.open(source))
    alpha, info = extract_ink(rgb)
    assert alpha[45:60,25:45].max() > .5
    assert alpha[0:20].max() < .05
    lines, words = segment(alpha)
    assert len(lines) == 2 and len(words) >= 2
    m = metrics(alpha)
    assert m['components'] >= 4 and m['stroke_width_median'] > 0


def test_longest_verified_language_specific_and_exact_text(db, source):
    a = sample(db,source)
    plan = plan_text(db, 'test  test\n', 'en')
    assert plan['text'] == 'test  test\n'
    assert [x['sample_id'] for x in plan['tokens'] if x['kind']=='sample'] == [a['id'], a['id']]
    with pytest.raises(CoverageError): plan_text(db,'test missing','en')
    with pytest.raises(CoverageError): plan_text(db,'test','ru')
    db.update_sample(a['id'], status='rejected')
    with pytest.raises(CoverageError): plan_text(db,'test','en')


def test_ru_fragment_join_requires_matching_anchors(db, source):
    sample(db,source,'при','ru','fragment')
    sample(db,source,'вет','ru','fragment')
    with pytest.raises(CoverageError): plan_text(db,'привет','ru')
    for a in db.samples():
        db.update_sample(a['id'],entry=[0,30],exit=[130,30],status='verified',reviewer='author')
    assert len(plan_text(db,'привет','ru')['tokens']) == 2


def test_render_provenance_deterministic_and_overflow(db, source, tmp_path):
    sample(db,source)
    out = tmp_path/'render.png'
    manifest = render(db,'test test','en',out,width=500,height=250)
    first = out.read_bytes()
    assert manifest['text'] == 'test test' and len(manifest['placements']) == 2
    assert manifest['placements'][0]['source_sha256']
    assert json.loads(out.with_suffix('.json').read_text('utf-8'))['text']=='test test'
    render(db,'test test','en',out,width=500,height=250)
    assert out.read_bytes() == first
    with pytest.raises(ValueError): render(db,'test '*20,'en',out,width=200,height=100)


def test_geometric_identity_and_linear_composite():
    im = np.full((30,40,3),255,np.uint8)
    im[5:10,5:10] = 0
    warped, H = perspective(im, [[0,0],[39,0],[39,29],[0,29]], (40,30))
    assert np.array_equal(warped, im)
    yy,xx = np.indices(im.shape[:2],dtype=np.float32)
    assert np.array_equal(mesh_warp(im,xx,yy),im)
    out = composite_linear(np.ones((1,1,3)),np.ones((1,1)),(0,0,0))
    assert out.max()==0


def test_composite_has_no_bright_or_dark_halo_on_controlled_fixture():
    paper=np.full((15,20,3),.75);alpha=np.linspace(0,1,300).reshape(15,20)
    ink=np.array([.1,.2,.5]);out=composite_linear(paper,alpha,ink)
    assert np.isfinite(out).all() and np.all(out>=ink-1e-10) and np.all(out<=paper+1e-10)
    assert np.allclose(out[0,0],paper[0,0]) and np.allclose(out[-1,-1],ink)


def test_web_import_and_edit_api(db,source):
    app = create_app(db); c = app.test_client()
    assert c.get('/').status_code==200
    r=c.post('/api/import',data={'file':(io.BytesIO(source.read_bytes()),'тетрадь.png')})
    assert r.status_code==200
    sid=r.json[0]['id']
    r=c.post('/api/samples',json={'source_id':sid,'bbox':[20,25,150,75],'text':'test','language':'en','kind':'word'})
    assert r.status_code==200
    aid=r.json['id']
    assert c.patch('/api/samples/'+aid,json={'status':'verified','reviewer':'author','expected_revision':r.json['revision']}).status_code==200
    assert c.get('/files/'+db.sources()[0]['page']).status_code==200
    assert c.post('/api/render',json={'text':'test','language':'en'}).status_code==200
    assert c.post('/api/render',json={'text':'missing','language':'en'}).status_code==422
    assert c.get('/files/../../pyproject.toml').status_code==404
    assert c.post('/api/samples',json={'source_id':sid,'bbox':[-1,0,10,10]}).status_code==400


def test_words_share_baseline_despite_different_crop_heights(db,source,tmp_path):
    a=sample(db,source,'one')
    a=db.update_sample(a['id'],baseline=35,status='verified',reviewer='author')
    sid=a['source_id']
    b=db.add_sample(sid,[20,20,150,80],text='two',language='en',baseline=45)
    db.update_sample(b['id'],status='verified',reviewer='author')
    m=render(db,'one two','en',tmp_path/'x.png',width=500,height=250)
    assert len({p['y']+p['baseline'] for p in m['placements']})==1


def test_poly_validation_hash_tampering_and_origins(db,source,tmp_path):
    a=sample(db,source)
    with pytest.raises(ValueError): db.update_sample(a['id'],polygon=[[20,25],[150,75],[20,75],[150,25]])
    c=create_app(db).test_client()
    assert c.post('/api/render',json={'text':'test'},headers={'Origin':'https://evil.invalid'}).status_code==403
    assert c.get('/api/state',headers={'Host':'evil.invalid'}).status_code==403
    s=db.source(a['source_id']);db.file(s['page']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): render(db,'test','en',tmp_path/'x.png')


def test_stale_revision_is_rejected_and_png_is_immutable(db,source):
    a=sample(db,source)
    old_alpha=db.file(a['alpha']).read_bytes()
    changed=db.update_sample(a['id'],bbox=[20,20,150,80],expected_revision=a['revision'])
    assert changed['alpha']!=a['alpha']
    assert db.file(a['alpha']).read_bytes()==old_alpha
    with pytest.raises(ValueError,match='revision|Concurrent'):
        db.update_sample(a['id'],text='stale',expected_revision=a['revision'])
    c=create_app(db).test_client()
    assert c.patch('/api/samples/'+a['id'],json={'text':'unsafe'}).status_code==400


def test_render_rejects_plan_revision_race(db,source,tmp_path,monkeypatch):
    import hre.compose as composer
    a=sample(db,source)
    real=composer.plan_text
    def racing_plan(*args,**kw):
        p=real(*args,**kw)
        db.update_sample(a['id'],text='changed',status='verified',reviewer='author')
        return p
    monkeypatch.setattr(composer,'plan_text',racing_plan)
    with pytest.raises(ValueError,match='revision|changed'):
        composer.render(db,'test','en',tmp_path/'race.png')


def test_losing_edit_never_corrupts_winner_alpha(db,source,monkeypatch):
    a=sample(db,source)
    real=db._crop
    winner={}
    def race(item,alpha_info=None):
        monkeypatch.setattr(db,'_crop',real)
        winner.update(db.update_sample(a['id'],bbox=[20,20,155,80]))
        real(item,alpha_info)
    monkeypatch.setattr(db,'_crop',race)
    with pytest.raises(ValueError,match='Concurrent'):
        db.update_sample(a['id'],bbox=[20,30,160,85])
    assert hashlib.sha256(db.file(winner['alpha']).read_bytes()).hexdigest()==winner['alpha_sha256']
    assert db.sample(a['id'])['bbox']==winner['bbox']


def test_closed_database_handles_allow_move_on_windows(db,source,tmp_path):
    sample(db,source)
    moved=tmp_path/'moved.sqlite3'
    db.db.rename(moved)
    moved.rename(db.db)


def test_multiframe_and_empty_verified_samples_rejected(db,tmp_path):
    p=tmp_path/'multi.tiff'
    Image.new('RGB',(20,20),'white').save(p,save_all=True,append_images=[Image.new('RGB',(20,20),'black')])
    with pytest.raises(ValueError,match='Multi-frame'): db.import_file(p)
    p=tmp_path/'blank.png'; Image.new('RGB',(50,50),'white').save(p)
    s=db.import_file(p); a=db.add_sample(s['id'],[0,0,40,40],text='x',language='en')
    with pytest.raises(ValueError,match='empty'): db.update_sample(a['id'],status='verified',reviewer='author')
