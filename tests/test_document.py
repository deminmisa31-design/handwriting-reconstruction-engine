import io
import numpy as np
import pytest
from PIL import Image,ImageDraw

from hre.document import exif_matrix, transform_points, quality_report, detect_grid
from hre.store import Store
from hre.compose import plan_text, CoverageError
from hre.validation import validate_manifest


@pytest.mark.parametrize('orientation',range(1,9))
def test_exif_coordinate_roundtrip(orientation):
    H=exif_matrix(orientation,80,50)
    p=np.array([[0,0],[80,50],[17.3,22.8]])
    assert np.allclose(transform_points(transform_points(p,H),np.linalg.inv(H)),p)


def test_exif_pixel_transform_matches_pillow():
    from PIL import ImageOps
    rgb=np.zeros((5,8,3),np.uint8);rgb[1,2]=255
    for o in range(1,9):
        im=Image.fromarray(rgb);im.getexif()[274]=o
        out=np.array(ImageOps.exif_transpose(im))
        x,y=transform_points([[2.5,1.5]],exif_matrix(o,8,5))[0]
        assert np.all(out[int(y),int(x)]==255)


def test_quality_grid_and_duplicate_metadata(tmp_path):
    im=Image.new('RGB',(300,240),'white');d=ImageDraw.Draw(im)
    for x in range(10,300,20):d.line((x,0,x,239),fill=(185,205,220))
    for y in range(10,240,20):d.line((0,y,299,y),fill=(185,205,220))
    rgb=np.array(im)
    g=detect_grid(rgb)
    assert abs(g['grid_spacing_x']-20)<2 and abs(g['grid_spacing_y']-20)<2
    q=quality_report(rgb)
    assert q['overall_score'] is None and q['sharpness_laplacian_variance']>=0
    db=Store(tmp_path/'lib');stream=io.BytesIO();im.save(stream,format='PNG')
    a=db.import_bytes(stream.getvalue(),'grid.png')
    assert a['perceptual_hash'] and a['quality_metrics'] and a['coordinate_spaces']


def test_rectified_view_preserves_source_and_maps_crop(tmp_path):
    db=Store(tmp_path/'lib')
    im=Image.new('RGB',(200,150),'white');ImageDraw.Draw(im).line((40,40,100,90),fill='blue',width=4)
    b=io.BytesIO();im.save(b,format='PNG');s=db.import_bytes(b.getvalue(),'page.png')
    original=db.file(s['original']).read_bytes()
    view=db.rectify(s['id'],[[10,10],[190,10],[190,140],[10,140]],(180,130))
    assert view['root_source_id']==s['id'] and view['id']!=s['id']
    assert db.file(s['original']).read_bytes()==original
    a=db.add_sample(view['id'],[20,20,100,100],text='x',language='en')
    assert len(a['original_polygon'])==4 and a['evidence']
    assert np.allclose(transform_points([[0,0]],view['normalized_to_original']),[[10,10]],atol=1)


def test_strict_real_excludes_fixture_and_inventory_does_not_guess_letters(tmp_path):
    db=Store(tmp_path/'lib');im=Image.new('RGB',(100,80),'white')
    ImageDraw.Draw(im).line((20,20,70,60),fill='blue',width=3)
    b=io.BytesIO();im.save(b,format='PNG');s=db.import_bytes(b.getvalue(),'SYNTHETIC.png',synthetic=True)
    a=db.add_sample(s['id'],[10,10,80,70],text='hello',language='en')
    db.update_sample(a['id'],status='verified',reviewer='fixture')
    with pytest.raises(CoverageError):plan_text(db,'hello','en')
    assert plan_text(db,'hello','en',mode='experimental-synthesis')['tokens']
    inventory=db.inventory('en')
    assert inventory['characters']['h']['verified_real_variants']==0
    assert inventory['word_count']==1 and 'h' in inventory['missing_real_characters']


def test_validation_reports_text_mismatch():
    report=validate_manifest({'text':'abc','tokens':[{'text':'abd'}], 'placements':[], 'width':100,'height':100})
    assert not report['passed'] and any(e['code']=='text_mismatch' for e in report['errors'])


def test_rectify_api_persists_after_reopen(tmp_path):
    from hre.web import create_app
    db=Store(tmp_path/'lib');im=Image.new('RGB',(100,80),'white')
    b=io.BytesIO();im.save(b,format='PNG');s=db.import_bytes(b.getvalue(),'page.png')
    c=create_app(db).test_client()
    r=c.post('/api/rectify/'+s['id'],json={'corners':[[0,0],[99,0],[99,79],[0,79]],'output_size':[100,80]})
    assert r.status_code==200
    assert Store(tmp_path/'lib').source(r.json['id'])['parent_view_id']==s['id']
    assert c.get('/api/quality/'+s['id']).json['quality']['overall_score'] is None


def test_validator_binds_source_geometry_to_verified_sample(tmp_path):
    from hre.compose import render
    db=Store(tmp_path/'lib');sources=[]
    for color in ['blue','black']:
        im=Image.new('RGB',(100,80),'white');ImageDraw.Draw(im).line((20,20,70,60),fill=color,width=3)
        b=io.BytesIO();im.save(b,format='PNG');sources.append(db.import_bytes(b.getvalue(),color+'.png'))
    a=db.add_sample(sources[0]['id'],[10,10,80,70],text='x',language='en')
    db.update_sample(a['id'],status='verified',reviewer='test')
    path=tmp_path/'result.png';m=render(db,'x','en',path,width=300,height=250,line_height=100)
    m['placements'][0].update(source_id=sources[1]['id'],source_sha256=sources[1]['sha256'],
                              source_bbox=[0,0,1,1],source_derivative_sha256='wrong')
    assert not validate_manifest(m,db,path)['passed']


def test_polygon_provenance_is_actual_selection(tmp_path):
    db=Store(tmp_path/'lib');im=Image.new('RGB',(100,80),'white')
    ImageDraw.Draw(im).line((20,20,70,60),fill='blue',width=3)
    b=io.BytesIO();im.save(b,format='PNG');s=db.import_bytes(b.getvalue(),'photo.png')
    polygon=[[10,10],[80,10],[80,70]]
    a=db.add_sample(s['id'],[10,10,80,70],text='x',language='en',polygon=polygon)
    assert np.allclose(a['original_polygon'],polygon)
