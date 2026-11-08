import numpy as np
from PIL import Image, ImageDraw

from hre.imaging import extract_ink, extract_ink_grid_aware, ink_layers


def grid_fixture():
    im=Image.new('RGB',(320,240),(245,244,240));d=ImageDraw.Draw(im)
    for x in range(20,320,20):d.line((x,0,x,239),fill=(150,149,146),width=2)
    for y in range(20,240,20):d.line((0,y,319,y),fill=(150,149,146),width=2)
    truth=Image.new('L',im.size);t=ImageDraw.Draw(truth)
    points=[(45,105),(70,145),(95,105),(120,145),(145,105)]
    d.line(points,fill=(190,200,225),width=3);t.line(points,fill=255,width=3)
    d.ellipse((173,94,177,98),fill=(150,165,210));t.ellipse((173,94,177,98),fill=255)
    return np.array(im),np.array(truth)>0


def test_grid_only_pixels_suppressed_and_pale_crossings_retained():
    rgb,truth=grid_fixture();base,_=extract_ink(rgb)
    candidate,info,mask=extract_ink_grid_aware(rgb)
    grid=(mask>0)&~truth
    assert info['experimental'] and info['grid_line_count']>0
    assert base[grid].mean()>.1  # Fixture must actually expose baseline grid leakage.
    assert candidate[grid].mean()<base[grid].mean()*.35
    crossings=truth&(mask>0)
    assert crossings.any()
    assert candidate[crossings].mean()>=base[crossings].mean()*.8
    assert candidate[94:99,173:178].max()>.1
    assert candidate.dtype==np.float32 and np.isfinite(candidate).all()


def test_black_ink_is_not_erased_when_blue_evidence_absent():
    im=Image.new('RGB',(240,180),'white');d=ImageDraw.Draw(im)
    for x in range(20,240,20):d.line((x,0,x,179),fill=(190,190,190))
    d.line([(40,50),(70,100),(100,50)],fill='black',width=3)
    rgb=np.array(im);base,_=extract_ink(rgb);candidate,info,_=extract_ink_grid_aware(rgb)
    assert np.array_equal(candidate,base)
    assert info['decision']=='fallback_no_blue_evidence'


def test_grid_candidate_is_diagnostic_and_leaves_main_mask_unchanged():
    rgb,_=grid_fixture();base,_=extract_ink(rgb);layers,_=ink_layers(rgb,include_experimental=True)
    assert 'grid_aware_alpha' in layers and 'grid_candidates' in layers
    assert np.array_equal(layers['ink_score'],np.uint8(base*255))


def test_default_analysis_does_not_run_experimental_grid_extraction(monkeypatch):
    import hre.imaging as imaging
    rgb,_=grid_fixture()
    def forbidden(*args,**kw):raise AssertionError('Experimental extraction ran without opt-in')
    monkeypatch.setattr(imaging,'extract_ink_grid_aware',forbidden)
    layers,_=ink_layers(rgb)
    assert 'grid_aware_alpha' not in layers


def test_local_diagnostic_reports_input_hash_without_altering_it(tmp_path):
    import hashlib,json
    from hre.diagnostics import diagnose_images
    rgb,_=grid_fixture();path=tmp_path/'synthetic.png';Image.fromarray(rgb).save(path)
    original=path.read_bytes()
    reports=diagnose_images([path],tmp_path/'private-diagnostics')
    assert reports[0]['input_sha256']==hashlib.sha256(original).hexdigest()
    assert path.read_bytes()==original
    assert (tmp_path/'private-diagnostics/PRIVATE-comparison.jpg').exists()
    assert json.loads((tmp_path/'private-diagnostics/PRIVATE-diagnostics.json').read_text('utf-8'))[0]['experiment']['experimental']


def test_low_chroma_pale_stroke_crossings_match_same_stroke_off_grid():
    # Independent optical-density fixture: grid and ink add, rather than PIL
    # replacing dark ruling with a lighter pen pixel at their intersections.
    h,w=240,320;paper=np.full((h,w,3),245.,dtype=float)
    ruling=np.zeros((h,w),bool);ruling[:,20::20]=True;ruling[20::20,:]=True
    pen=np.zeros((h,w),bool);pen[91:94,35:175]=True
    density=ruling[:,:,None]*np.array([.5,.5,.5])+pen[:,:,None]*np.array([.07,.07,.068])
    density[150:155,250:255]+=np.array([.3,.25,.1])
    rgb=np.uint8(paper*np.exp(-density))
    candidate,info,mask=extract_ink_grid_aware(rgb)
    off=pen&~ruling&(mask==0);cross=pen&ruling
    assert off.any() and cross.any()
    assert candidate[off].mean()>.03
    assert abs(float(candidate[cross].mean()-candidate[off].mean()))<.08


def test_single_long_pen_stroke_is_not_a_periodic_grid():
    im=Image.new('RGB',(320,240),'white');d=ImageDraw.Draw(im)
    d.line((20,100,300,100),fill=(80,90,160),width=3)
    rgb=np.array(im);base,_=extract_ink(rgb);candidate,info,_=extract_ink_grid_aware(rgb)
    assert info['decision']=='fallback_no_periodic_grid'
    assert np.array_equal(candidate,base)
