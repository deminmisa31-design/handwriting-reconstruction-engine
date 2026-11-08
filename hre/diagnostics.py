"""Local-only before/after diagnostics; never imports or verifies glyph labels."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .imaging import extract_ink, extract_ink_grid_aware
from .store import save_json


def diagnose_images(paths, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    reports=[];rows=[]
    for path in map(Path,paths):
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        with Image.open(path) as im:
            if im.width*im.height>40_000_000:raise ValueError('Diagnostic input exceeds 40 MP')
            rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
        before,base_info=extract_ink(rgb);after,info,grid=extract_ink_grid_aware(rgb)
        key=sha[:16]
        for suffix,arr in [('baseline',np.uint8(before*255)),('experimental',np.uint8(after*255)),('grid',grid)]:
            Image.fromarray(arr).save(output/(key+'-'+suffix+'.png'))
        preview=Image.new('RGB',(rgb.shape[1]*3,rgb.shape[0]+40),'#f5f5f5');draw=ImageDraw.Draw(preview)
        for i,(title,arr) in enumerate([('Source input',rgb),('Baseline: unverified',np.uint8(before*255)),
                                       ('Experimental: unverified',np.uint8(after*255))]):
            draw.text((i*rgb.shape[1]+6,10),title,fill='black')
            preview.paste(Image.fromarray(arr).convert('RGB'),(i*rgb.shape[1],40))
        preview.thumbnail((1800,1800));preview.save(output/(key+'-comparison.jpg'),quality=95)
        rows.append(preview)
        reports.append({'input_name':path.name,'input_sha256':sha,'input_size':[rgb.shape[1],rgb.shape[0]],
                        'baseline_fraction_alpha_gt_0_2':float((before>.2).mean()),
                        'experimental_fraction_alpha_gt_0_2':float((after>.2).mean()),
                        'baseline':base_info,'experiment':info,
                        'warning':'Foreground reduction is not accuracy or stroke recall; no verified manual mask provided.'})
    if rows:
        combined=Image.new('RGB',(max(r.width for r in rows),sum(r.height for r in rows)),'#f5f5f5')
        y=0
        for row in rows:combined.paste(row,(0,y));y+=row.height
        combined.save(output/'PRIVATE-comparison.jpg',quality=95)
    save_json(output/'PRIVATE-diagnostics.json',reports)
    return reports
