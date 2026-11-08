"""Immutable content-addressed source storage, derivatives and revisioned labels."""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from skimage.morphology import skeletonize

from .imaging import extract_ink, metrics, segment, valid_polygon
from .document import exif_matrix, transform_points, quality_report, perceptual_hash, page_candidates, detect_grid

MAX_BYTES=50*1024*1024
MAX_PIXELS=40_000_000
EXTENSIONS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}


def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


class Store:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ('originals','pages','analysis','samples','renders'):
            (self.root/name).mkdir(exist_ok=True)
        self.db=self.root/'library.sqlite3'
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, sample_id TEXT, data TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30)
        try:
            c.execute('PRAGMA journal_mode=WAL')
            with c:
                yield c
        finally:
            c.close()

    def file(self,relative):
        p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root): raise ValueError('Path escapes library')
        return p

    def _get(self,table,id):
        with self.connect() as c:
            row=c.execute(f'SELECT data FROM {table} WHERE id=?',(id,)).fetchone()
        if row is None: raise ValueError('Unknown '+table+' id')
        return json.loads(row[0])

    def _list(self,table):
        with self.connect() as c: rows=c.execute(f'SELECT data FROM {table} ORDER BY rowid').fetchall()
        return [json.loads(r[0]) for r in rows]

    def source(self,id): return self._get('sources',id)
    def sources(self): return self._list('sources')
    def sample(self,id): return self._get('samples',id)
    def samples(self): return self._list('samples')

    def import_file(self,path):
        path=Path(path).resolve()
        if path.stat().st_size>MAX_BYTES: raise ValueError('Image exceeds 50 MiB')
        return self.import_bytes(path.read_bytes(),path.name,str(path))

    def import_bytes(self,data,name,original_path=None,synthetic=False):
        if len(data)>MAX_BYTES: raise ValueError('Image exceeds 50 MiB')
        sha=digest(data)
        if any(s['id']==sha for s in self.sources()): return self.source(sha)
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width*opened.height>MAX_PIXELS: raise ValueError('Image exceeds 40 megapixels')
            if getattr(opened,'n_frames',1)!=1: raise ValueError('Multi-frame images must be split explicitly')
            fmt=opened.format
            orientation=int(opened.getexif().get(274,1))
            raw_size=list(opened.size)
            icc=opened.info.get('icc_profile')
            oriented=ImageOps.exif_transpose(opened)
            steps=[{'op':'exif_transpose','orientation':orientation}]
            if icc:
                try:
                    oriented=ImageCms.profileToProfile(oriented,ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                                      ImageCms.createProfile('sRGB'),outputMode='RGB')
                    steps.append({'op':'icc_to_srgb'})
                except (ValueError, OSError, ImageCms.PyCMSError) as e:
                    steps.append({'op':'icc_failed_assume_srgb','reason':str(e)})
            if oriented.mode=='RGBA' or 'transparency' in oriented.info:
                layer=oriented.convert('RGBA'); background=Image.new('RGBA',layer.size,'white')
                background.alpha_composite(layer); oriented=background.convert('RGB')
                steps.append({'op':'flatten_transparency','paper':'white'})
            rgb=oriented.convert('RGB')
        ext={ 'JPEG':'.jpg','TIFF':'.tif'}.get(fmt,'.'+str(fmt).lower())
        original='originals/'+sha+ext
        page='pages/'+sha+'.png'
        target=self.file(original)
        if target.exists():
            if digest(target.read_bytes())!=sha: raise ValueError('Original archive integrity failure')
        else:
            with target.open('xb') as f: f.write(data)
        rgb.save(self.file(page))
        metadata={'id':sha,'sha256':sha,'original_name':str(name),'original_path':original_path,
                  'original':original,'page':page,'bytes':len(data),'width':rgb.width,'height':rgb.height,
                  'original_dimensions':raw_size,'exif_orientation':orientation,'image_type':fmt,
                  'icc_profile':{'present':bool(icc),'sha256':digest(icc) if icc else None},
                  'transforms':steps,'imported_at':now(),'synthetic':bool(synthetic)}
        if icc:
            profile='originals/'+sha+'.icc'; self.file(profile).write_bytes(icc)
            metadata['icc_profile']['file']=profile
        metadata['page_sha256']=digest(self.file(page).read_bytes())
        rgb_array=np.array(rgb)
        matrix=exif_matrix(orientation,*raw_size)
        metadata.update(root_source_id=sha,source_id=sha,view_id=sha,
                        perceptual_hash=perceptual_hash(rgb_array),quality_metrics=quality_report(rgb_array),
                        detected_document_regions=page_candidates(rgb_array),color_space='sRGB',
                        coordinate_spaces=['original_image','normalized_image'],
                        original_to_normalized=matrix.tolist(),normalized_to_original=np.linalg.inv(matrix).tolist())
        metadata['similar_sources']=[{'source_id':s['id'],'hamming_distance':
            (int(s['perceptual_hash'],16)^int(metadata['perceptual_hash'],16)).bit_count()}
            for s in self.sources() if s.get('perceptual_hash') and
            (int(s['perceptual_hash'],16)^int(metadata['perceptual_hash'],16)).bit_count()<=6]
        with self.connect() as c:
            c.execute('INSERT INTO sources VALUES (?,?)',(sha,json.dumps(metadata,ensure_ascii=False)))
        return metadata

    def import_zip(self,data):
        if len(data)>MAX_BYTES: raise ValueError('ZIP exceeds 50 MiB')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries=z.infolist()
            if len(entries)>1000: raise ValueError('ZIP exceeds 1000 entries')
            total=0; selected=[]
            for member in entries:
                p=PurePosixPath(member.filename.replace('\\','/'))
                if p.is_absolute() or '..' in p.parts or ':' in member.filename:
                    raise ValueError('Unsafe ZIP path')
                if (member.external_attr>>16)&0o170000==0o120000: raise ValueError('ZIP symlink rejected')
                if member.flag_bits&1: raise ValueError('Encrypted ZIP rejected')
                total+=member.file_size
                if total>200*1024*1024 or member.file_size>MAX_BYTES: raise ValueError('ZIP expanded size limit')
                if not member.is_dir() and p.suffix.lower() in EXTENSIONS: selected.append(member)
            # No extraction to filesystem; path traversal cannot write anything.
            return [self.import_bytes(z.read(m),m.filename,'zip:'+m.filename) for m in selected]

    def verify_source(self,id):
        s=self.source(id)
        if digest(self.file(s['original']).read_bytes())!=s['sha256']:
            raise ValueError('Original hash mismatch')
        if digest(self.file(s['page']).read_bytes())!=s['page_sha256']:
            raise ValueError('Derived page hash mismatch')
        return s

    def alpha(self,source_id):
        s=self.verify_source(source_id)
        rgb=np.array(Image.open(self.file(s['page'])).convert('RGB'))
        return extract_ink(rgb)

    def rectify(self,id,corners,output_size):
        """Create a separately indexed view; existing page/annotations never change."""
        from .imaging import perspective
        s=self.verify_source(id)
        rgb=np.array(Image.open(self.file(s['page'])).convert('RGB'))
        warped,H_center=perspective(rgb,corners,output_size)
        # OpenCV maps pixel centers at integer coordinates. Convert to edge coordinates.
        plus=np.array([[1,0,.5],[0,1,.5],[0,0,1.]])
        minus=np.array([[1,0,-.5],[0,1,-.5],[0,0,1.]])
        H=plus@H_center@minus
        root=s.get('root_source_id',s['sha256'])
        key=digest(json.dumps({'parent':id,'corners':corners,'size':output_size},sort_keys=True).encode())[:16]
        view_id=root+'-view-'+key
        if any(v['id']==view_id for v in self.sources()):return self.source(view_id)
        view=dict(s);page='pages/'+view_id+'.png'
        Image.fromarray(warped).save(self.file(page))
        old_matrix=np.asarray(s.get('original_to_normalized',exif_matrix(s['exif_orientation'],*s['original_dimensions'])))
        matrix=H@old_matrix
        view.update(id=view_id,view_id=view_id,source_id=root,root_source_id=root,parent_view_id=id,
                    page=page,page_sha256=digest(self.file(page).read_bytes()),width=warped.shape[1],height=warped.shape[0],
                    display_name=s['original_name']+' · выровнено',original_to_normalized=matrix.tolist(),
                    normalized_to_original=np.linalg.inv(matrix).tolist(),coordinate_spaces=['original_image','normalized_image','page_local'],
                    quality_metrics=quality_report(warped),detected_document_regions=[],
                    transforms=s['transforms']+[{'op':'manual_perspective','parent_view_id':id,'corners':corners,
                                                'matrix_edge_coordinates':H.tolist(),'opencv_matrix':H_center.tolist()}])
        with self.connect() as c:c.execute('INSERT INTO sources VALUES (?,?)',(view_id,json.dumps(view,ensure_ascii=False)))
        return view

    def analyze(self,id,include_experimental=False):
        s=self.verify_source(id)
        alpha,info=self.alpha(id)
        prefix='analysis/'+id
        Image.fromarray(np.uint8(alpha*255)).save(self.file(prefix+'-alpha.png'))
        Image.fromarray(np.uint8(skeletonize(alpha>.2))*255).save(self.file(prefix+'-skeleton.png'))
        lines,words=segment(alpha)
        overlay=Image.open(self.file(s['page'])).convert('RGB')
        from PIL import ImageDraw
        from .imaging import ink_layers
        draw=ImageDraw.Draw(overlay)
        for b in lines: draw.rectangle(b,outline='#d79530',width=2)
        for b in words: draw.rectangle(b,outline='#00a995',width=1)
        overlay.save(self.file(prefix+'-overlay.png'))
        report={'source_id':id,'method':info,'metrics':metrics(alpha),'lines':lines,'words':words,
                'alpha':prefix+'-alpha.png','skeleton':prefix+'-skeleton.png','overlay':prefix+'-overlay.png',
                'warning':'Heuristic candidates only, no character recognition.'}
        report['grid']=detect_grid(np.array(Image.open(self.file(s['page'])).convert('RGB')))
        layers,layer_info=ink_layers(np.array(Image.open(self.file(s['page'])).convert('RGB')),
                                    include_experimental=include_experimental)
        if include_experimental:report['experimental_grid']=layer_info['experimental_grid']
        for name,array in layers.items():
            path=prefix+'-'+name+'.png';Image.fromarray(array).save(self.file(path));report[name]=path
        report['ink_score_calibration']='Heuristic score and thresholds, not calibrated probabilities'
        save_json(self.file(prefix+'.json'),report)
        existing={(tuple(a['bbox']),a['origin']) for a in self.samples() if a['source_id']==id}
        for b in words:
            if (tuple(b),'automatic') not in existing:
                self.add_sample(id,b,origin='automatic',_alpha=(alpha,info))
        return report

    def add_sample(self,source_id,bbox,text='',language='ru',kind='word',origin='manual',**kw):
        a={'id':uuid.uuid4().hex,'source_id':source_id,'bbox':bbox,'text':text,'language':language,
           'kind':kind,'origin':origin,'status':'transcribed' if text else 'candidate',
           'revision':1,'created_at':now(),'reviewer':None,'entry':None,'exit':None,'polygon':None}
        for k in ('baseline','entry','exit','polygon'): 
            if k in kw: a[k]=kw[k]
        self._validate(a)
        self._crop(a,kw.get('_alpha'))
        with self.connect() as c:
            c.execute('INSERT INTO samples VALUES (?,?)',(a['id'],json.dumps(a,ensure_ascii=False)))
            self._event(c,a)
        return a

    def _validate(self,a):
        s=self.source(a['source_id'])
        box=a['bbox']
        if not isinstance(box,(list,tuple)) or len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) and int(x)==x for x in box):
            raise ValueError('bbox needs four integer coordinates')
        x0,y0,x1,y1=map(int,box)
        if not (0<=x0<x1<=s['width'] and 0<=y0<y1<=s['height']): raise ValueError('Crop outside page or empty')
        a['bbox']=[x0,y0,x1,y1]
        if a['language'] not in ('ru','en'): raise ValueError('language must be ru or en')
        if a['kind'] not in ('word','fragment','character','stroke'): raise ValueError('Invalid sample kind')
        if a['status'] not in ('candidate','transcribed','verified','rejected'): raise ValueError('Invalid status')
        if not isinstance(a['text'],str) or len(a['text'])>1000 or any(c.isspace() for c in a['text']):
            raise ValueError('Sample text must be one token without whitespace, up to 1000 characters')
        if a['kind']=='character' and len(a['text'])>1: raise ValueError('Character must contain one codepoint')
        if a['status']=='verified' and (not a['text'] or not str(a.get('reviewer') or '').strip()):
            raise ValueError('Verification requires exact text and reviewer')
        w,h=x1-x0,y1-y0
        baseline=a.get('baseline',h*.8)
        if not isinstance(baseline,(int,float)) or not math.isfinite(baseline) or not 0<=baseline<=h:
            raise ValueError('Baseline must be within crop (local y)')
        a['baseline']=float(baseline)
        for k in ('entry','exit'):
            p=a.get(k)
            if p is not None and (not isinstance(p,(list,tuple)) or len(p)!=2 or
                not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) or not 0<=p[0]<=w or not 0<=p[1]<=h):
                raise ValueError(k+' must be local crop [x,y] or null')
        if a.get('polygon') is not None:
            p=valid_polygon(a['polygon'],s['width'],s['height'])
            if (p[:,0]<x0).any() or (p[:,0]>x1).any() or (p[:,1]<y0).any() or (p[:,1]>y1).any():
                raise ValueError('Polygon must lie inside bbox in global coordinates')

    def _crop(self,a,alpha_info=None):
        alpha,info=alpha_info if alpha_info is not None else self.alpha(a['source_id'])
        x0,y0,x1,y1=a['bbox']; crop=alpha[y0:y1,x0:x1].copy()
        if a.get('polygon'):
            mask=np.zeros_like(crop,dtype=np.uint8)
            cv2.fillPoly(mask,[np.rint(np.array(a['polygon'])-[x0,y0]).astype(np.int32)],1)
            crop*=mask
        if a['status']=='verified' and not (crop>.1).any(): raise ValueError('Cannot verify empty ink sample')
        a['metrics']=metrics(crop); a['ink_color_srgb']=info['ink_color_srgb']
        s=self.source(a['source_id'])
        corners=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        inverse=s.get('normalized_to_original',np.linalg.inv(exif_matrix(s['exif_orientation'],*s['original_dimensions'])))
        a['coordinate_space']='normalized_image';a['original_polygon']=transform_points(a.get('polygon') or corners,inverse).tolist()
        a['root_source_id']=s.get('root_source_id',s['sha256'])
        a['origin_type']='synthetic' if s['synthetic'] else 'real'
        a['review_status']={'candidate':'unlabelled','transcribed':'proposed','verified':'manually_verified','rejected':'rejected'}[a['status']]
        # Unmodified RGB evidence is separate from the estimated ink mask.
        evidence=Image.open(self.file(s['page'])).convert('RGB').crop((x0,y0,x1,y1))
        evidence_bytes=io.BytesIO();evidence.save(evidence_bytes,format='PNG');evidence_data=evidence_bytes.getvalue()
        a['evidence_sha256']=digest(evidence_data)
        a['evidence']='samples/'+a['id']+'-evidence-'+a['evidence_sha256']+'.png'
        if not self.file(a['evidence']).exists():
            with self.file(a['evidence']).open('xb') as f:f.write(evidence_data)
        encoded=io.BytesIO()
        Image.fromarray(np.uint8(crop*255)).save(encoded,format='PNG')
        data=encoded.getvalue(); a['alpha_sha256']=digest(data)
        a['alpha']='samples/'+a['id']+'-r'+str(a['revision'])+'-'+a['alpha_sha256']+'.png'
        target=self.file(a['alpha'])
        try:
            with target.open('xb') as f: f.write(data)
        except FileExistsError:
            if digest(target.read_bytes())!=a['alpha_sha256']: raise ValueError('Immutable sample hash mismatch')
        a['updated_at']=now()

    def update_sample(self,id,expected_revision=None,**changes):
        old=self.sample(id); a=dict(old)
        if expected_revision is not None and expected_revision!=old['revision']:
            raise ValueError('Concurrent edit: expected revision no longer current; reload sample')
        allowed={'text','language','kind','status','reviewer','bbox','baseline','polygon','entry','exit'}
        if set(changes)-allowed: raise ValueError('Unknown sample fields')
        a.update(changes)
        content={'text','language','kind','bbox','baseline','polygon','entry','exit'}
        if old['status']=='verified' and any(k in changes and changes[k]!=old.get(k) for k in content):
            a['status']='transcribed' if a['text'] else 'candidate'
            a['reviewer']=None
            if changes.get('status')=='verified': a['status']='verified'; a['reviewer']=changes.get('reviewer')
        a['revision']=old['revision']+1
        self._validate(a); self._crop(a)
        with self.connect() as c:
            # Prevent silently overwriting another editor's revision.
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT data FROM samples WHERE id=?',(id,)).fetchone()
            if json.loads(row[0])['revision']!=old['revision']: raise ValueError('Concurrent edit: reload sample')
            c.execute('UPDATE samples SET data=? WHERE id=?',(json.dumps(a,ensure_ascii=False),id))
            self._event(c,a)
        return a

    def _event(self,c,a):
        c.execute('INSERT INTO events (sample_id,data) VALUES (?,?)',
                  (a['id'],json.dumps({'at':now(),'sample':a},ensure_ascii=False)))

    def audit(self,id):
        with self.connect() as c: rows=c.execute('SELECT data FROM events WHERE sample_id=? ORDER BY seq',(id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def stats(self):
        items=self.samples(); result={}
        for lang in ('ru','en'):
            all_lang=[a for a in items if a['language']==lang]
            verified=[a for a in all_lang if a['status']=='verified']
            def summary(key):
                values=[a['metrics'][key] for a in verified if a['metrics'][key]>0]
                return {'n':len(values),'median':float(np.median(values)) if values else None,
                        'std':float(np.std(values)) if values else None}
            result[lang]={'samples':len(all_lang),'verified':len(verified),
                          'vocabulary':sorted({a['text'] for a in verified}),
                          'height_px':summary('height'),'width_px':summary('width'),
                          'stroke_width_px':summary('stroke_width_median'),
                          'warning':'Descriptive crop statistics only; not a fitted writer model.'}
        return result

    def inventory(self,language):
        if language not in ('ru','en'):raise ValueError('language must be ru or en')
        alphabet=('абвгдеёжзийклмнопрстуфхцчшщъыьэюя' if language=='ru' else 'abcdefghijklmnopqrstuvwxyz')
        alphabet=alphabet+alphabet.upper()+'0123456789.,!?-:;()'
        items=[a for a in self.samples() if a['status']=='verified' and a['language']==language]
        sources={s['id']:s for s in self.sources()}
        chars={}
        for c in alphabet:
            exact=[a for a in items if a['kind']=='character' and a['text']==c]
            real=[a for a in exact if not sources[a['source_id']]['synthetic']]
            chars[c]={'verified_real_variants':len(real),'synthetic_variants':len(exact)-len(real),
                      'independent_sources':len({sources[a['source_id']]['sha256'] for a in real})}
        return {'language':language,'characters':chars,'word_count':sum(a['kind']=='word' for a in items),
                'fragment_count':sum(a['kind']=='fragment' for a in items),
                'missing_real_characters':[c for c,n in chars.items() if n['verified_real_variants']==0],
                'notice':'Letters inside labelled words are not separately verified character variants.'}
