"""Machine-checkable result integrity; no claims about handwriting realism."""
from collections import Counter
import json
from pathlib import Path
from .store import digest, save_json


def validate_manifest(m,store=None,image_path=None):
    errors=[];warnings=[]
    def error(code,message):errors.append({'code':code,'message':message})
    if ''.join(t.get('text','') for t in m.get('tokens',[]))!=m.get('text',''):
        error('text_mismatch','Structured tokens differ from source text')
    placements=m.get('placements',[])
    tokens=[t for t in m.get('tokens',[]) if t.get('kind')=='sample']
    if len(tokens)!=len(placements):error('placement_count','Sample token/placement counts differ')
    for i,p in enumerate(placements):
        x,y,w,h=[p.get(k,0) for k in ('x','y','width','height')]
        if not (w>0 and h>0 and 0<=x and 0<=y and x+w<=m['width'] and y+h<=m['height']):
            error('bounds',f'Placement {i} outside canvas')
        if i<len(tokens) and any(p.get(k)!=tokens[i].get(k) for k in ('sample_id','revision','text','start','end')):
            error('placement_token_mismatch',f'Placement {i} does not match token')
        if store:
            try:
                source=store.verify_source(p['source_id'])
                if source['sha256']!=p['source_sha256'] or source['synthetic']!=p.get('synthetic'):
                    error('provenance',f'Placement {i} source mismatch')
                versions=[e['sample'] for e in store.audit(p['sample_id']) if e['sample']['revision']==p['revision']]
                if not versions or versions[-1]['status']!='verified' or versions[-1]['text']!=p['text']:
                    error('review_status',f'Placement {i} has no matching verified revision')
                else:
                    a=versions[-1]
                    expected={'source_id':a['source_id'],'source_bbox':a['bbox'],
                              'source_polygon':a.get('polygon'),'original_polygon':a.get('original_polygon'),
                              'source_derivative_sha256':source['page_sha256'],
                              'normalized_to_original':source.get('normalized_to_original'),
                              'baseline':a['baseline'],'entry':a.get('entry'),'exit':a.get('exit')}
                    if any(p.get(k)!=v for k,v in expected.items()):
                        error('provenance_binding',f'Placement {i} differs from its audited source geometry/transform')
                    if digest(store.file(a['alpha']).read_bytes())!=p['alpha_sha256'] or a['alpha_sha256']!=p['alpha_sha256']:
                        error('alpha_hash',f'Placement {i} mask mismatch')
                    if m.get('mode')=='strict-real' and source['synthetic']:
                        error('strict_real',f'Placement {i} contains a synthetic source in strict-real mode')
            except (ValueError,OSError,KeyError) as e:error('provenance',str(e))
    repeats={id:n for id,n in Counter(p.get('sample_id') for p in placements).items() if n>1}
    if repeats:warnings.append({'code':'reused_variants','counts':repeats,'message':'Limited variation; no invented alternatives'})
    if m.get('synthetic_sources'):warnings.append({'code':'synthetic','message':'Contains explicitly synthetic fixtures'})
    if image_path and digest(Path(image_path).read_bytes())!=m.get('image_sha256'):
        error('image_hash','Rendered image hash mismatch')
    return {'passed':not errors,'errors':errors,'warnings':warnings,
            'checks':['structured_text','placement_count','canvas_bounds','token_order']+
                     (['source_hash','verified_revision','alpha_hash'] if store else []),
            'not_measured':['writer_similarity','optical_realism','OCR_accuracy','perceptual_halo_quality']}


def write_report(manifest_path,store=None):
    path=Path(manifest_path);m=json.loads(path.read_text('utf-8'))
    report=validate_manifest(m,store,path.with_suffix('.png'))
    target=path.with_name(path.stem+'-validation.json');save_json(target,report)
    # Self-contained plain HTML report; escape all user-controlled strings.
    import html
    preview=path.with_suffix('.png').name
    page='<html lang="ru"><meta charset="utf-8"><title>HRE Quality Report</title><h1>HRE Quality Report</h1>'
    page+='<p>Техническая проверка; не оценка сходства почерка.</p><img style="max-width:100%" src="'+html.escape(preview,quote=True)+'">'
    page+='<h2>Проверки</h2><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>Происхождение размещений</h2><pre>'+html.escape(json.dumps(m.get('placements',[]),ensure_ascii=False,indent=2))+'</pre></html>'
    target.with_suffix('.html').write_text(page,'utf-8')
    return report
