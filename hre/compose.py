"""Exact-text sample selection and deterministic page layout."""
import hashlib
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import __version__
from .imaging import composite_linear
from .store import digest, save_json


class CoverageError(ValueError):
    def __init__(self,missing):
        self.missing=missing
        super().__init__('Нет подтвержденных образцов для: '+', '.join(repr(m['text']) for m in missing))


def _can_join(a,b,language):
    if language=='en': return True
    if a.get('exit') is None or b.get('entry') is None: return False
    dy=(a['exit'][1]-a['baseline'])-(b['entry'][1]-b['baseline'])
    tolerance=max(2.,min(5.,a['metrics']['stroke_width_median']))
    return abs(dy)<=tolerance and a['exit'][0]-b['entry'][0]>0


def _word_plan(word,candidates,language):
    # A confirmed whole token always wins over recombining the same text.
    exact=[a for a in candidates if a['text']==word]
    if exact: return [sorted(exact,key=lambda a:(a['kind']!='word',a['id']))[0]]
    parts=[a for a in candidates if a['kind'] in ('fragment','character')]
    states={0:{None:(0.,[])}}
    for i in range(len(word)):
        for cost,path in list(states.get(i,{}).values()):
            for a in parts:
                if not word.startswith(a['text'],i): continue
                if path and not _can_join(path[-1],a,language): continue
                j=i+len(a['text']); score=cost+1+(1/max(len(a['text']),1))
                by_last=states.setdefault(j,{})
                if a['id'] not in by_last or score<by_last[a['id']][0]: by_last[a['id']]=(score,path+[a])
    endings=states.get(len(word),{})
    return min(endings.values(),key=lambda p:p[0])[1] if endings else None


# HRE-MIXED-SCRIPT-PUNCTUATION-V1
# Punctuation is a separate verified glyph, never silently rendered as a font.
def _character_script(ch):
    name = unicodedata.name(ch, '')
    if 'CYRILLIC' in name and ch.isalpha():
        return 'ru'
    if 'LATIN' in name and ch.isalpha():
        return 'en'
    return 'common'


def _split_token_runs(token, candidates):
    """Separate Cyrillic, Latin, digits and individual marks without altering text."""
    runs = []
    i = 0
    while i < len(token):
        # Keep already verified compounds/contractions intact before separating
        # surrounding punctuation; never turn a word prefix into isolated letters.
        embedded = []
        for a in candidates:
            value = a['text']
            if a['kind'] != 'word' or len(value) < 2 or not token.startswith(value, i):
                continue
            scripts = {_character_script(c) for c in value} - {'common'}
            if len(scripts) > 1 or (scripts and a['language'] not in scripts):
                continue
            kind = next(iter(scripts), 'common')
            end = i + len(value)
            if end < len(token) and (_character_script(token[end]) == kind != 'common'
                                    or unicodedata.category(token[end]).startswith('M')):
                continue
            embedded.append((value, kind))
        if embedded:
            value, kind = max(embedded, key=lambda x: len(x[0]))
            runs.append((value, kind, i)); i += len(value)
            continue
        kind = _character_script(token[i])
        j = i + 1
        if kind != 'common':
            while j < len(token) and (_character_script(token[j]) == kind or
                                     unicodedata.category(token[j]).startswith('M')):
                j += 1
        elif token[i].isdigit():
            while j < len(token) and token[j].isdigit():
                j += 1
        # Marks (commas, quotes, dashes...) remain individual characters.
        runs.append((token[i:j], kind, i))
        i = j
    return runs


def _plan_one_token(token, candidates, language, start, group):
    """Return verified sample tokens, plus missing spans, for one whitespace token.

    A verified whole token has priority. Otherwise split into script runs and
    punctuation marks, allowing a word sample plus a separate comma/period.
    """
    scripts = {_character_script(c) for c in token} - {'common'}
    if language != 'auto' and scripts - {language}:
        return [], [{'text': token, 'start': start, 'end': start + len(token),
                     'reason': 'script mismatch; use --language auto'}]
    exact = [a for a in candidates if a['text'] == token and
             (language == 'auto' or a['language'] == language) and
             (len(scripts) != 1 or a['language'] in scripts)]
    if exact:
        chosen = sorted(exact, key=lambda a: (a['kind'] != 'word', a['id']))[0]
        return [{'kind': 'sample', 'sample_id': chosen['id'],
                 'revision': chosen['revision'], 'text': token,
                 'start': start, 'end': start + len(token),
                 'group': group, 'join_previous': False}], []
    # Preserve the original whole-token fragment DP, e.g. do + n't + comma.
    # Script splitting is a fallback, not a reason to discard valid coverage.
    fragment_fallback=None
    if len(scripts) <= 1:
        trial_languages = list(scripts) if scripts else ([language] if language != 'auto' else ['ru','en'])
        for lang in trial_languages:
            path = _word_plan(token, [a for a in candidates if a['language'] == lang], lang if scripts else 'en')
            if path:
                output=[];cursor=start
                for index,a in enumerate(path):
                    join=bool(index and lang=='ru' and _character_script(path[index-1]['text'][-1])=='ru'
                              and _character_script(a['text'][0])=='ru')
                    output.append({'kind':'sample','sample_id':a['id'],'revision':a['revision'],'text':a['text'],
                                   'start':cursor,'end':cursor+len(a['text']),'group':group,'join_previous':join})
                    cursor+=len(a['text'])
                fragment_fallback=output
                break
    runs = _split_token_runs(token, candidates)
    outputs = []
    missing = []
    preceding_script = None
    for i, (part, kind, offset) in enumerate(runs):
        next_script = next((r[1] for r in runs[i+1:] if r[1] != 'common'), None)
        if kind != 'common':
            trial = [kind]
        elif language != 'auto':
            trial = [language]
        else:
            preferred = preceding_script or next_script or 'ru'
            trial = [preferred, 'en' if preferred == 'ru' else 'ru']
        selected = None
        selected_lang = None
        for trial_lang in trial:
            relevant = [a for a in candidates if a['language'] == trial_lang]
            # Neutral digits/marks are discrete shapes, even when their sample
            # metadata belongs to the Russian library; no cursive anchors needed.
            path = _word_plan(part, relevant, trial_lang if kind != 'common' else 'en')
            if path:
                selected = path
                selected_lang = trial_lang
                break
        if selected is None:
            missing.append({'text': part, 'start': start + offset,
                            'end': start + offset + len(part),
                            'reason': 'no verified real sample' })
            continue
        cursor = start + offset
        for index, a in enumerate(selected):
            # Joining is only for consecutive verified Russian cursive fragments,
            # not between an entire word and trailing punctuation.
            join = bool(index and selected_lang == 'ru' and kind == 'ru')
            outputs.append({'kind': 'sample', 'sample_id': a['id'],
                            'revision': a['revision'], 'text': a['text'],
                            'start': cursor, 'end': cursor + len(a['text']),
                            'group': group, 'join_previous': join})
            cursor += len(a['text'])
        if kind != 'common':
            preceding_script = kind
    if missing and fragment_fallback is not None:
        return fragment_fallback,[]
    return outputs, missing


def plan_text(store, text, language, mode='strict-real'):
    if language not in ('ru', 'en', 'auto'):
        raise ValueError('language must be ru, en or auto')
    if not isinstance(text, str) or len(text) > 10000:
        raise ValueError('Text limit: 10000 characters')
    if any((c.isspace() and c not in ' \n') or (ord(c) < 32 and c != '\n') for c in text):
        raise ValueError('Use spaces and LF newlines; tabs/control characters unsupported')
    if mode not in ('strict-real', 'experimental-synthesis'):
        raise ValueError('Unsupported composition mode')
    sources = {s['id']: s for s in store.sources()}
    candidates = sorted(
        [a for a in store.samples()
         if a['status'] == 'verified' and
         (language == 'auto' or a['language'] == language) and
         a['text'] and a['kind'] != 'stroke' and
         (mode != 'strict-real' or not sources[a['source_id']]['synthetic'])],
        key=lambda a: a['id'])
    tokens, missing, group = [], [], 0
    for match in re.finditer(r' +|\n|[^ \n]+', text):
        value, offset = match.group(), match.start()
        if value.isspace():
            tokens.append({'kind': 'newline' if value == '\n' else 'space',
                           'text': value, 'start': offset, 'end': match.end()})
            continue
        planned, unavailable = _plan_one_token(value, candidates, language, offset, group)
        tokens.extend(planned)
        missing.extend(unavailable)
        group += 1
    if missing:
        raise CoverageError(missing)
    if ''.join(t['text'] for t in tokens) != text:
        raise AssertionError('Internal token segmentation must preserve exact input text')
    return {'text': text, 'language': language, 'mode': mode,
            'tokens': tokens, 'engine_version': __version__}


def render(store,text,language,output,width=1200,height=1600,margin=48,
           line_height=80,space=22,grid='squared',grid_step=20,blur=0.,mode='strict-real'):
    width,height,margin,line_height,space,grid_step=map(int,(width,height,margin,line_height,space,grid_step))
    if not 100<=width<=8000 or not 100<=height<=8000 or width*height>20_000_000:
        raise ValueError('Page dimensions out of limits')
    if not 0<=margin<min(width,height)/2 or line_height<10 or space<1 or grid_step<5:
        raise ValueError('Invalid layout settings')
    if grid not in ('squared','ruled','none') or not 0<=float(blur)<=3: raise ValueError('Invalid paper/blur')
    plan=plan_text(store,text,language,mode=mode)
    samples={t['sample_id']:store.sample(t['sample_id']) for t in plan['tokens'] if t['kind']=='sample'}
    for t in plan['tokens']:
        if t['kind']=='sample':
            a=samples[t['sample_id']]
            if a['revision']!=t['revision'] or a['text']!=t['text'] or a['status']!='verified':
                raise ValueError('Sample revision changed during planning; retry composition')
    for a in samples.values():
        store.verify_source(a['source_id'])
        if a['alpha_sha256']!=digest(store.file(a['alpha']).read_bytes()): raise ValueError('Sample alpha hash mismatch')
    # Group samples into words; whitespace remains explicit in source manifest.
    groups=[]
    for token in plan['tokens']:
        if token['kind']=='sample' and groups and isinstance(groups[-1],list) and groups[-1][-1]['group']==token['group']:
            groups[-1].append(token)
        else: groups.append([token] if token['kind']=='sample' else token)
    x=margin; line_top=margin; placements=[]
    for group in groups:
        if isinstance(group,dict):
            if group['kind']=='newline': x=margin; line_top+=line_height
            else: x+=space*len(group['text'])
            continue
        local=[]; pen=0.; prev=None
        for token in group:
            a=samples[token['sample_id']]; w=a['bbox'][2]-a['bbox'][0]; h=a['bbox'][3]-a['bbox'][1]
            # All baselines share a row; anchors tested for vertical compatibility.
            if token['join_previous'] and prev:
                pen=local[-1][1]+prev['exit'][0]-a['entry'][0]
                y=local[-1][2]+prev['exit'][1]-a['entry'][1]
            else: y=-a['baseline']
            local.append((token,pen,y,w,h)); pen+=w; prev=a
        min_x=min(p[1] for p in local); max_x=max(p[1]+p[3] for p in local)
        min_y=min(p[2] for p in local); max_y=max(p[2]+p[4] for p in local)
        word_width=max_x-min_x; word_height=max_y-min_y
        if word_width>width-2*margin or word_height>line_height:
            raise ValueError('Fragment exceeds line size; increase page width/line height or correct crop')
        if x+word_width>width-margin: x=margin; line_top+=line_height
        baseline_y=line_top+round(line_height*.75)
        if baseline_y+min_y<margin or baseline_y+max_y>height-margin:
            raise ValueError('Page overflow: text not rendered; increase page height/line height')
        for token,px,py,w,h in local:
            a=samples[token['sample_id']]; s=store.source(a['source_id'])
            px=int(round(x+px-min_x)); py=int(round(baseline_y+py))
            placements.append({**token,'x':px,'y':py,'width':w,'height':h,
                               'source_id':s['id'],'source_sha256':s['sha256'],
                               'source_bbox':a['bbox'],'source_polygon':a.get('polygon'),
                               'source_derivative_sha256':s['page_sha256'],
                               'original_polygon':a.get('original_polygon'),
                               'normalized_to_original':s.get('normalized_to_original'),
                               'alpha_sha256':a['alpha_sha256'],'baseline':a['baseline'],
                               'entry':a.get('entry'),'exit':a.get('exit'),'ink_color_srgb':a['ink_color_srgb'],
                               'synthetic':s['synthetic']})
        x+=word_width
    paper=Image.new('RGB',(width,height),(250,249,244)); draw=ImageDraw.Draw(paper)
    if grid!='none':
        for gy in range(grid_step,height,grid_step): draw.line((0,gy,width,gy),fill=(216,227,232))
        if grid=='squared':
            for gx in range(grid_step,width,grid_step): draw.line((gx,0,gx,height),fill=(216,227,232))
    canvas=np.array(paper).astype(float)/255
    for p in placements:
        a=samples[p['sample_id']]
        alpha=np.array(Image.open(store.file(a['alpha'])),dtype=float)/255
        x,y,w,h=p['x'],p['y'],p['width'],p['height']
        canvas[y:y+h,x:x+w]=composite_linear(canvas[y:y+h,x:x+w],alpha,a['ink_color_srgb'])
    result=Image.fromarray(np.uint8(np.clip(canvas,0,1)*255))
    if blur: result=result.filter(ImageFilter.GaussianBlur(float(blur)))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    result.save(output,format='PNG')
    manifest={**plan,'placements':placements,'width':width,'height':height,
              'settings':{'margin':margin,'line_height':line_height,'space':space,'grid':grid,
                          'grid_step':grid_step,'global_blur':float(blur),'color_space':'linear RGB compositing'},
              'text_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
              'image_sha256':digest(output.read_bytes()),'generated':True,
              'synthetic_sources':any(p['synthetic'] for p in placements),
              'notice':'Digital composition from verified samples. Not an original handwritten page.'}
    save_json(output.with_suffix('.json'),manifest)
    from .validation import validate_manifest
    save_json(output.with_name(output.stem+'-validation.json'),validate_manifest(manifest,store,output))
    return manifest
