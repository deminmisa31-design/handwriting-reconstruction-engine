"""Loopback-only Flask editor with no external assets."""
from pathlib import Path
import uuid

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from PIL import UnidentifiedImageError

from .compose import CoverageError, plan_text, render
from .store import MAX_BYTES, save_json


def create_app(store):
    app=Flask(__name__,static_folder='static',static_url_path='/static')
    app.config.update(MAX_CONTENT_LENGTH=MAX_BYTES+1024*1024)

    @app.before_request
    def local_only():
        if request.host.split(':')[0] not in ('127.0.0.1','localhost','[::1]'):
            return jsonify(error='Loopback host required'),403
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('Origin')
            if origin and origin not in ('http://'+request.host,'https://'+request.host):
                return jsonify(error='Cross-origin write rejected'),403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(Exception)
    def errors(e):
        if isinstance(e,HTTPException): return jsonify(error=e.description),e.code
        if isinstance(e,CoverageError): return jsonify(error=str(e),missing=e.missing),422
        if isinstance(e,(ValueError,KeyError,TypeError,UnidentifiedImageError)):
            return jsonify(error=str(e)),400
        app.logger.exception('HRE operation failed')
        return jsonify(error='Operation failed. See local server log.'),500

    @app.get('/')
    def index(): return app.send_static_file('index.html')

    @app.get('/api/state')
    def state(): return jsonify(sources=store.sources(),samples=store.samples(),stats=store.stats(),
                               inventory={lang:store.inventory(lang) for lang in ('ru','en')})

    @app.post('/api/import')
    def imports():
        file=request.files['file']; data=file.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES: raise ValueError('File exceeds 50 MiB')
        name=file.filename or 'upload'
        return jsonify(store.import_zip(data) if name.lower().endswith('.zip') else [store.import_bytes(data,name)])

    @app.post('/api/analyze/<id>')
    def analyze(id): return jsonify(store.analyze(id))

    @app.post('/api/rectify/<id>')
    def rectify(id):
        data=request.get_json()
        return jsonify(store.rectify(id,data['corners'],data['output_size']))

    @app.get('/api/analysis/<id>')
    def analysis(id):
        store.source(id)
        path=store.file('analysis/'+id+'.json')
        if not path.exists(): return jsonify(error='Run analysis first'),404
        import json
        return jsonify(json.loads(path.read_text('utf-8')))

    @app.post('/api/samples')
    def add_sample(): return jsonify(store.add_sample(**request.get_json()))

    @app.patch('/api/samples/<id>')
    def edit_sample(id):
        data=request.get_json()
        if 'expected_revision' not in data: raise ValueError('expected_revision is required; reload sample')
        return jsonify(store.update_sample(id,**data))

    @app.get('/api/audit/<id>')
    def audit(id): return jsonify(store.audit(id))

    @app.get('/api/sample/<id>')
    def sample(id):return jsonify(store.sample(id))

    @app.post('/api/plan')
    def plan():
        data=request.get_json()
        return jsonify(plan_text(store,data.get('text',''),data.get('language','ru'),data.get('mode','strict-real')))

    @app.post('/api/render')
    def compose():
        data=request.get_json(); name=uuid.uuid4().hex
        settings={k:v for k,v in data.items() if k in {'width','height','margin','line_height','space','grid','grid_step','blur','mode'}}
        manifest=render(store,data.get('text',''),data.get('language','ru'),store.file('renders/'+name+'.png'),**settings)
        return jsonify(image='renders/'+name+'.png',manifest='renders/'+name+'.json',report=manifest)

    @app.get('/api/export')
    def export():
        return jsonify(schema='hre-library-v1',sources=store.sources(),samples=store.samples(),
                       audit={a['id']:store.audit(a['id']) for a in store.samples()})

    @app.get('/api/quality/<id>')
    def quality(id):
        source=store.source(id)
        import numpy as np
        from PIL import Image
        from .document import quality_report
        return jsonify(quality=source.get('quality_metrics') or quality_report(np.array(Image.open(store.file(source['page'])).convert('RGB'))),regions=source.get('detected_document_regions',[]),
                       transform=source.get('original_to_normalized'),grid=source.get('grid'),
                       note='Uncalibrated measurements, not a writer similarity score')

    @app.get('/files/<path:path>')
    def files(path):
        p=Path(path)
        if not p.parts or p.parts[0] not in ('pages','analysis','samples','renders') or '..' in p.parts:
            return jsonify(error='Not found'),404
        try: target=store.file(path)
        except ValueError: return jsonify(error='Not found'),404
        return send_from_directory(store.root,target.relative_to(store.root).as_posix())

    return app
