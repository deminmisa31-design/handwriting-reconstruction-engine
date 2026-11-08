import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from .store import Store, save_json


def main():
    parser=argparse.ArgumentParser(description='HRE: local handwriting reconstruction workbench')
    parser.add_argument('--library',default='library',help='Local library folder (default: library)')
    sub=parser.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import',help='Import images/directories/ZIP without changing originals')
    imp.add_argument('paths',nargs='+')
    analyze=sub.add_parser('analyze'); analyze.add_argument('source_id',nargs='?',default='all')
    analyze.add_argument('--experimental-grid',action='store_true')
    serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=8765)
    ren=sub.add_parser('render'); ren.add_argument('--text',required=True,type=Path)
    ren.add_argument('--language',required=True,choices=['ru','en','auto']); ren.add_argument('--output',required=True,type=Path)
    ren.add_argument('--width',type=int,default=1200); ren.add_argument('--height',type=int,default=1600)
    ren.add_argument('--line-height',type=int,default=80)
    ren.add_argument('--mode',choices=['strict-real','experimental-synthesis'],default='strict-real')
    rectify=sub.add_parser('rectify');rectify.add_argument('source_id')
    rectify.add_argument('--corners',required=True,help='JSON [[x,y],...] in TL,TR,BR,BL order')
    rectify.add_argument('--width',required=True,type=int);rectify.add_argument('--height',required=True,type=int)
    validate=sub.add_parser('validate');validate.add_argument('manifest',type=Path)
    report=sub.add_parser('report');report.add_argument('manifest',type=Path)
    inv=sub.add_parser('inventory');inv.add_argument('--language',choices=['ru','en'],required=True)
    sub.add_parser('stats'); sub.add_parser('demo'); sub.add_parser('doctor')
    diag=sub.add_parser('diagnose');diag.add_argument('images',type=Path,nargs='+')
    diag.add_argument('--output',type=Path,required=True,help='Private local diagnostics folder; never upload it')
    export=sub.add_parser('export'); export.add_argument('output',type=Path)
    args=parser.parse_args()
    store=Store(args.library)
    try:
        if args.command=='diagnose':
            from .diagnostics import diagnose_images
            result=diagnose_images(args.images,args.output)
        elif args.command=='import':
            from .store import EXTENSIONS
            results=[]
            for item in args.paths:
                path=Path(item)
                files=sorted(path.rglob('*')) if path.is_dir() else [path]
                for f in files:
                    if not f.is_file(): continue
                    if f.suffix.lower()=='.zip':
                        if f.stat().st_size>50*1024*1024: raise ValueError('ZIP exceeds 50 MiB')
                        results.extend(store.import_zip(f.read_bytes()))
                    elif f.suffix.lower() in EXTENSIONS: results.append(store.import_file(f))
            result=results
        elif args.command=='analyze':
            ids=[s['id'] for s in store.sources()] if args.source_id=='all' else [args.source_id]
            result=[store.analyze(id,include_experimental=args.experimental_grid) for id in ids]
        elif args.command=='serve':
            from .web import create_app
            print(f'HRE local editor: http://127.0.0.1:{args.port}',flush=True)
            create_app(store).run(host='127.0.0.1',port=args.port,debug=False,threaded=False)
            return
        elif args.command=='render':
            from .compose import render
            result=render(store,args.text.read_text(encoding='utf-8-sig'),args.language,args.output,
                          width=args.width,height=args.height,line_height=args.line_height,mode=args.mode)
        elif args.command=='rectify':
            result=store.rectify(args.source_id,json.loads(args.corners),(args.width,args.height))
        elif args.command=='inventory':result=store.inventory(args.language)
        elif args.command in ('validate','report'):
            from .validation import write_report
            result=write_report(args.manifest,store)
        elif args.command=='stats': result=store.stats()
        elif args.command=='export':
            result={'schema':'hre-library-v1','sources':store.sources(),'samples':store.samples(),
                    'audit':{a['id']:store.audit(a['id']) for a in store.samples()}}
            save_json(args.output,result)
        elif args.command=='doctor':
            failures=[]
            for s in store.sources():
                try: store.verify_source(s['id'])
                except ValueError as e: failures.append({'source':s['id'],'error':str(e)})
            from .store import digest
            for a in store.samples():
                if not store.file(a['alpha']).exists() or digest(store.file(a['alpha']).read_bytes())!=a['alpha_sha256']:
                    failures.append({'sample':a['id'],'error':'Sample hash mismatch/missing'})
            result={'python':sys.version,'library':str(store.root),'source_count':len(store.sources()),
                    'failures':failures,'versions':{p:importlib.metadata.version(p) for p in
                    ['numpy','pillow','scipy','opencv-python-headless','scikit-image','flask']}}
        else:
            from .demo import demo
            result=demo(store)
        print(json.dumps(result,ensure_ascii=True,indent=2))
        if args.command=='doctor' and result['failures']: sys.exit(1)
        if args.command in ('validate','report') and not result['passed']:sys.exit(1)
    except (ValueError,OSError) as e:
        print(json.dumps({'error':str(e)},ensure_ascii=True),file=sys.stderr)
        sys.exit(2)


if __name__=='__main__': main()
