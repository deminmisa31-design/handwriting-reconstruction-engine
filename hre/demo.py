"""Explicitly synthetic font-rendered fixture, never user handwriting."""
from pathlib import Path
import io

from PIL import Image, ImageDraw, ImageFont

from .compose import render
from .store import save_json


def demo(store):
    candidates=[Path('C:/Windows/Fonts/segoepr.ttf'),Path('C:/Windows/Fonts/arial.ttf'),
                Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font_path=next((p for p in candidates if p.exists()),None)
    if font_path is None: raise ValueError('Demo requires a Unicode font; pass real data through import instead')
    font=ImageFont.truetype(str(font_path),36)
    image=Image.new('RGB',(1000,440),(250,249,244)); draw=ImageDraw.Draw(image)
    for x in range(0,1000,20): draw.line((x,0,x,440),fill=(218,229,233))
    for y in range(0,440,20): draw.line((0,y,1000,y),fill=(218,229,233))
    draw.text((30,15),'SYNTHETIC FONT FIXTURE / NOT USER HANDWRITING',fill=(90,90,90),font=ImageFont.truetype(str(font_path),17))
    specs=[]
    for text,language,x,y in [('hello','en',70,100),('world','en',320,100),('test','en',580,100),
                              ('Привет','ru',70,240),('мир','ru',400,240)]:
        draw.text((x,y),text,font=font,fill=(37,59,142))
        b=draw.textbbox((x,y),text,font=font)
        box=[b[0]-3,b[1]-3,b[2]+3,b[3]+3]
        specs.append((text,language,box))
    stream=io.BytesIO(); image.save(stream,format='PNG')
    source=store.import_bytes(stream.getvalue(),'SYNTHETIC-demo.png','generated:hre.demo',synthetic=True)
    for text,language,box in specs:
        if any(a['source_id']==source['id'] and a['text']==text and a['status']=='verified' for a in store.samples()): continue
        a=store.add_sample(source['id'],box,text=text,language=language,kind='word')
        store.update_sample(a['id'],status='verified',reviewer='synthetic-fixture-labels; not handwriting verification')
    report=store.analyze(source['id'])
    render(store,'hello world\ntest world','en',store.file('renders/SYNTHETIC-english.png'),width=1000,height=600,mode='experimental-synthesis')
    render(store,'Привет мир\nмир Привет','ru',store.file('renders/SYNTHETIC-russian.png'),width=1000,height=600,mode='experimental-synthesis')
    save_json(store.file('SYNTHETIC-NOTICE.json'),{'synthetic':True,'font':str(font_path),
              'purpose':'Software fixture only, not writer-specific reconstruction','source_id':source['id']})
    return {'source_id':source['id'],'synthetic':True,'analysis':report,
            'english':'renders/SYNTHETIC-english.png','russian':'renders/SYNTHETIC-russian.png'}
