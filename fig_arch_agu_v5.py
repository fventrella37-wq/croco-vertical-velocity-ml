# AGU FiLM CNN architecture figure (Figure 2), version 5: one row of boxes with the block names
# (sizes only for input and output); the layers and parameters are described in the caption. the layers and parameters of every block are listed below the
# diagram, one entry per box. All text at one size S, printing at 8 pt at natural size.
# Usage: python3 fig_arch_agu_v5.py out.svg out.pdf
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
from PIL import ImageFont
import cairosvg, sys
FP={'H':'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf',
    'Hi':'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-italic.otf',
    'Hb':'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf',
    'M':'/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf'}
ATTR={'H':'font-family="TeX Gyre Heros"','Hi':'font-family="TeX Gyre Heros" font-style="italic"',
      'Hb':'font-family="TeX Gyre Heros" font-weight="bold"','M':'font-family="Latin Modern Math"'}
u,v,w,z,r,g,b='\U0001D462','\U0001D463','\U0001D464','\U0001D467','\U0001D45F','\U0001D6FE','\U0001D6FD'
_c={}
def W(t,k,s):
    key=(k,s)
    if key not in _c: _c[key]=ImageFont.truetype(FP[k],size=int(round(s*10)))
    return _c[key].getlength(t)/10.0
def segw(segs,s): return sum(W(t,k,s) for t,k in segs)
def text(cx,y,s,segs,fill='#000'):
    x=cx-segw(segs,s)/2; out=[]
    for t,k in segs:
        lead=len(t)-len(t.lstrip(' ')); body=t.strip(' '); x+=lead*W(' ',k,s)
        if body: out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{s}" fill="{fill}" {ATTR[k]}>{body}</text>')
        x+=W(body,k,s)+(len(t)-lead-len(body))*W(' ',k,s)
    return ''.join(out)
S=20.0; BASE=HEAD=SMALL=S; PRINT_W=396.0; PRINT_PT=8.0
def box(lines, fill, stroke, sw=1.8, minw=0, padx=10, pady=12):
    # lines: list of (size, segs, color)
    wmax=max(segw(sg,s) for s,sg,c in lines); wbox=max(wmax+2*padx,minw)
    h=sum(1.28*s for s,sg,c in lines)+2*pady-0.28*lines[-1][0]
    return dict(lines=lines,w=wbox,h=h,fill=fill,stroke=stroke,sw=sw)
def draw_box(B,x,yc):
    y0=yc-B['h']/2; out=[f'<rect x="{x:.1f}" y="{y0:.1f}" width="{B["w"]:.1f}" height="{B["h"]:.1f}" rx="9" fill="{B["fill"]}" stroke="{B["stroke"]}" stroke-width="{B["sw"]}"/>']
    y=y0+12; cx=x+B['w']/2
    for s,sg,c in B['lines']:
        y+=s*0.95; out.append(text(cx,y,s,sg,c)); y+=0.33*s
    B['x']=x; B['y0']=y0; B['cx']=cx
    return ''.join(out)
K='#000'; RED=GRN=K; ORG='#000'; GREY='#f0f0f0'   # grey scale; only the inputs (stencil, depth) and the output in colour
x_='×'; ar='→'
def io(a,b_): return [(f'{a}','M')]
def bx(title,lines,fill,stroke,**kw):
    # lines: math strings, or (segs, colour) for text with mixed fonts / colours
    L=[(S,[(title,'Hb')],K)]
    for t in lines: L.append((S,[(t,'M')],K) if isinstance(t,str) else (S,t[0],t[1]))
    return box(L,fill,stroke,padx=8,**kw)
NOTZ=[([('not ','Hi'),(z,'M'),('-conditioned','Hi')],RED)]
BX={
 'inp': bx('Input',[([(f'({u}, {v})','M'),(' stencil','H')],K),f'{r} = 3',f'2 {x_} 7 {x_} 7'],'#eaf2fb','#4a7fb5'),
 'c1':  bx('conv1',NOTZ,GREY,'#000'),
 'fl':  bx('FiLM',[f'{x_} {g}({z}) + {b}({z})',([('only ','Hi'),(z,'M'),('-entry point','Hi')],GRN)],GREY,ORG),
 'c2':  bx('conv2',[([('no FiLM','Hi')],RED)],GREY,'#000'),
 'fc':  bx('FC head',[],GREY,'#000'),
 'wo':  bx('Output',[w],'#eaf6ea','#3a8a4a'),
 'zb':  bx('Depth',[z],'#eaf2fb','#4a7fb5'),   # input, same colour as Input
 'gen': bx('FiLM generator',[f'1 {ar} {g} [8], {b} [8]',([('init: ','H'),(f'{g} = 1, {b} = 0','M')],GRN)],GREY,ORG),
}
def arrow(x1,y1,x2,y2,col='#000'):
    m='arrow' if col=='#000' else 'arrowO'
    return f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{col}" stroke-width="1.8" fill="none" marker-end="url(#{m})"/>'
def poly(pts,col='#000'):
    m='arrow' if col=='#000' else 'arrowO'
    d=' '.join(('M' if n==0 else 'L')+f'{x:.1f},{y:.1f}' for n,(x,y) in enumerate(pts))
    return f'<path d="{d}" stroke="{col}" stroke-width="1.8" fill="none" stroke-linejoin="round" marker-end="url(#{m})"/>'
def put(x,y,t,k,c): return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{S}" fill="{c}" {ATTR[k]} xml:space="preserve">{t}</text>'
MARG=4; out=[]
row=['inp','c1','fl','c2','fc','wo']
# one row; the gap between boxes fills the AGU text width (S*PRINT_W/PRINT_PT in SVG units)
Wmax=S*PRINT_W/PRINT_PT
GAP=(Wmax-2*MARG-sum(BX[k]['w'] for k in row))/(len(row)-1)
print('GAP',GAP); assert GAP>=22, GAP
xs={}; x=MARG
for k in row: xs[k]=x; x+=BX[k]['w']+GAP
Wtot=x-GAP+MARG
# generator above FiLM, depth on its left
G=BX['gen']; gx=xs['fl']+BX['fl']['w']/2-G['w']/2; gyc=6+G['h']/2
out.append(draw_box(G,gx,gyc))
Z=BX['zb']; zx=gx-50-Z['w']; out.append(draw_box(Z,zx,gyc)); out.append(arrow(zx+Z['w'],gyc,gx-2,gyc,ORG))
hrow=max(BX[k]['h'] for k in row)
yc2=G['y0']+G['h']+55+hrow/2
for k in row: out.append(draw_box(BX[k],xs[k],yc2))
for a,b_ in zip(row[:-1],row[1:]): out.append(arrow(BX[a]['x']+BX[a]['w'],yc2,BX[b_]['x']-2,yc2))
F=BX['fl']; out.append(arrow(F['cx'],G['y0']+G['h'],F['cx'],F['y0']-2,ORG))
out.append(f'<text x="{F["cx"]+10:.1f}" y="{(G["y0"]+G["h"]+F["y0"])/2+7:.1f}" font-size="{S}" fill="{ORG}" {ATTR["M"]}>{g}, {b}</text>')
H=yc2+hrow/2+6
PRINT_W=Wtot*PRINT_PT/S
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{Wtot:.0f}" height="{H:.0f}">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker>
<marker id="arrowO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ORG}"/></marker></defs>
<rect width="{Wtot:.0f}" height="{H:.0f}" fill="#ffffff"/>
{"".join(out)}
</svg>'''
open(sys.argv[1],'w',encoding='utf-8').write(svg)
cairosvg.svg2pdf(url=sys.argv[1], write_to=sys.argv[2], output_width=PRINT_W*4/3, output_height=PRINT_W*4/3*H/Wtot)   # cairosvg sizes in px (1 px = 0.75 pt)
print('svg',round(Wtot),round(H),'print',round(PRINT_W,1),'x',round(PRINT_W*H/Wtot,1),'pt, text',S*PRINT_W/Wtot,'pt')
