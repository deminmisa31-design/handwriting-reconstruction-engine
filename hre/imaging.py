"""Pure image processing. All arrays use oriented derivative coordinates."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def extract_ink(rgb):
    """Approximate alpha using a smooth closing-estimated paper background.

    Not physical matting. Retains dark/blue strokes; weak straight ruling is
    suppressed only when its contrast is below strong ink. No glyph labels.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB image')
    size = max(9, min(61, (min(rgb.shape[:2])//12)|1))
    background = cv2.morphologyEx(rgb, cv2.MORPH_CLOSE, np.ones((size,size),np.uint8))
    background = cv2.GaussianBlur(background,(0,0),max(2,size/4)).astype(float)/255
    image = rgb.astype(float)/255
    contrast = np.clip(background-image,0,1)
    strength = np.max(contrast,axis=2)
    # Remove low-contrast scanner noise while retaining anti-aliased edges.
    alpha = np.clip((strength-.025)/.75,0,1)
    weak = ((alpha>.045)&(alpha<.32)).astype(np.uint8)
    h,w = weak.shape
    horizontal = cv2.morphologyEx(weak,cv2.MORPH_OPEN,np.ones((1,max(30,w//5)),np.uint8))
    vertical = cv2.morphologyEx(weak,cv2.MORPH_OPEN,np.ones((max(30,h//5),1),np.uint8))
    ruling = (horizontal|vertical)>0
    alpha[ruling] = 0
    strong = alpha>.35
    color = np.median(image[strong],axis=0) if strong.any() else np.array([.12,.2,.55])
    # Re-estimate alpha under a constant measured ink-color mixing model.
    denom = background-color
    fit = np.sum((background-image)*denom,axis=2)/np.maximum(np.sum(denom**2,axis=2),.02)
    alpha = np.clip(fit,0,1) * (strength>.025)
    alpha[ruling] = 0
    return alpha.astype(np.float32), {
        'method':'local-closing-constant-ink-alpha-v1',
        'ink_color_srgb':color.tolist(), 'background_kernel':size,
        'ruling_pixels_suppressed':int(ruling.sum()),
        'warning':'Approximate alpha; inspect grid intersections and pale strokes manually.'}


def _runs(values):
    edges = np.diff(np.r_[False,np.asarray(values,dtype=bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))


def extract_ink_grid_aware(rgb):
    """Experimental blue-ink/neutral-grid candidate; never replaces saved masks.

    Robust profiles along periodic near-axis lines estimate printed grid density.
    Blue spectral evidence protects strokes crossing those lines. Black ink,
    colored ruling and strong page curvature remain ambiguous; inspect manually.
    """
    rgb=np.asarray(rgb,dtype=np.uint8)
    alpha,base_info=extract_ink(rgb)
    h,w=alpha.shape
    info={'method':'experimental-periodic-grid-profile-v2','experimental':True,
          'grid_line_count':0,'decision':'candidate',
          'warning':'Diagnostic candidate only; not calibrated. Black/blue grids and ink may be ambiguous.'}
    mask=np.zeros((h,w),np.uint8)
    if h*w>4_000_000:
        info['decision']='fallback_diagnostic_size_limit'
        return alpha,info,mask
    size=base_info['background_kernel']
    background=cv2.morphologyEx(rgb,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
    background=cv2.GaussianBlur(background,(0,0),max(2,size/4))
    density=np.maximum(0,np.log((background.astype(float)+1)/(rgb.astype(float)+1)))
    blue=(density[:,:,0]+density[:,:,1])/2-density[:,:,2]
    if np.count_nonzero(blue>.025)<max(8,int(h*w*.0002)):
        info['decision']='fallback_no_blue_evidence'
        return alpha,info,mask
    neutral=np.min(density,axis=2)
    binary=np.uint8(neutral>.018)*255
    lines=cv2.HoughLinesP(binary,1,np.pi/720,threshold=max(15,min(h,w)//8),
                          minLineLength=max(25,int(min(h,w)*.42)),maxLineGap=max(3,min(h,w)//35))
    families={'horizontal':[],'vertical':[]}
    if lines is not None:
        for x0,y0,x1,y1 in lines[:,0,:]:
            dx,dy=abs(x1-x0),abs(y1-y0)
            horizontal=dx>=w*.42 and dy<=max(2,dx*.14)
            vertical=dy>=h*.42 and dx<=max(2,dy*.14)
            if horizontal:
                slope=(y1-y0)/(x1-x0)
                families['horizontal'].append((y0+slope*(w/2-x0),slope))
            elif vertical:
                slope=(x1-x0)/(y1-y0)
                families['vertical'].append((x0+slope*(h/2-y0),slope))
    field=np.zeros_like(density)
    for orientation,segments in families.items():
        groups=[]
        for center,slope in sorted(segments):
            if groups and abs(center-np.median([p[0] for p in groups[-1]]))<=3:
                groups[-1].append((center,slope))
            else:groups.append([(center,slope)])
        centers=np.array([np.median([p[0] for p in g]) for g in groups])
        if len(centers)<4:continue
        spacing=float(np.median(np.diff(centers)))
        if spacing<6:continue
        distance=abs((centers[:,None]-centers[None,:]+spacing/2)%spacing-spacing/2)
        support=(distance<=max(2,spacing*.12)).sum(axis=1)
        keep=support>=max(4,int(support.max()*.7))
        for group,center,accept in zip(groups,centers,keep):
            if not accept:continue
            slope=float(np.median([p[1] for p in group]))
            axis=np.arange(w if orientation=='horizontal' else h)
            cross=center+slope*(axis-(w if orientation=='horizontal' else h)/2)
            for offset in range(-3,4):
                other=np.rint(cross+offset).astype(int)
                if orientation=='horizontal':xs,ys=axis,other
                else:xs,ys=other,axis
                inside=(xs>=0)&(xs<w)&(ys>=0)&(ys<h);xs,ys=xs[inside],ys[inside]
                if len(xs)<8:continue
                profile=np.quantile(density[ys,xs],.35,axis=0)
                if profile.max()<.012:continue
                field[ys,xs]=np.maximum(field[ys,xs],profile)
            info['grid_line_count']+=1
    mask[np.max(field,axis=2)>.012]=255
    residual=np.maximum(0,density-field)
    corrected=background.astype(float)/255*(1-np.exp(-residual))
    denom=background.astype(float)/255-np.asarray(base_info['ink_color_srgb'])
    fitted=np.clip(np.sum(corrected*denom,axis=2)/np.maximum(np.sum(denom**2,axis=2),.02),0,1)
    protection=np.clip((blue-.01)/.04,0,1)
    candidate=alpha.copy()
    candidate[mask>0]=np.maximum(fitted,alpha*protection)[mask>0]
    info['grid_candidate_pixels']=int((mask>0).sum())
    info['suppressed_alpha_mass']=float(np.sum(np.maximum(0,alpha-candidate)))
    if not info['grid_line_count']:info['decision']='fallback_no_periodic_grid'
    return candidate.astype(np.float32),info,mask


def segment(alpha):
    binary = (alpha>.16).astype(np.uint8)
    h,w=binary.shape
    rows = binary.sum(axis=1) >= max(2,w*.002)
    rows = ndi.binary_closing(rows,structure=np.ones(7))
    lines=[]; words=[]
    for y0,y1 in _runs(rows):
        if y1-y0<3: continue
        ys,xs=np.where(binary[y0:y1])
        if not xs.size: continue
        box=[max(0,int(xs.min())-2),max(0,y0-2),min(w,int(xs.max())+3),min(h,y1+2)]
        lines.append(box)
        band=binary[y0:y1]
        join=max(3,min(25,int((y1-y0)*.28)))
        joined=cv2.dilate(band,np.ones((1,join),np.uint8))
        for x0,x1 in _runs(joined.any(axis=0)):
            py,px=np.where(band[:,x0:x1])
            if px.size<4: continue
            words.append([max(0,x0+int(px.min())-2),max(0,y0+int(py.min())-2),
                          min(w,x0+int(px.max())+3),min(h,y0+int(py.max())+3)])
    return lines, words


def metrics(alpha):
    binary=alpha>.2
    labels,n=ndi.label(binary)
    skeleton=skeletonize(binary)
    distance=ndi.distance_transform_edt(binary)
    widths=2*distance[skeleton]
    ys,xs=np.where(binary)
    return {'components':int(n),'ink_pixels':int(binary.sum()),
            'ink_fraction':float(binary.mean()),'skeleton_pixels':int(skeleton.sum()),
            'stroke_width_median':float(np.median(widths)) if widths.size else 0.,
            'height':int(np.ptp(ys)+1) if ys.size else 0,
            'width':int(np.ptp(xs)+1) if xs.size else 0,
            'baseline_hint':float(np.quantile(ys,.95)) if ys.size else 0.}


def ink_layers(rgb,include_experimental=False):
    alpha,info=extract_ink(rgb)
    size=info['background_kernel']
    background=cv2.morphologyEx(rgb,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
    background=cv2.GaussianBlur(background,(0,0),max(2,size/4))
    strength=np.max(np.clip(background.astype(float)-rgb.astype(float),0,255),axis=2)/255
    weak=((strength>.025)&(strength<.25)).astype(np.uint8)
    h,w=weak.shape
    ruling=cv2.morphologyEx(weak,cv2.MORPH_OPEN,np.ones((1,max(30,w//5)),np.uint8))|\
           cv2.morphologyEx(weak,cv2.MORPH_OPEN,np.ones((max(30,h//5),1),np.uint8))
    near=cv2.dilate(ruling,np.ones((3,3),np.uint8))>0
    layers={'background':background,'ink_score':np.uint8(alpha*255),
            'high_ink':np.uint8(alpha>.7)*255,
            'uncertain':np.uint8((alpha>.03)&(alpha<=.7))*255,
            'grid_intersections':np.uint8(near&(alpha>.1))*255}
    if include_experimental:
        grid_alpha,grid_info,grid_candidates=extract_ink_grid_aware(rgb)
        layers.update(grid_aware_alpha=np.uint8(grid_alpha*255),grid_candidates=grid_candidates)
        info={**info,'experimental_grid':grid_info}
    return layers,info


def valid_polygon(points, width, height):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or len(p)>1000 or not np.isfinite(p).all():
        raise ValueError('Polygon needs 3–1000 finite points')
    if (p<0).any() or (p[:,0]>width).any() or (p[:,1]>height).any():
        raise ValueError('Polygon outside page')
    if abs(cv2.contourArea(p.astype(np.float32)))<1:
        raise ValueError('Degenerate polygon')
    def orient(a,b,c):
        u,v=b-a,c-a
        return float(u[0]*v[1]-u[1]*v[0])
    def intersects(a,b,c,d):
        return orient(a,b,c)*orient(a,b,d)<=0 and orient(c,d,a)*orient(c,d,b)<=0 and \
            max(min(a[0],b[0]),min(c[0],d[0]))<=min(max(a[0],b[0]),max(c[0],d[0])) and \
            max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1]))
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1): continue
            if intersects(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]):
                raise ValueError('Self-intersecting polygon')
    return p


def perspective(rgb, corners, output_size):
    w,h=map(int,output_size)
    if min(w,h)<2 or w*h>40_000_000: raise ValueError('Invalid output size')
    points=valid_polygon(corners,rgb.shape[1],rgb.shape[0]).astype(np.float32)
    if len(points)!=4 or not cv2.isContourConvex(points):
        raise ValueError('Four convex corners ordered TL,TR,BR,BL required')
    dest=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    H=cv2.getPerspectiveTransform(points,dest)
    return cv2.warpPerspective(rgb,H,(w,h),borderMode=cv2.BORDER_REPLICATE),H


def mesh_warp(rgb, map_x, map_y):
    if map_x.shape!=map_y.shape or not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise ValueError('Finite, equally shaped inverse maps required')
    return cv2.remap(rgb,map_x.astype(np.float32),map_y.astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def to_linear(v):
    v=np.asarray(v,dtype=float)
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)


def to_srgb(v):
    return np.where(v<=.0031308,12.92*v,1.055*np.maximum(v,0)**(1/2.4)-.055)


def composite_linear(paper, alpha, ink_color):
    a=np.clip(alpha,0,1)[...,None]
    return np.clip(to_srgb(to_linear(paper)*(1-a)+to_linear(ink_color)*a),0,1)
