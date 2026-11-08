"""Document-space geometry and uncalibrated measurements, no learned claims."""
import cv2
import numpy as np
from scipy.fft import dctn
from scipy.signal import find_peaks


def exif_matrix(o,w,h):
    # Homogeneous continuous pixel-edge coordinates, centers at (x+.5,y+.5).
    matrices={1:[[1,0,0],[0,1,0],[0,0,1]],2:[[-1,0,w],[0,1,0],[0,0,1]],
              3:[[-1,0,w],[0,-1,h],[0,0,1]],4:[[1,0,0],[0,-1,h],[0,0,1]],
              5:[[0,1,0],[1,0,0],[0,0,1]],6:[[0,-1,h],[1,0,0],[0,0,1]],
              7:[[0,-1,h],[-1,0,w],[0,0,1]],8:[[0,1,0],[-1,0,w],[0,0,1]]}
    return np.array(matrices.get(o,matrices[1]),dtype=float)


def transform_points(points,H):
    p=np.asarray(points,dtype=float);H=np.asarray(H,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or H.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(H).all():
        raise ValueError('Finite Nx2 points and 3x3 transform required')
    mapped=np.c_[p,np.ones(len(p))]@H.T
    if np.any(abs(mapped[:,2])<1e-10):raise ValueError('Point mapped to infinity')
    return mapped[:,:2]/mapped[:,2,None]


def perceptual_hash(rgb):
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    low=dctn(cv2.resize(grey,(32,32)).astype(float),norm='ortho')[:8,:8]
    bits=(low>np.median(low.ravel()[1:])).ravel();bits[0]=False
    return f'{int("".join("1" if x else "0" for x in bits),2):016x}'


def quality_report(rgb):
    small=rgb
    if max(rgb.shape[:2])>1600:
        scale=1600/max(rgb.shape[:2]);small=cv2.resize(rgb,None,fx=scale,fy=scale)
    grey=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY).astype(float)/255
    lap=cv2.Laplacian(grey,cv2.CV_64F)
    background=cv2.GaussianBlur(grey,(0,0),max(3,min(grey.shape)/20))
    local=grey-background
    return {'overall_score':None,'calibration':'not calibrated on real author photographs',
            'measurement_scale':list(small.shape[:2]),
            'sharpness_laplacian_variance':float(lap.var()),
            'local_contrast_rms':float(np.sqrt(np.mean(local**2))),
            'highlight_fraction':float((grey>.98).mean()),'dark_fraction':float((grey<.08).mean()),
            'background_nonuniformity':float(background.std()),
            'text_sharpness':None,'occlusion_probability':None,'motion_blur_probability':None,
            'perspective_complexity':None,
            'recommended_strategy':'Inspect ink mask and page geometry; measurements are not quality verdicts.'}


def detect_grid(rgb):
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    def axis(signal):
        signal=signal.mean()-signal
        if signal.std()<.01:return None,None,None
        # Remove slow illumination trend, then find ruling candidates in projection.
        baseline=cv2.GaussianBlur(signal.reshape(1,-1),(0,0),8).ravel()
        signal=signal-baseline
        peaks,_=find_peaks(signal,prominence=max(1,signal.std()*.5),distance=4)
        if len(peaks)<4:return None,None,None
        gaps=np.diff(peaks);spacing=float(np.median(gaps))
        consistency=float(np.mean(abs(gaps-spacing)<=max(1,spacing*.12)))
        if consistency<.6:return None,None,consistency
        return spacing,float(peaks[0]%spacing),consistency
    sx,px,cx=axis(grey.mean(axis=0));sy,py,cy=axis(grey.mean(axis=1))
    return {'grid_spacing_x':sx,'grid_spacing_y':sy,'grid_phase_x':px,'grid_phase_y':py,
            'grid_angle':None,'grid_confidence':None,'projection_consistency_x':cx,
            'projection_consistency_y':cy,'local_grid_distortion':None,
            'method':'axis projection peaks; only approximately axis-aligned uniform grids',
            'warning':'No calibrated confidence or local deformation model.'}


def page_candidates(rgb):
    scale=min(1,1200/max(rgb.shape[:2]));small=cv2.resize(rgb,None,fx=scale,fy=scale)
    grey=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(grey,(5,5),0),40,120)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    choices=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:20]:
        p=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(p)==4 and cv2.isContourConvex(p) and cv2.contourArea(p)>.15*grey.size:
            points=p[:,0,:].astype(float)/scale
            # clockwise from top-left-ish vertex, explicit manual review required.
            center=points.mean(axis=0);order=np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
            points=points[order];points=np.roll(points,-np.argmin(points.sum(axis=1)),axis=0)
            choices.append({'corners':points.tolist(),'review_status':'proposed','confidence':None})
    return choices
