import numpy as np, subprocess, math
from PIL import Image
exec(open('seg.py').read().split('# symbol components')[0])
lab=np.load('lab.npy')
def dil(mask,r=3):
    o=np.zeros_like(mask)
    for a in range(-r,r+1):
        for b in range(-r,r+1):
            o|=np.roll(np.roll(mask,a,0),b,1)
    return o
# assign each symbol pixel to nearest-label via dilation priority
sym_alpha=blue*sym
own=np.zeros((H,W),int)
for i in (1,2,3):
    d=dil(lab==i)&(own==0)&(sym_alpha>0)
    own[d]=i
# outline (2) takes priority where it overlaps bands' fringe
layers['top']=sym_alpha*(own==1)
layers['outline']=sym_alpha*(own==2)
layers['bot']=sym_alpha*(own==3)
BROWN=np.array([58,43,36],float);BLUE=np.array([160,179,209],float)
col={'sua':BROWN,'imobi':BROWN,'tag':BLUE,'top':BLUE,'outline':BLUE,'bot':BLUE}
tot=sum(layers.values()); ys,xs=np.nonzero(tot>0.05)
cx=(xs.min()+xs.max()+1)/2; cy=(ys.min()+ys.max()+1)/2
print('bbox',xs.min(),xs.max(),ys.min(),ys.max(),cx,cy)
OW,OH=1920,1080; S=1.6
ox=OW/2-cx*S; oy=OH/2-cy*S
def big(a):
    im=Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((round(W*S),round(H*S)),Image.LANCZOS)
    c=Image.new('L',(OW,OH),0); c.paste(im,(round(ox),round(oy)))
    return np.asarray(c).astype(np.float32)/255
A={k:big(v) for k,v in layers.items()}
# outline sweep parameter (in big coords)
o=layers['outline']; oyy,oxx=np.nonzero(o>0.05)
ccx,ccy=oxx.mean(),oyy.mean()
left=oxx.argmin(); a0=math.atan2(oyy[left]-ccy,oxx[left]-ccx)
Y,X=np.mgrid[0:OH,0:OW]
ang=np.arctan2((Y-oy)/S-ccy,(X-ox)/S-ccx)
U=((ang-a0)%(2*np.pi))/(2*np.pi)   # increasing clockwise on screen (y down)
UY=(Y.astype(np.float32))
def ease(t): t=min(max(t,0),1); return t*t*t*(t*(6*t-15)+10) if False else (4*t**3 if t<.5 else 1-(-2*t+2)**3/2)
def sine(t): t=min(max(t,0),1); return -(math.cos(math.pi*t)-1)/2
def seg(t,a,b): return (t-a)/(b-a)
def shift(a,dx,dy):
    if abs(dx)<1e-3 and abs(dy)<1e-3: return a
    im=Image.fromarray(a,mode='F')
    return np.asarray(im.transform(im.size,Image.AFFINE,(1,0,-dx,0,1,-dy),resample=Image.BICUBIC))
ang_s=math.radians(32.5); dvx,dvy=math.cos(ang_s),math.sin(ang_s)
D=900*1.0
def frame(t):
    out=np.ones((OH,OW,3),np.float32)*255
    def over(a,c):
        nonlocal out
        out=out*(1-a[...,None])+c*a[...,None]
    # bands: top enters from the left, bottom from the right, along slant
    e=ease(seg(t,0.35,1.45)); 
    if e>0:
        d=(1-e)*D; a=shift(A['top'],-d*dvx,-d*dvy)*min(1,e*3)
        over(a,col['top'])
    e=ease(seg(t,0.55,1.65))
    if e>0:
        d=(1-e)*D; a=shift(A['bot'],d*dvx,d*dvy)*min(1,e*3)
        over(a,col['bot'])
    p=sine(seg(t,1.55,2.5))
    if p>0:
        f=0.05; m=np.clip((p*(1+f)-U)/f,0,1).astype(np.float32)
        over(A['outline']*m,col['outline'])
    def wipe(name,a,b,y0,y1):
        p=sine(seg(t,a,b))
        if p<=0: return
        f=40
        edge=y1+f-(y1-y0+f)*p
        m=np.clip((Y-edge)/f,0,1).astype(np.float32)
        over(A[name]*m,col[name])
    wipe('sua',2.45,3.25,150*S+oy,330*S+oy)
    wipe('imobi',2.75,3.55,320*S+oy,485*S+oy)
    p=sine(seg(t,3.55,4.2))
    if p>0: over(A['tag']*p,col['tag'])
    return out
FPS=60;N=360
pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-',
 '-c:v','libx264','-crf','12','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart','sua_imobi_logo_reveal.mp4'],stdin=subprocess.PIPE)
# final frame: original pixels composed at layer sums (identical to layered full reveal)
for i in range(N):
    f=frame(i/FPS)
    pr.stdin.write(np.clip(f+0.5,0,255).astype(np.uint8).tobytes())
    if i in(0,45,75,105,135,165,195,225,255,359): Image.fromarray(np.clip(f,0,255).astype(np.uint8)).save(f'f{i}.png')
pr.stdin.close();pr.wait()
