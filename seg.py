import numpy as np
from PIL import Image
im=np.array(Image.open('../images/1.png').convert('RGB')).astype(float)
R=im[...,0]
H,W=R.shape
brown=np.clip((255-R)/197,0,1)
blue=np.clip((255-R)/95,0,1)
yy,xx=np.mgrid[0:H,0:W]
sym=xx<430
text=(~sym)&(yy<485)
tag=(~sym)&(yy>=485)
layers={}
layers['sua']=brown*(text&(yy<322))
layers['imobi']=brown*(text&(yy>=322))
layers['tag']=blue*tag
# symbol components
m=(blue>0.05)&sym
lab=np.zeros((H,W),int);n=0
for y,x in zip(*np.nonzero(m)):
    if lab[y,x]: continue
    n+=1;st=[(y,x)];lab[y,x]=n
    while st:
        a,b=st.pop()
        for da in(-1,0,1):
            for db in(-1,0,1):
                c,d=a+da,b+db
                if 0<=c<H and 0<=d<W and m[c,d] and not lab[c,d]:
                    lab[c,d]=n;st.append((c,d))
print(n,[(i,(lab==i).sum(), np.nonzero(lab==i)[0].mean()) for i in range(1,n+1)])
np.save('lab.npy',lab)
