import numpy as np
import matplotlib.pyplot as plt

def main(pd):
  indata=np.load(pd['fname'],allow_pickle=True)
  klist=indata['klist']
  nelist=indata['nelist']
  data=indata['data']
  pd_data=indata['pd'][()]

  
  print("ne : "+str(nelist[pd['ine']]/1e16))

  nk=data[pd['ine'],1]

  glist=indata['glist']
  glist_temp=glist.tolist()
  gmax=np.max(np.abs(glist))
  
  bangle=pd_data['bangle']
  bmod=pd_data['bmod']
  b1=bmod*np.array([np.cos(bangle),np.sin(bangle)])
  b2=bmod*np.array([np.cos(bangle+2*np.pi/3),np.sin(bangle+2*np.pi/3)])
  
  a1=np.array([b2[1],-b2[0]])/np.dot(b1,np.array([b2[1],-b2[0]]))
  a2=np.array([-b1[1],+b1[0]])/np.dot(b2,np.array([-b1[1],+b1[0]]))
  
  
  XX=np.zeros((2*gmax+1,2*gmax+1))
  YY=np.zeros((2*gmax+1,2*gmax+1))
  occ=np.zeros((2*gmax+1,2*gmax+1))
  

  
  for iX in range(2*gmax+1):
    for iY in range(2*gmax+1):
      XX[iX,iY]=((iX-gmax)*b1+(iY-gmax)*b2)[0]
      YY[iX,iY]=((iX-gmax)*b1+(iY-gmax)*b2)[1]
      # if [iX-gmax,iY-gmax] in glist_temp:
        # occ[iX,iY]=glist_temp.index([iX-gmax,iY-gmax])

  for k in klist[nk]:
    iX=int(np.round(np.dot(k,a1)))
    iY=int(np.round(np.dot(k,a2)))
    occ[iX+gmax,iY+gmax]+=1
  
  fig,axs=plt.subplots(1,1,figsize=(4.5,4.5))
  axs=np.array([[axs]])
  axs[0,0].contour(XX,YY,occ,[0.5],colors='black')
  axs[0,0].set_aspect('equal')
  axs[0,0].axis('off')


pd={}

epsr="30"
pd['epsr']=epsr
Dlist="0.042 0.046 0.048 0.052".split()
for D in Dlist:
  pd['fname']='../data/hartree-fock/single_flavor_HF/FS_set1_nonscreened/U%s_epsr%s.npz' %(D,epsr)

  for ine in [50]: #there are 130 values of ne, equally spaced from 0.1 to 1.4 in units of 10^12 cm^{-2}
    pd['ine']=ine
    main(pd)
