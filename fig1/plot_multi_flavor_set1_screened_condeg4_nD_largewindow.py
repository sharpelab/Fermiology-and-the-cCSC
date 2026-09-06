import numpy as np
import matplotlib.pyplot as plt 


def main(pd):
  temp=np.load(pd['fname'],allow_pickle=True)
  
  
  data=temp['collect_data'][:pd['iUmax']] #[iU,iepsr,ine,occ or unocc]
  

  Us=np.array(temp['Us']).astype(float)[:pd['iUmax']]
  epsrs=np.array(temp['epsrs']).astype(float)
  nelist=np.array(temp['nelist']).astype(float)
  numne=len(nelist)
  nU=len(Us)
  
  print("epsilons:")
  print(epsrs)
  
  XX,YY=np.meshgrid(nelist,Us,indexing='ij')
  YY*=1e-9
  XX/=1e16
  
  DOS=np.load(pd['DOSfname'])['DOS_data']

  for iepsr,epsr in enumerate(epsrs):
    if iepsr in [pd['iepsr']]: 
      
      DOSproc=np.zeros((numne,nU))
      fracproc=np.zeros((numne,nU))
      
      for iU in range(nU):
        for ine in range(numne):
          #only not-FP
          if np.isnan(data[iU,iepsr,ine,0]) and not np.isnan(data[iU,iepsr,ine,1]):
            fracproc[ine,iU]=data[iU,iepsr,ine,2]/np.sum(data[iU,iepsr,ine,2:6])
            DOSproc[ine,iU]=DOS[iU,iepsr,ine,1]
          #only FP
          elif not np.isnan(data[iU,iepsr,ine,0]) and np.isnan(data[iU,iepsr,ine,1]):
            fracproc[ine,iU]=1
            DOSproc[ine,iU]=DOS[iU,iepsr,ine,0]
          #FP lower energy
          elif data[iU,iepsr,ine,1]>data[iU,iepsr,ine,0]:
            fracproc[ine,iU]=1
            DOSproc[ine,iU]=DOS[iU,iepsr,ine,0]  
            #FP lower energy
          elif data[iU,iepsr,ine,1]<data[iU,iepsr,ine,0]:
            fracproc[ine,iU]=data[iU,iepsr,ine,2]/np.sum(data[iU,iepsr,ine,2:6])
            DOSproc[ine,iU]=DOS[iU,iepsr,ine,1]

            


            
      """
      PLOTTING STARTS HERE
      """
      fig,axs=plt.subplots(1,2,figsize=(5,3))
      
      #plot nmax/n
      ccc=axs[0].pcolormesh(XX,YY,fracproc,cmap='coolwarm',shading='nearest',vmin=0,vmax=1)
      plt.colorbar(ccc,ax=axs[0])
      axs[0].set_title(r"$n_\mathrm{max}/n$")
      
      #plot DOS
      ccc=axs[1].pcolormesh(XX,YY,DOSproc/1e18,cmap='magma',shading='nearest',vmin=0,
                            )
      plt.colorbar(ccc,ax=axs[1])
      axs[1].set_title(r"DOS (eV$^{-1}\mathrm{nm}^{-2}$)")  
      
      #axis labels
      for iax in range(2):
        axs[iax].set_xlabel(r"$n_e$ ($10^{12}$ cm$^{-2}$)")
        axs[iax].set_ylabel(r"$D$ (V/nm)")
      
      
      plt.tight_layout()
      



pd={}

pd['fname']='../data/fig1/collect_data.npz'
pd['DOSfname']='../data/fig1/DOS_data.npz'
pd['iepsr']=2 #0,1,2 correspond to epsr=8,14,20,30 respectively
pd['iUmax']=1000 #set upper limit of U
main(pd)