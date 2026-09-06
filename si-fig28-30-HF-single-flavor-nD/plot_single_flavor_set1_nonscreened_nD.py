import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 


def main(pd):
  temp=np.load(pd['fname'],allow_pickle=True)
  
  
  data=temp['collect_data'] #[iU,iepsr,ine,occ or unocc]
  
  DOS=np.load(pd['DOSfname'],allow_pickle=True)['DOS_data']
  
  Us=np.array(temp['Us']).astype(float)
  epsrs=np.array(temp['epsrs']).astype(float)
  nelist=np.array(temp['nelist']).astype(float)
  
  print("epsilons:")
  print(epsrs)
  
  XX,YY=np.meshgrid(nelist,Us,indexing='ij')
  YY*=1e3
  XX/=1e16

  greylinewidth=0.000

  data[data[:,:,:,1]<1.6,1]=np.nan
  for iepsr,epsr in enumerate(epsrs):
   if iepsr in [pd['iepsr']]: 
    fig,axs=plt.subplots(1,3,figsize=(8,3))
    print("max num pockets : %d, %d" %(np.nanmax(data[:,iepsr,:,0].T),np.nanmax(data[:,iepsr,:,1].T)))

    cm=plt.get_cmap("plasma",4)

    ccc=axs[0].pcolormesh(XX,YY,data[:,iepsr,:,0].T,shading='nearest',cmap=cm,vmin=1-0.5,vmax=4+0.5,edgecolor='grey', linewidth=greylinewidth)
    plt.colorbar(ccc,ax=axs[0],ticks=np.arange(0,5))
    axs[0].pcolormesh(XX,YY,data[:,iepsr,:,1].T-1,shading='nearest',cmap='Greys',vmin=-100,vmax=100,zorder=1)
    for iax in range(3):
      axs[iax].set_xlabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
      axs[iax].set_ylabel(r"$u_D$ (meV)")
    axs[0].set_title(r"# electron pockets")

    
    ccc=axs[1].pcolormesh(XX,YY,data[:,iepsr,:,2].T,shading='nearest',cmap='jet',vmin=0,vmax=1,edgecolor='grey', linewidth=greylinewidth)
    plt.colorbar(ccc,ax=axs[1])
    axs[1].set_title(r"$C_{3}$-breaking")
    
    ccc=axs[2].pcolormesh(XX,YY,DOS[:,iepsr,:,0].T/1e18,shading='nearest',cmap='magma',vmin=0,edgecolor='grey', linewidth=greylinewidth)
    plt.colorbar(ccc,ax=axs[2])
    axs[2].set_title(r"DOS (eV$^{-1}\mathrm{nm}^{-2}$)")  
    
    
    plt.tight_layout()

pd={}

pd['fname']='../data/hartree-fock/single_flavor_HF/FS_collect_data_set1_nonscreened.npz'
pd['DOSfname']='../data/hartree-fock/single_flavor_HF/DOS_collect_set1_nonscreened.npz'
pd['iepsr']=2  #0,1,2 correspond to epsr=14,20,30,1e6 respectively

main(pd)