import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 


def main(pd):
  temp=np.load(pd['fname'],allow_pickle=True)
  
  
  data=temp['collect_data'] #[iU,iepsr,ine, len(occs) or len(unoccs) or C3break or elecfreqs or holefreqs or ne]
  
  Us=np.array(temp['Us']).astype(float)
  epsrs=np.array(temp['epsrs']).astype(float)
  nelist=np.array(temp['nelist']).astype(float)
  
  XX,YY=np.meshgrid(nelist,Us,indexing='ij')
  YY*=1e3
  XX/=1e16

  greylinewidth=0.000

  print("U list")
  print(Us)

  print("selected U")
  print(Us[pd['iU']])
  
  print("epsr list")
  print(epsrs)
  
  print("selected epsr")
  print(epsrs[pd['iepsr']])
  
  fig,axs=plt.subplots(1,1,figsize=(3,3))
  for ine,ne in enumerate(nelist):
   try: 

    axs.plot(data[pd['iU'],pd['iepsr'],ine,5]/1e16,data[pd['iU'],pd['iepsr'],ine,5]/1e16,
             marker='.',color='grey',linestyle='none',markersize=0.5
             )
    axs.plot(data[pd['iU'],pd['iepsr'],ine,5]/1e16,data[pd['iU'],pd['iepsr'],ine,5]/1e16/3,
             marker='.',color='grey',linestyle='none',markersize=0.5
             )
    
    if len(data[pd['iU'],pd['iepsr'],ine,4])>0:
      axs.plot(data[pd['iU'],pd['iepsr'],ine,5]/1e16*np.ones(len(data[pd['iU'],pd['iepsr'],ine,3])),
               np.array(data[pd['iU'],pd['iepsr'],ine,3])/1e16+np.array(data[pd['iU'],pd['iepsr'],ine,4])/1e16,
               marker='.',color='black',linestyle='none'
               )
      axs.plot(data[pd['iU'],pd['iepsr'],ine,5]/1e16*np.ones(len(data[pd['iU'],pd['iepsr'],ine,4])),
               np.array(data[pd['iU'],pd['iepsr'],ine,4])/1e16,
               marker='.',color='blue',linestyle='none'
               )
  
    else:
      axs.plot(data[pd['iU'],pd['iepsr'],ine,5]/1e16*np.ones(len(data[pd['iU'],pd['iepsr'],ine,3])),
               np.array(data[pd['iU'],pd['iepsr'],ine,3])/1e16,
               marker='.',color='black',linestyle='none'
               )

   except:
     None
  

  axs.set_ylabel(r"$n_\mathrm{SdH}$ ($10^{12}\,\mathrm{cm}^{-2}$)")
  axs.set_xlabel(r"$n$ ($10^{12}\,\mathrm{cm}^{-2}$)")
  if iepsr!=3:
    
    plt.suptitle(r"$u_D=%.1f\,\mathrm{meV};\,\,\epsilon=%.1f$" %(1e3*Us[pd['iU']],epsrs[pd['iepsr']]))
  else:
    
    plt.suptitle(r"$u_D=%.1f\,\mathrm{meV};\,\,$non-int." %(1e3*Us[pd['iU']],))
  plt.tight_layout()


pd={}

pd['fname']='../data/hartree-fock/single_flavor_HF/FS_collect_data_Onsager_set1_nonscreened.npz'

for iU in [8]: #uD takes 16 values uniformly spaced from 30meV to 60meV
  for iepsr in [2]: #epsr takes values 14,20,30,1e6
    
    pd['iU']=iU
    pd['iepsr']=iepsr

    main(pd)
