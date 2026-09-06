import numpy as np
import matplotlib.pyplot as plt 


def main(pd):
  temp=np.load(pd['fname'],allow_pickle=True)
  
  
  data=temp['collect_data'][:,:,pd['inemin']:] #[iU,iepsr,ine,idata]
  
  """
  idata entries:
  0 EIVC
  1 popsIVC
  2 IVCIVC
  3 Epol
  4 popspol
  5 IVCpol
  6 Enonpol
  7 popsnonpol
  8 IVCnonpol
  """

  Us=np.array(temp['Us']).astype(float)*1e3
  epsrs=np.array(temp['epsrs']).astype(float)
  nelist=(np.array(temp['nelist']).astype(float)/1e16)[pd['inemin']:]
  numne=len(nelist)
  nU=len(Us)
  nepsr=len(epsrs)


    

  for iU in range(nU):
    for iepsr in range(nepsr):
      for ine in range(numne):
        for idata in [0,3,6]:
          if data[iU,iepsr,ine,idata]>1e6:
            data[iU,iepsr,ine,idata]=np.nan
        
  print("Us")
  print(Us)
  print("epsrs")
  print(epsrs)


  

  
  for iU,U in enumerate(Us):
   if iU in [2,3,4,5,6]: 
    print(U)
    if np.sum(np.isnan(data[iU,pd['plotepsr'],:,0].astype(float)))>0:
      print("invalid IVC")

    fig,axs=plt.subplots(1,3,figsize=(7,2.5))
    
    # for irow in range(2):
    for icol in range(3):
        axs[icol].set_xlim(left=np.min(nelist),right=np.max(nelist))
    
    
    plt.suptitle(r"$u_D=%.1f\,$meV;  $\epsilon=%.1f$" %(U,epsrs[pd['plotepsr']]))
    axs[0].plot(nelist,
                1e3*(data[iU,pd['plotepsr'],:,3]-data[iU,pd['plotepsr'],:,0]),
                marker='.',linestyle='none',color='blue',label='fully pol.')
    axs[0].plot(nelist,
                1e3*(data[iU,pd['plotepsr'],:,6]-data[iU,pd['plotepsr'],:,0]),
                marker='x',linestyle='none',color='red',label='not fully pol.')
    axs[0].legend()
    axs[0].set_ylabel(r"$(E-E_\mathrm{IVC})/N_e$ (meV)")
    axs[0].axhline(y=0,color='gray',linestyle='dashed')
    axs[0].set_ylim(top=0.15,bottom=-0.05)
    axs[2].plot(nelist,data[iU,pd['plotepsr'],:,2],linestyle='none',
                  color='green',marker='+')
    axs[2].set_ylabel(r"$O_\mathrm{IVC}/N_e$")
    axs[2].set_ylim(bottom=-0.025,top=0.525)
    axs[1].set_ylim(bottom=-0.05,top=1.05)
    axs[0].set_xlabel(r"$n_e$ ($10^{12}\,$)cm$^{-2}$")
    axs[1].set_xlabel(r"$n_e$ ($10^{12}\,$)cm$^{-2}$")
    axs[2].set_xlabel(r"$n_e$ ($10^{12}\,$)cm$^{-2}$")
    axs[1].set_ylabel(r"$n_i/n_e$")
    
    
    axs[1].plot([],
                [],
                marker='x',color='red',linestyle='none',label='not fully pol.')
    axs[1].plot([],
                [],
                marker='+',color='green',linestyle='none',label='IVC')
    for ine in range(numne):
      
      try:
        for iflav in range(4):
          

            

            axs[1].plot(nelist[ine],
                        np.array(data[iU,pd['plotepsr'],ine,7])[iflav]/
                        np.sum(np.array(data[iU,pd['plotepsr'],ine,7])),
                        marker='x',color='red',linestyle='none')
      except:
        None
      
      try:
        for iflav in range(4):

            axs[1].plot(nelist[ine],
                        np.array(data[iU,pd['plotepsr'],ine,1])[iflav]/
                        np.sum(np.array(data[iU,pd['plotepsr'],ine,1])),
                        marker='+',color='green',linestyle='none')
      except:
        None
    axs[1].legend()
    
  
    plt.tight_layout()
    plt.savefig("epsr%d_U%.2f.png"%(epsrs[pd['plotepsr']],U),dpi=180)
  

pd={}
pd['inemin']=5 #set lower cutoff on density
pd['fname']='../data/hartree-fock/multi_flavor_HF/collect_data_IVC_set1_nonscreened.npz'
pd['plotepsr']=1 #epsr takes values 14,20,30

main(pd)