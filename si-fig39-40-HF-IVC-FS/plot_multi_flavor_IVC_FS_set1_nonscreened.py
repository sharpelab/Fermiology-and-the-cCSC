import numpy as np
import matplotlib.pyplot as plt

def main(pd_plot):
  indata=np.load(pd_plot['fname'],allow_pickle=True)
  nelist=indata['nelist']
  print("ne")
  print(nelist[pd_plot['ine']]/1e16)
  data=indata['data']
  
  """
  data[ine,0]=EIVC
  data[ine,1]=occkIVC
  data[ine,2]=FockIVC
  data[ine,3]=pdIVC
  data[ine,4]=PIVC
  data[ine,5]=popsIVC
  data[ine,6]=IVCIVC

  data[ine,7]=Epol
  data[ine,8]=occkpol
  data[ine,9]=Fockpol
  data[ine,10]=pdpol
  data[ine,11]=Ppol
  data[ine,12]=popspol
  data[ine,13]=IVCpol

  data[ine,14]=Enonpol
  data[ine,15]=occknonpol
  data[ine,16]=Focknonpol
  data[ine,17]=pdnonpol
  data[ine,18]=Pnonpol
  data[ine,19]=popsnonpol
  data[ine,20]=IVCnonpol
  """

  P=data[pd_plot['ine'],4]
  klist=indata['klist']
  occk=data[pd_plot['ine'],1]
  fig,axs=plt.subplots(1,1,figsize=(3,3))
  axs=np.array([[axs]])
  axs[0,0].set_aspect('equal')
  axs[0,0].axis('off')
  for  irow in range(1):
    for icol in range(1):
      axs[irow,icol].set_aspect('equal')
      axs[irow,icol].plot(klist[:,0],klist[:,1],color='k',
                          # marker='x',
                          markersize=1,linestyle='None')

  
  for irow in range(1):
    for icol in range(1):
      ccc=axs[irow,icol].scatter(klist[occk[icol],0],klist[occk[icol],1],
                             c=np.real(P[icol][:,0,0]-P[icol][:,1,1]),
                          s=1,cmap='coolwarm',vmin=-1,vmax=1)

      
  plt.suptitle(r"$n_e=%.3f\times 10^{12}\,$cm$^{-2}$" %(nelist[pd_plot['ine']]/1e16))
  plt.tight_layout()

  P=data[pd_plot['ine'],11]
  klist=indata['klist']
  occk=data[pd_plot['ine'],8]
  
  fig,axs=plt.subplots(1,1,figsize=(3,3))
  axs=np.array([[axs]])
  axs[0,0].set_aspect('equal')
  axs[0,0].axis('off')
  for  irow in range(1):
    for icol in range(1):
      axs[irow,icol].set_aspect('equal')
      axs[irow,icol].plot(klist[:,0],klist[:,1],color='k',
                          # marker='x',
                          markersize=1,linestyle='None')

  
  for irow in range(1):
    for icol in range(1):
      if (len(klist[occk[0],0])+len(klist[occk[0],1]))>0:
        s=0
      else:
        s=1

      ccc=axs[irow,icol].scatter(klist[occk[s],0],klist[occk[s],1],
                             c=np.real(P[s][:,0,0]-P[s][:,1,1]),
                          s=1,cmap='coolwarm',vmin=-1,vmax=1)

      
  plt.suptitle(r"$n_e=%.3f\times 10^{12}\,$cm$^{-2}$" %(nelist[pd_plot['ine']]/1e16))
  plt.tight_layout()


for ine in [20]: #ne takes 51 values uniformed spaced from 0.1 to 1.0 in units of 10^12 cm^{-2}
  
  pd_plot={}
  
  """
  files:
    "../data/hartree-fock/multi_flavor_HF/HF_IVC_set1_nonscreened_U0.044_epsr20.npz"
    "../data/hartree-fock/multi_flavor_HF/HF_IVC_set1_nonscreened_U0.052_epsr20.npz"
    "../data/hartree-fock/multi_flavor_HF/HF_IVC_set1_nonscreened_U0.052_epsr30.npz"
  """
  
  # NOTE: the published SI figures use the epsr20 files:
  #   SI Fig. 39 (u_D = 52 meV, eps = 20): ..._U0.052_epsr20.npz
  #   SI Fig. 40 (u_D = 44 meV, eps = 20): ..._U0.044_epsr20.npz
  # The epsr30 default below is as received and is not shown in the SI.
  pd_plot['fname']="../data/hartree-fock/multi_flavor_HF/HF_IVC_set1_nonscreened_U0.052_epsr30.npz"
  pd_plot['ine']=ine 

  main(pd_plot)

