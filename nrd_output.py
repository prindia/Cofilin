# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:41:21 2020

@author: kblackw1
"""
import numpy as np
import h5py as h5
import h5utilsV2 as h5utils

Avogadro=6.02214179e14 #to convert to nanoMoles
mol_per_nM_u3=Avogadro*1e-15 #0.6022 = PUVC

class nrdh5_output(object):
    def __init__(self,ftuple):
        self.fname=ftuple[0]
        self.data = h5.File(self.fname,"r")
        self.parval=ftuple[1]
        self.TotVol=self.data['model']['grid'][:]['volume'].sum()
        self.maxvols=len(self.data['model']['grid'])
        self.trials=[a for a in self.data.keys() if 'trial' in a]
        self.seeds=[self.data[trial].attrs['simulation_seed'] for trial in self.trials]
        self.outputsets=self.data[self.trials[0]]['output'].keys()
        self.head=None
        self.spinelist=None
        self.dsm_vox=None
        self.spatial_dict=None
        print('file:',self.fname,'parameters:',self.parval,'volume',self.TotVol,'voxels',self.maxvols,'trials',self.trials)
        
    def region_structures(self,dendname,submembname,spinehead,stimspine=None): #spinehead may not need to be parameter
        structType=self.data['model']['grid'][:]['type']
        self.region_list,self.region_dict,self.region_struct_dict=h5utils.subvol_list(structType,self.data['model'])
        #Replace the following with test for whether there is more than one "group"
        if spinehead in self.region_dict.keys():
            self.head=spinehead
            self.spinelist,self.spinevox=h5utils.multi_spines(self.data['model'])
            self.stim_spine_index=[self.spinelist.index(stimsp) for stimsp in stimspine]        #
            #create "group" dictionary-spinelist & spinevox, with one more for nonspine voxels
        dsm_name=dendname+submembname
        if dsm_name in self.region_struct_dict.keys():
            self.dsm_vox=self.region_struct_dict[dsm_name]
            self.dsm_index=list(self.region_struct_dict.keys()).index(dsm_name)
            
    def spatial_structures(self,bins,dendname):
        self.spatial_dict=h5utils.spatial_average(bins,dendname,self.data['model']['grid'])
        vol=[x['vol'] for x in self.spatial_dict.values()]
        if any(v==0 for v in vol):
            print ("**********Too many bins for spatial average****************")
#
    def rows_and_columns(self,molecules,args):
        #which columns in the data set contain counts of molecules of interest
        self.all_molecules=h5utils.decode(self.data['model']['species'][:])
        if molecules is not None:
            self.molecules = [mol for mol in molecules if mol in self.all_molecules]               
        else:
            self.molecules=self.all_molecules 
        out_location,dt,rows=h5utils.get_mol_info(self.data, self.molecules)
        self.out_location=out_location
        self.dt=dt
        self.rows=rows
        if len(args)>3: 
            arg=args[3]
        else:
            arg=None
        #Which "rows" should be used for baseline value, specifed in args[3].  If different for each file then will have problems later
        sstart,ssend=h5utils.sstart_end(self.molecules,self.out_location,self.dt,self.rows,arg)
        self.sstart=sstart
        self.ssend=ssend
        
    def molecule_population(self):
        self.counts={}
        self.time={}
        self.OverallMean={}
        for imol,molecule in enumerate(self.molecules):
              counts,time=h5utils.get_mol_pop(self,molecule)
              self.counts[molecule]=counts
              self.time[molecule]=time
              #sum over voxles, convert to concentration
              self.OverallMean[molecule]=np.sum(counts[:,:,:],axis=2)/(self.TotVol*mol_per_nM_u3)
        
    def average_over_voxels(self):
        #calculate various averages.  These will be used for plotting and output
        self.output_labels={'struct':{},'region':{}}
        self.means={'struct':{},'region':{}}
        if self.spinelist:
            self.output_labels['spines']={};self.means['spines']={}
        if self.spatial_dict:
            self.output_labels['space']={}; self.means['space']={}
        for imol,mol in enumerate(self.molecules):
            #calculate region-structure means, such as dendrite submembrane and dendrite cytosol
            #dimensions: number of trials x time samples x number of regions
            #labels hold the name of the region / structure, such as dendrite submembrane and dendrite cytosol
            self.output_labels['struct'][mol],self.means['struct'][mol]=h5utils.region_means_dict(self,mol,self.region_struct_dict)            
            #regions, such as dendrite, soma, spine head
            self.output_labels['region'][mol],self.means['region'][mol]=h5utils.region_means_dict(self,mol,self.region_dict)
            #if more than one spine, calculate individual spine means
            if self.spinelist:
                self.output_labels['spines'][mol],self.means['spines'][mol]=h5utils.region_means_dict(self,mol,self.spinevox)
            if self.spatial_dict:
                self.output_labels['space'][mol],self.means['space'][mol]=h5utils.region_means_dict(self,mol,self.spatial_dict)

    def write_average(self):
        import os ########### NEED TO PUT SPATIAL STUFF IN SEPARATE FILE?
        for mol in self.molecules:
            outfilename=os.path.splitext(os.path.basename(self.fname))[0]+mol+'_avg.txt'            
            output_means=np.mean(self.OverallMean[mol],axis=0)
            output_std=np.std(self.OverallMean[mol],axis=0)
            mean_header=mol+'_'.join([str(q) for q in self.parval])+'_All '
            if self.maxvols>1:
                for mean_dict in self.means.values():
                    output_means=np.column_stack((output_means,np.mean(mean_dict[mol],axis=0)))
                    output_std=np.column_stack((output_std,np.std(mean_dict[mol],axis=0)))
                for label in self.output_labels.values():
                    mean_header=mean_header+label[mol]
            std_header='_std '.join(mean_header.split())
            output_header='Time '+ mean_header+std_header+'_std\n'
            #print(outfilename,output_header)
            f=open(outfilename, 'w')
            f.write(output_header)
            np.savetxt(f, np.column_stack((self.time[mol],output_means,output_std)), fmt='%.4f', delimiter=' ')
            f.close()
            
    def total_subspecies(self,tot_species,sub_species,outset='__main__'):
        samples=len(self.data['trial0']['output'][outset]['times'])
        self.endtime=self.data['trial0']['output'][outset]['times'][-1]
        self.ss_tot=np.zeros((len(tot_species),len(self.trials),samples))
        self.tot_species={t:[] for t in tot_species}
        if self.dsm_vox:
            self.dsm_tot=np.zeros((len(tot_species),len(self.trials),samples))
        if self.spinelist:
            self.head_tot=np.zeros((len(tot_species),len(self.trials),samples))
        for imol,mol in enumerate(tot_species):
            mol_set=[]
            #first set up arrays of all species (sub_species) that are a form of the molecule
            if mol in sub_species.keys():
                mol_set=sub_species[mol]
            else:
                mol_set=[subsp for subsp in h5utils.decode(self.data['model']['output'][outset]['species'][:]) if mol in subsp]
            self.tot_species[mol]=mol_set
            #second, find molecule index of the sub_species and total them
            for subspecie in mol_set:
                mol_index=h5utils.get_mol_index(self.data,outset,subspecie)
                for trialnum,trial in enumerate(self.trials):
                    mol_pop=self.data[trial]['output'][outset]['population'][:,:,mol_index]
                    self.ss_tot[imol,trialnum,:]+=np.sum(mol_pop,axis=1)/self.TotVol/mol_per_nM_u3
                    #then total sub_species in submembrane and spine head, if they exist
                    if self.dsm_vox:
                        self.dsm_tot[imol,trialnum,:]+=np.sum(mol_pop[:,self.dsm_vox['vox']],axis=1)/self.dsm_vox['vol']*self.dsm_vox['depth']/mol_per_nM_u3
                    if self.spinelist: 
                        self.head_tot[imol,trialnum,:]+=np.sum(mol_pop[:,self.region_dict[self.head]['vox']],axis=1)/self.region_dict[self.head]['vol']/mol_per_nM_u3
            outputline=str(self.parval)+' TOTAL: '+str(np.round(self.ss_tot[imol,0,0],3))+' nM'
            if self.dsm_vox:
                outputline +=',  sp: '+str(np.round(self.head_tot[imol,0,0],3))+' nM'
            if self.spinelist:
                outputline +=',  dsm: '+str(np.round(self.dsm_tot[imol,0,0],3))+' pSD'
            print(outputline)
  
    def print_head_stats(self):
        for imol,mol in enumerate(self.molecules):
            outputline=mol.rjust(14)
            if self.spinelist:
                if len(self.spinelist)>1:
                    headmean=[np.mean(np.mean(self.means['spines'][mol][:,self.sstart[imol]:self.ssend[imol],sp],axis=1),axis=0) for sp in self.stim_spine_index]
                    headmax=[np.mean(np.max(self.means['spines'][mol][:,self.ssend[imol]:,sp],axis=1),axis=0) for sp in self.stim_spine_index]
                else:
                    headmean=[np.mean(np.mean(self.means['region'][mol][:,self.sstart[imol]:self.ssend[imol],self.head],axis=1),axis=0)]
                    headmax=[np.mean(np.max(self.means['region'][mol][:,self.ssend[imol]:,self.head],axis=1),axis=0)]
                outputline+="   head ss: "+' '.join([str(np.round(h,3)) for h in headmean])+'pk: '+' '.join([str(np.round(h,3)) for h in headmax])
            if self.dsm_vox:
                dsm_mean=np.mean(np.mean(self.means['struct'][mol][:,self.sstart[imol]:self.ssend[imol],self.dsm_index],axis=1),axis=0)
                dsm_max=np.mean(np.max(self.means['struct'][mol][:,self.ssend[imol]:,self.dsm_index],axis=1),axis=0)
                outputline+="   dend sm: %8.4f pk %8.4f" %(dsm_mean,dsm_max)
            print(outputline)
