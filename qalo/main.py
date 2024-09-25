from qalo.arguments import arguments
from qalo.op.annealer import *
from qalo.op.fm import *
from qalo.op.mlp import *
from qalo.utils.data import write_libffm_data, generate_ffm_data
from qalo.utils.utils import redirect

import os
from tqdm import tqdm


class qalo(arguments):
    def __init__(self):
        super().__init__()
        
    def annealing_module(self, initial_composition, annealer):
        Nx, Ny, Nz = self.spc_size[0], self.spc_size[1], self.spc_size[2]
        nsites = int(len(self.unit_site)) * Nx * Ny * Nz
        nspecies = len(self.elements)
        composition = initial_composition
        energy_solution = []
        structuredf = pd.DataFrame()
        
        initial_k = self.qa_relax * self.qa_constr
        growth_rate = self.qa_relax
        
        for i in range(self.qa_mix_circle):
            k = growth_rate ** (1 - i / self.qa_mix_circle)
            annealing_loose = annealer(nspecies=nspecies, nsites=nsites,
                                       placeholder=self.qa_constr, fmpath=self.fm_directory,
                                       composition=composition, annealer=annealer, mode="loose", 
                                       ks=k)
            annealing_loose.run_annealer(n_sim=self.qa_shots)
            sdfl = annealing_loose.extract_solutions()
            energy = []
            energy_min = 0
            for n, s in sdfl.iterrows():
                e = snap_model_inference(binvec=np.array(s.values), 
                                        infile=os.path.join(self.input_directory, self.lmps_infile), 
                                        coeffile=os.path.join(self.lmps_directory, self.lmps_coeffile), 
                                        path_of_tmp=self.tmp_directory)
                energy.append(e)
                energy_min = min(energy_min, e)
            
            if i != self.qa_mix_circle - 1:
                min_index = energy.index(energy_min)
                min_structure = sdfl.iloc[min_index, :].tolist()
                for j in range(nspecies):
                    composition[j] = sum(min_structure[j * nsites: (j + 1) * nsites])
            else:
                energy_solution = energy
                structuredf = sdfl
    
    def run(self):
        print("Converting DFT data to libffm...")
        libffm_data = write_libffm_data(datapath=self.dft_data_directory, fmpath=self.fm_directory, spc_size=self.spc_size, unit_site=self.unit_site)
        ffmModel = ffm(lr=self.fm_learning_rate, reg=self.fm_reg_lambda, opt=self.fm_opt, k=self.fm_latent_space, epoch=self.fm_epoch, metric=self.fm_metric)
        composition = self.init_composition
        print("Finished! Start optimization!")
        print("Total iteration: ", self.iterations)
        print("System: ", self.elements)
        for i in tqdm(range(self.iterations)):
            generate_ffm_data(fmpath=self.fm_directory, sample_ratio=self.fm_sampling_ratio)
            with redirect(os.path.join(self.output_directory, "xlearn.log")):
                ffmModel.train(trainSet=os.path.join(self.fm_directory, "train_ffm.txt"),
                               validSet=os.path.join(self.fm_directory, "valid_ffm.txt"),
                               model_txt=os.path.join(self.fm_directory, "model.txt"),
                               model_out=os.path.join(self.fm_directory, "train.model"),
                               restart=True)
            composition = self.annealing_module(initial_composition=composition, annealer=self.qa_type)