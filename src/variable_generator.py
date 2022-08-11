import numpy as np
import torch
from sklearn.metrics import pairwise

class WcGAN_variable_generator():
    def __init__(self,GAN_model,
    training_sequences,
    properties_values,
    properties_values_scaled,
    scaler,
    surrogate_model,
    latent_dim = 10):
        ## Collect preprocess data
        self.GAN_model = GAN_model
        self.training_sequences = training_sequences
        self.properties_values = properties_values
        self.properties_values_scaled = properties_values_scaled
        self.scaler = scaler
        
        self.surrogate_model = surrogate_model
        self.latent_dim=latent_dim

    def create_target_ys(self,N_samples,toughness_value,shear_value):
        target_ys = np.zeros((N_samples,2))
        for i in range(N_samples):
            target_ys[i] = np.array([toughness_value,shear_value])
        return target_ys

    def g_sample(self,N_samples,conds):
        gen_input = torch.from_numpy(self.GAN_model.noise_sampler(N_samples, 10))
        conds = self.create_target_ys(N_samples,conds[0],conds[1])
        conds = self.scaler.transform(conds)
        conds = torch.tensor(conds).float()
        recipes = self.GAN_model.generator(gen_input,conds)
        surr_value = self.surrogate_model.predict(recipes).detach().numpy()
        surr_value = self.scaler.inverse_transform(surr_value)
        return recipes.detach().numpy(),surr_value
    
    def preprocess_violinplot(self,N_samples=2000,conds_1="default",conds_2="default",interpo=20):
        # make latentcode
        latent_code = self.GAN_model.noise_sampler(N_samples,10)
        latent_code = torch.tensor(latent_code).float()

        if(conds_1=="default" and conds_2=="default"):
            input_toughness = np.linspace(self.properties_values[:,0].min(),self.properties_values[:,0].max(),interpo)
            input_shear = np.linspace(self.properties_values[:,1].min(),self.properties_values[:,1].max(),interpo)
        elif(len(conds_1)==2 and len(conds_2)==2):
            input_toughness = np.linspace(conds_1[0],conds_2[0],interpo)
            input_shear = np.linspace(conds_2[1],conds_2[1],interpo)
        else:
            raise "Please set both conds_1 and conds_2 in form as \"[1,2]\"(toughness,shear modulus) or set as \"default\""

        surr_value = np.zeros((2,N_samples,interpo))
        Recipes = np.zeros((interpo,N_samples,22))

        for i in range(20):
            cond = self.create_target_ys(N_samples,input_toughness[i],input_shear[i])
            cond = self.scaler.transform(cond)
            cond = torch.tensor(cond).float()
            recipes = self.GAN_model.generator(latent_code,cond)
            surr_res = self.scaler.inverse_transform(self.surrogate_model.predict(recipes).detach().numpy())
            Recipes[i] = recipes.detach().numpy()
            surr_value[0,:,i] = surr_res[:,0]
            surr_value[1,:,i] = surr_res[:,1]
            # surr_value[i] = Scaler.inverse_transform(surr_value[i])
        return surr_value,[input_toughness,input_shear],Recipes

    def preprocess_mapping(self,N_samples=2000,interpo=5):
        #latent code setting
        latent_code = self.GAN_model.noise_sampler(N_samples,self.latent_dim)
        latent_code = torch.tensor(latent_code).float()

        # conditions setting
        input_toughness = np.linspace(self.properties_values[:,0].min(),self.properties_values[:,0].max(),interpo)
        input_shear = np.linspace(self.properties_values[:,1].max(),self.properties_values[:,1].min(),interpo)

        surr_value = np.zeros((interpo*interpo,N_samples,2))
        count = 0
        for i in range (interpo):
            for j in range(interpo):
                cond = self.create_target_ys(N_samples,input_toughness[i],input_shear[j])
                cond = self.scaler.transform(cond)
                cond = torch.tensor(cond).float()
                recipes = self.GAN_model.generator(latent_code,cond)
                surr_value[count] = self.surrogate_model.predict(recipes).detach().numpy()
                surr_value[count] = self.scaler.inverse_transform(surr_value[count])
                count += 1
        return surr_value,[input_toughness,input_shear]
        
    def preprocess_interpolation_GAN_fixed_conds(self,conds,interpo=5):
        latent_code_A,latent_code_B = self.GAN_model.noise_sampler(2,self.latent_dim)
        latent_code = np.zeros((interpo,self.latent_dim))
        for i in range(interpo):
            latent_code[i] = (latent_code_A + latent_code_B-latent_code_A)/interpo*i
        latent_code = torch.from_numpy(latent_code).float()
        conds = np.repeat([conds],interpo,axis = 0)
        conds = self.scaler.transform(conds)
        conds = torch.from_numpy(conds).float()
        Recipes = self.GAN_model.generator(latent_code,conds)
        Results = self.scaler.inverse_transform(self.surrogate_model.predict(Recipes).detach().numpy())
        return Recipes.detach().numpy(),Results

    def cos_mean(self,rand_sample=100):
        # compare cosine similarity with dataset
        pick = np.random.randint(500,size=rand_sample)
        value = np.zeros((rand_sample))
        for i in range(rand_sample):
            # sampling with randomly picked conditions in dataset
            GAN_samples = self.g_sample(100,self.properties_values[pick[i]])
            # Compare the similarity between real data and generated data and collect the mean value
            value[i] = pairwise.cosine_similarity(self.training_sequences,GAN_samples).mean()
        return value,value.mean()













class AE_variable_generator():
    def __init__(self,AE_model,
    training_sequences,
    properties_values,
    properties_values_scaled,
    scaler,
    surrogate_model,
    model_name = "cAE",
    latent_dim = 5):
        ## Collect preprocess data
        self.AE_model = AE_model
        self.training_sequences = training_sequences
        self.properties_values = properties_values
        self.properties_values_scaled = properties_values_scaled
        self.scaler = scaler
        
        self.surrogate_model = surrogate_model
        self.latent_dim=latent_dim

        ## Collect initial latent space variables from encoder of AE, including latent code, and its std, mean, max and min of each neuron
        if(model_name=="cAE"):
            self.latent_min_Max,self.latent_mean_std,self.latent_code = self.__latent_code_inspector_cAE()
        elif(model_name=="cVAE"):
            self.latent_min_Max,self.latent_mean_std,self.latent_code = self.__latent_code_inspector_cVAE()
        else:
            raise "The model_name is neither \"cAE\" nor \"cVAE\", Please check that you input correct model name"

        

    # The two functions below are to initialize parameters (attributed with private functions)
    def __latent_code_inspector_cAE(self):
        #Pass with encoder of cAE
        
        Sequences_input = torch.from_numpy(self.training_sequences)
        Conditions_input = torch.from_numpy(self.properties_values_scaled)
        
        latent_code=self.AE_model.model.encode(Sequences_input,Conditions_input).detach().numpy()
        
        min_Max_collection = np.zeros((self.latent_dim,2))
        mean_std_collection = np.zeros((self.latent_dim,2))
        for i in range(self.latent_dim):
            min_Max_collection[i] = np.array([latent_code[:,i].min(),latent_code[:,i].max()])
            mean_std_collection[i] = np.array([latent_code[:,i].mean(),latent_code[:,i].std()])
        return min_Max_collection,mean_std_collection,latent_code

    def __latent_code_inspector_cVAE(self):
        #Pass with encoder of cVAE

        Sequences_input = torch.from_numpy(self.training_sequences)
        Conditions_input = torch.from_numpy(self.properties_values_scaled)
        mu,var = self.AE_model.model.encode(Sequences_input,Conditions_input)
        latent_code = self.AE_model.model.reparameterize(mu,var).detach().numpy()
        
        min_Max_collection = np.zeros((self.latent_dim,2))
        mean_std_collection = np.zeros((self.latent_dim,2))
        for i in range(self.latent_dim):
            min_Max_collection[i] = np.array([latent_code[:,i].min(),latent_code[:,i].max()])
            mean_std_collection[i] = np.array([latent_code[:,i].mean(),latent_code[:,i].std()])
        return min_Max_collection,mean_std_collection,latent_code


    def create_latentcode(self,N_samples):
        latent_output = np.zeros((N_samples,self.latent_dim))
        for j in range(self.latent_dim):
            latent_output[:,j] = np.random.normal(self.latent_mean_std[j][0],self.latent_mean_std[j][1],N_samples)
        return torch.from_numpy(latent_output).float()

    def create_target_ys(self,N_samples,toughness_value,shear_value):
        target_ys = np.zeros((N_samples,2))
        for i in range(N_samples):
            target_ys[i] = np.array([toughness_value,shear_value])
        return target_ys

    def AE_sampler(self,N_samples,conds):
        latent_code = self.create_latentcode(N_samples)
        conds = self.create_target_ys(N_samples,conds[0],conds[1])
        conds = self.scaler.transform(conds)
        conds = torch.tensor(conds).float()
        recipes = self.AE_model.decode(latent_code,conds)
        surr_value = self.surrogate_model.predict(recipes)
        return recipes.detach().numpy(),surr_value.detach().numpy()
    
    def preprocess_violinplot(self,N_samples=2000,conds_1="default",conds_2="default",interpo=20):
        #N_samples represents as desired numbers of samples in each pillar
        #Decide the value want to explore 
        if(conds_1=="default" and conds_2=="default"):
            input_toughness = np.linspace(self.properties_values[:,0].min(),self.properties_values[:,0].max(),interpo)
            input_shear = np.linspace(self.properties_values[:,1].min(),self.properties_values[:,1].max(),interpo)
        elif(len(conds_1)==2 and len(conds_2)==2):
            input_toughness = np.linspace(conds_1[0],conds_2[0],interpo)
            input_shear = np.linspace(conds_2[1],conds_2[1],interpo)
        else:
            raise "Please set both conds_1 and conds_2 in form as \"[1,2]\"(toughness,shear modulus) or set as \"default\""
        #Create latent code and condition code
        latent_code = self.create_latentcode(N_samples)

        surr_value = np.zeros((2,N_samples,interpo))
        Recipes = np.zeros((interpo,N_samples,22))

        for i in range(20):
            cond = self.create_target_ys(N_samples,input_toughness[i],input_shear[i])
            cond = self.scaler.transform(cond)
            cond = torch.tensor(cond).float()
            recipes = self.AE_model.decode(latent_code,cond)
            surr_res = self.scaler.inverse_transform(self.surrogate_model.predict(recipes).detach().numpy())
            Recipes[i] = recipes.detach().numpy()
            surr_value[0,:,i] = surr_res[:,0]
            surr_value[1,:,i] = surr_res[:,1]
            # surr_value[i] = Scaler.inverse_transform(surr_value[i])
        return surr_value,[input_toughness,input_shear],Recipes
        
    def preprocess_mapping(self,N_samples=2000,interpo=5):
        #latent code setting
        latent_code = latent_value=self.create_latentcode(N_samples)

        # conditions setting
        input_toughness = np.linspace(self.properties_values[:,0].min(),self.properties_values[:,0].max(),interpo)
        input_shear = np.linspace(self.properties_values[:,1].max(),self.properties_values[:,1].min(),interpo)

        surr_value = np.zeros((interpo*interpo,N_samples,2))
        count = 0
        for i in range (interpo):
            for j in range(interpo):
                cond = self.create_target_ys(N_samples,input_toughness[i],input_shear[j])
                cond = self.scaler.transform(cond)
                cond = torch.tensor(cond).float()
                recipes = self.AE_model.decode(latent_code,cond)
                surr_value[count] = self.surrogate_model.predict(recipes).detach().numpy()
                surr_value[count] = self.scaler.inverse_transform(surr_value[count])
                count += 1
        return surr_value,[input_toughness,input_shear]

    def preprocess_interpolation_AE_fixed_conds(self,conds,interpo=5):
        latent_code_A,latent_code_B = self.create_latentcode(2)
        latent_code = torch.zeros((interpo,5))
        for i in range(interpo):
            latent_code[i] = latent_code_A+(latent_code_B-latent_code_A)/interpo*i
        conds = np.array([conds])
        conds = np.repeat(conds,interpo,axis = 0)
        conds = self.scaler.transform(conds)
        conds = torch.from_numpy(conds).float()

        Recipes = self.AE_model.decode(latent_code,conds)
        Results = self.surrogate_model.predict(Recipes).detach().numpy()
        Results = self.scaler.inverse_transform(Results)
        return Recipes.detach().numpy(),Results
    
    def cos_mean(self,rand_sample=100):
        # compare cosine similarity with dataset
        pick = np.random.randint(500,size=rand_sample)
        value = np.zeros((rand_sample))
        for i in range(rand_sample):
            # sampling with randomly picked conditions in dataset
            cAE_samples = self.AE_sampler(100,self.properties_values[pick[i]])
            # Compare the similarity between real data and generated data and collect the mean value
            value[i] = pairwise.cosine_similarity(self.training_sequences,cAE_samples[0]).mean()
        return value,value.mean()
