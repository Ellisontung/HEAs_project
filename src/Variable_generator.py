import numpy as np
import torch


def create_latentcode_AE(rows,latent_mean_std,latent_dim=5):
  # Create sets of normal distribution latent code within latent mean std range.
  # example: rows = 2000
  latent_output = np.zeros((rows,latent_dim))
  for j in range(5):
    latent_output[:,j] = np.random.normal(latent_mean_std[j][0],latent_mean_std[j][1],rows)
  return torch.from_numpy(latent_output).float()

def create_target_ys(rows,toughness_value,shear_value):
  """This function is the function to create sets(rows) of y label"""
  # example: rows = 2000
  target_ys = np.zeros((rows,2))
  for i in range(rows):
    target_ys[i] = np.array([toughness_value,shear_value])
  return target_ys


def AE_sampler(N_samples,latent_mean_std,conds,Scaler,AE,surrogate_model):
    latent_code = create_latentcode_AE(N_samples,latent_mean_std)
    conds = create_target_ys(N_samples,conds[0],conds[1])
    conds = Scaler.transform(conds)
    conds = torch.tensor(conds).float()
    recipes = AE.decode(latent_code,conds)
    surr_value = surrogate_model(recipes)
    return recipes.detach().numpy(),surr_value.detach().numpy()

def g_sample(N_samples,conds,GAN,surrogate_model,Scaler):
    gen_input = torch.from_numpy(GAN.noise_sampler(N_samples, 10))
    conds = create_target_ys(N_samples,conds[0],conds[1])
    conds = Scaler.transform(conds)
    conds = torch.tensor(conds).float()
    recipes = GAN.generator(gen_input,conds)
    surr_value = surrogate_model(recipes)
    return recipes.detach().numpy(),surr_value.detach().numpy()



def preprocess_violinplot_AE(properties_values,Scaler,latent_mean_std,AE_model,surrogate_model,latent_dim = 5,interpolation = 20):
    #Decide the value want to explore 
    input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max(),20)
    input_shear = np.linspace(properties_values[:,1].min(),properties_values[:,1].max(),20)
    #Create latent code and condition code
    latent_code = create_latentcode_AE(2000,latent_mean_std,latent_dim=latent_dim)

    surr_value = np.zeros((2,2000,interpolation))

    for i in range(20):
        cond = create_target_ys(2000,input_toughness[i],input_shear[i])
        cond = Scaler.transform(cond)
        cond = torch.tensor(cond).float()
        recipes = AE_model.decode(latent_code,cond)
        surr_res = Scaler.inverse_transform(surrogate_model(recipes).detach().numpy())
        surr_value[0,:,i] = surr_res[:,0]
        surr_value[1,:,i] = surr_res[:,1]
        # surr_value[i] = Scaler.inverse_transform(surr_value[i])
    return surr_value,[input_toughness,input_shear]



def preprocess_violinplot_GAN(properties_values,Scaler,cGAN,surrogate_model,interpolation=20,latent_dim = 10):
  # make latentcode
  latent_code = cGAN.noise_sampler(2000,10)
  latent_code = torch.tensor(latent_code).float()

  #make input conditions
  input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max(),20)
  input_shear = np.linspace(properties_values[:,1].min(),properties_values[:,1].max(),20)

  surr_value = np.zeros((2,2000,interpolation))

  for i in range(20):
      cond = create_target_ys(2000,input_toughness[i],input_shear[i])
      cond = Scaler.transform(cond)
      cond = torch.tensor(cond).float()
      recipes = cGAN.generator(latent_code,cond)
      surr_res = Scaler.inverse_transform(surrogate_model(recipes).detach().numpy())
      surr_value[0,:,i] = surr_res[:,0]
      surr_value[1,:,i] = surr_res[:,1]
      # surr_value[i] = Scaler.inverse_transform(surr_value[i])
  return surr_value,[input_toughness,input_shear]








def preprocess_variance_map_GAN(properties_values,Scaler,cGAN,surrogate_model,interpolation = 5):
  #latent code setting
  latent_code = cGAN.noise_sampler(2000,10)
  latent_code = torch.tensor(latent_code).float()

  # conditions setting
  input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max(),interpolation)
  input_shear = np.linspace(properties_values[:,1].max(),properties_values[:,1].min(),interpolation)

  surr_value = np.zeros((interpolation*interpolation,2000,2))
  count = 0
  for i in range (interpolation):
      for j in range(interpolation):
          cond = create_target_ys(2000,input_toughness[i],input_shear[j])
          cond = Scaler.transform(cond)
          cond = torch.tensor(cond).float()
          recipes = cGAN.generator(latent_code,cond)
          surr_value[count] = surrogate_model(recipes).detach().numpy()
          surr_value[count] = Scaler.inverse_transform(surr_value[count])
          count += 1
  return surr_value,[input_toughness,input_shear]



def preprocess_variance_map_AE(properties_values,latent_mean_std_collection,Scaler,AE,surrogate_model,interpolation = 5):
  #latent code setting
  latent_code = latent_value=create_latentcode_AE(2000,latent_mean_std_collection)

  # conditions setting
  input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max(),interpolation)
  input_shear = np.linspace(properties_values[:,1].max(),properties_values[:,1].min(),interpolation)

  surr_value = np.zeros((interpolation*interpolation,2000,2))
  count = 0
  for i in range (interpolation):
      for j in range(interpolation):
          cond = create_target_ys(2000,input_toughness[i],input_shear[j])
          cond = Scaler.transform(cond)
          cond = torch.tensor(cond).float()
          recipes = AE.decode(latent_code,cond)
          surr_value[count] = surrogate_model(recipes).detach().numpy()
          surr_value[count] = Scaler.inverse_transform(surr_value[count])
          count += 1
  return surr_value,[input_toughness,input_shear]










def latent_code_inspector_cAE(cAE,Sequences_input,Conditions_input,neurons=5):
  """To observe the variance of latent space and return [min,max], [mean,std] for each neuron"""
  """This function is to generate latent code from series of inputs of compositions sequence and conditions"""
  #Pass with encoder of cAE

  """Turn input from numpy array to Tensor array"""
  Sequences_input = torch.from_numpy(Sequences_input)
  Conditions_input = torch.from_numpy(Conditions_input)
  """Input the Sequences and conditions to encoder and output latent code"""
  latent_code=cAE.encode(Sequences_input,Conditions_input).detach().numpy()
  """collect the data in [min,max],[mean,std] for each neuron"""
  min_Max_collection = np.zeros((neurons,2))
  mean_std_collection = np.zeros((neurons,2))
  for i in range(neurons):
      min_Max_collection[i] = np.array([latent_code[:,i].min(),latent_code[:,i].max()])
      mean_std_collection[i] = np.array([latent_code[:,i].mean(),latent_code[:,i].std()])
  return min_Max_collection,mean_std_collection,latent_code

def latent_code_inspector_cVAE(cVAE,Sequences_input,Conditions_input,neurons=5):
  """To observe the variance of latent space and return [min,max], [mean,std] for each neuron"""
  """This function is to generate latent code from series of inputs of compositions sequence and conditions"""
  #Pass with encoder of cVAE

  """Turn input from numpy array to Tensor array"""
  Sequences_input = torch.from_numpy(Sequences_input)
  Conditions_input = torch.from_numpy(Conditions_input)
  mu,var=cVAE.encode(Sequences_input,Conditions_input)
  latent_code = cVAE.reparameterize(mu,var).detach().numpy()
  """collect the data in [min,max] for each neuron"""
  min_Max_collection = np.zeros((neurons,2))
  mean_std_collection = np.zeros((neurons,2))
  for i in range(neurons):
      min_Max_collection[i] = np.array([latent_code[:,i].min(),latent_code[:,i].max()])
      mean_std_collection[i] = np.array([latent_code[:,i].mean(),latent_code[:,i].std()])
  return min_Max_collection,mean_std_collection,latent_code
  

def preprocess_interpolation_AE(AE,surrogate_model,scaler,latent_mean_std_collection,toughness,shear,interpolation=5):
    latent_code = create_latentcode_AE(2,latent_mean_std_collection,latent_dim=5)

    LATENT = torch.zeros(interpolation,latent_code.shape[1])
    for i in range(interpolation):
        LATENT[i] = latent_code[0]+(i+1)*(latent_code[1]-latent_code[0])

    conds = create_target_ys(interpolation,toughness,shear)
    conds = scaler.transform(conds)
    conds = torch.from_numpy(conds).float()
    Recipes = AE.decode(LATENT,conds)
    Results = surrogate_model(Recipes)
    return Recipes.detach().numpy(),Results.detach().numpy()