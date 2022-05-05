import numpy as np
import torch


def create_latentcode(rows,latent_mean_std):
  # Create sets of normal distribution latent code within latent mean std range.
  # example: rows = 2000
  latent_output = np.zeros((rows,5))
  for j in range(5):
    latent_output[:,j] = np.random.normal(latent_mean_std[j][0],latent_mean_std[j][1],rows)
  return torch.from_numpy(latent_output).float()

def create_target_ys(rows,shear_value,toughness_value):
  """This function is the function to create sets(rows) of y label"""
  # example: rows = 2000
  target_ys = np.zeros((rows,2))
  for i in range(rows):
    target_ys[i] = np.array([shear_value,toughness_value])
  return target_ys


def preprocess_for_violinplot(properties_values,Scaler,latent_mean_std,over_value):
    #Decide the value want to explore 
    over_t=over_value[0]
    over_s=over_value[1]
    input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max()+over_t,20)
    input_shear = np.linspace(properties_values[:,1].min(),properties_values[:,1].max()+over_s,20)
    #Create latent code and condition code
    latent_value = create_latentcode(2000,latent_mean_std)
    ys_collection = torch.zeros((20,2000,2))
    ys_temporary = np.zeros((2000,2))
    for i in range(20):
      ys_temporary = torch.tensor(Scaler.transform(create_target_ys(2000,input_toughness[i],input_shear[i]))).float()
      ys_collection[i] = ys_temporary

    return latent_value,ys_collection,[input_toughness,input_shear]

def calculate_mean(outputproperties):
  meandata=np.zeros((2,20))
  for i in range(20):
    meandata[0][i]=outputproperties[0,:,i].mean()
    meandata[1][i]=outputproperties[1,:,i].mean()
    
  return meandata

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