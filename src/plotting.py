import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors
import src.Variable_generator as vg


colorformat = {
    "background":"#FFFFFF",
    "text":"#140812",
    "Highlight":"#730000",
    "bar":"#333F67",
    "line":"#223542"
    }
order=[
    "First",
    "Second",
    "Third",
    "Forth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth"
]

cmap_format = matplotlib.colors.LinearSegmentedColormap.from_list("",[colorformat["background"],colorformat["bar"]])

def input_scatter_plot(properties_values,savefig=False):
    """plot the inputs of data in scatter plot"""
    fig,axes = plt.subplots(nrows=1,ncols=1,constrained_layout=True,figsize=(15,10))
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=20)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=20)
    fig.suptitle("Inputs variance",color=colorformat["text"],fontsize=30)
    axes.scatter(properties_values[:,0],properties_values[:,1],c=colorformat["bar"],linewidths=1)
    axes.set_xlim(0,190)
    axes.set_ylim(0,28)
    axes.grid(visible=True)
    axes.set_facecolor(colorformat["background"])
    if savefig==True:
        fig.savefig('Sample_figure\Inputs scatter.pdf')
    return fig



def latent_code_variance(latent_code,neurons=5,ymin=-10,ymax=10,savefig=False):
  """plotting function"""
  fig,axes=plt.subplots(neurons,1,figsize=(8,2.4*neurons),tight_layout=True)
  fig.set_facecolor(colorformat["background"])
  fig.supxlabel("Probility distribution",color=colorformat["text"],fontsize=4*neurons)
  fig.supylabel("Times",color=colorformat["text"],fontsize=4*neurons)
  fig.suptitle("Latent Space",color=colorformat["text"],fontsize=5*neurons)
  for i in range(neurons):
      axes[i].hist(latent_code[:, i],orientation='vertical',color=colorformat['bar'])
      axes[i].set_xlim(ymin,ymax)
      axes[i].set_ylim(0,200)
      axes[i].set_xlabel("{} neuron".format(order[i]))
      axes[i].grid(visible=True)
      axes[i].set_facecolor(colorformat["background"])
  if savefig==True:
      fig.savefig('Sample_figure\Latent_space.pdf')

  return fig

def output_variance_map(properties_values,latent_range,Scaler,surrogate_model,cAE,numbers_of_dividen=5,savefig=False):
    #over_t and over_s are defined as the exceed toughness and shear range want to explore
    over_t=0
    over_s=0
    input_toughness = np.linspace(properties_values[:,0].min(),properties_values[:,0].max()+over_t,numbers_of_dividen)
    input_shear = np.linspace(properties_values[:,1].max()+over_s,properties_values[:,1].min(),numbers_of_dividen)
    fig,axes = plt.subplots(numbers_of_dividen,numbers_of_dividen,figsize=(50,50),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=40)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=40)
    fig.suptitle("Outputs of specific values maping",color=colorformat["text"],fontsize=50)
    for i in range(numbers_of_dividen): #shear
        for j in range(numbers_of_dividen): #toughness
            latent_value=vg.create_latentcode(2000,latent_range)
            target_ys = vg.create_target_ys(2000,input_toughness[i],input_shear[j])
            target_ys = torch.tensor(Scaler.transform(target_ys)).float()
            forim = Scaler.inverse_transform(surrogate_model(cAE.decode(latent_value,target_ys)).detach().numpy())
            axes[j][i].hist2d(forim[:,0],forim[:,1],bins=20,range=np.array([[properties_values[:,0].min(),properties_values[:,0].max()+over_t],[properties_values[:,1].min(),properties_values[:,1].max()+over_s]]),cmap=cmap_format) #Blues
            axes[j][i].scatter(input_toughness[i],input_shear[j],marker = 's',color =colorformat["Highlight"],linewidths = 10)
            axes[j][i].set_title('toughness = {t:.2f}, shear = {s:.2f}'.format(t=input_toughness[i],s=input_shear[j]))
    if savefig==True:
        fig.savefig('Sample_figure\Mapping.pdf')
    return fig

def violin_plot(surrogate_model,model,Scaler,preprocess_data,savefig=False):
    #pick shear modulus or toughness
    #Over_value can offer the way to explore the properties out of the range
    #Violin plot (data,objects) (2000,20)

    input_properties = preprocess_data[2]
    properties_name=["toughness","shear modulus"]
    #Pass through decoder
    output_properties = np.zeros((2,2000,20))
    for i in range(20):
      temp = Scaler.inverse_transform(surrogate_model(model.decode(preprocess_data[0],preprocess_data[1][i])).detach().numpy())
      output_properties[0,:,i] = temp[:,0]
      output_properties[1,:,i] = temp[:,1]
    #calculate mean
    mean_data = vg.calculate_mean(output_properties)

    #Plotting violin plot
    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(200,100),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    # fig.suptitle("Outputs of specific values maping",color=colorformat["text"],fontsize=200)
    for i in range(2):
      if i == 0:
        w=3
      else:
        w=0.5 
      violin = axes[i].violinplot(output_properties[i],input_properties[i],widths=w,showextrema=False,showmeans=False)
      for pc in violin['bodies']:
        pc.set_facecolor(colorformat['bar'])
        pc.set_edgecolor(colorformat['line'])
        pc.set_alpha(1)
      axes[i].grid(visible=True,linewidth=2)
      axes[i].set_facecolor(colorformat["background"])
      axes[i].set_xlabel("setting",fontsize=120)
      axes[i].set_ylabel("generated and expected",fontsize=120)
      axes[i].set_title(properties_name[i],fontsize=150)
      axes[i].plot(input_properties[i],mean_data[i],c=colorformat["Highlight"],linewidth=20,label="generated")
      axes[i].plot(input_properties[i],input_properties[i],"--",c=colorformat["Highlight"],linewidth=20,label="expected")
      axes[i].legend(fontsize=150,loc=2)
    if savefig==True:
      fig.savefig('Sample_figure\Violin.pdf')
    return fig
  

#Need to modify. Combine with scatter function.
def input_scatter_plot_slice(properties_values,savefig=False):
    """plot the inputs of data in scatter plot"""
    fig,axes = plt.subplots(nrows=1,ncols=1,constrained_layout=True,figsize=(15,10))
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=20)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=20)
    fig.suptitle("Inputs variance",color=colorformat["text"],fontsize=30)
    axes.scatter(properties_values[:,0],properties_values[:,1],c=colorformat["bar"],linewidths=1)
    axes.plot([properties_values[:,0].min(),properties_values[:,0].max()],[properties_values[:,1].min(),properties_values[:,1].max()],"--",c=colorformat["Highlight"],linewidth=4)
    axes.set_xlim(0,190)
    axes.set_ylim(0,28)
    axes.grid(visible=True)
    axes.set_facecolor(colorformat["background"])
    if savefig==True:
        fig.savefig('Sample_figure\Inputs scatter.pdf')
    return fig