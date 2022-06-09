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
grid_style = (0, (5, 5))

cmap_format = matplotlib.colors.LinearSegmentedColormap.from_list("",[colorformat["background"],colorformat["bar"]])

def store_fig_pdf(fig,name):
    fig.savefig('Sample_figure/{}.pdf'.format(name))
def store_fig_png(fig,name):
    fig.savefig('Sample_figure\{}.png'.format(name))

def input_scatter_plot(properties_values):
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

    return fig



def latent_code_variance(latent_code,neurons=5,ymin=-10,ymax=10):
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

  return fig



def variance_map(forim,inter_cond,interpolation=5):
    fig,axes = plt.subplots(interpolation,interpolation,figsize=(50,50),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=40)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=40)
    fig.suptitle("Outputs of specific values maping",color=colorformat["text"],fontsize=50)
    # forim,inter_cond = preprocess_variance_map(properties_values,scaler,WcGAN,surrogate_model)
    count = 0
    for i in range(interpolation): #shear
        for j in range(interpolation): #toughness
            axes[j][i].hist2d(forim[count,:,0],forim[count,:,1],bins=20,range=np.array([[inter_cond[0].min(),inter_cond[0].max()],[inter_cond[1].min(),inter_cond[1].max()]]),cmap=cmap_format) #Blues
            axes[j][i].scatter(inter_cond[0][i],inter_cond[1][j],marker = 's',color =colorformat["Highlight"],linewidths = 15)
            axes[j][i].set_title('toughness = {t:.2f}, shear = {s:.2f}'.format(t=inter_cond[0][i],s=inter_cond[1][j]))
            count += 1
    return fig



def violin_plot(collected_data,collected_conds):
  #collected data in shape with (2, 2000, 20)
  #pick shear modulus or toughness
  #Over_value can offer the way to explore the properties out of the range
  #Violin plot (data,objects) (2000,20)

  properties_name=["toughness","shear modulus"]
  #calculate mean
  def calculate_mean(outputproperties):
    meandata = np.zeros((2,20))
    for i in range(20):
      meandata[0][i]= outputproperties[0,:,i].mean()
      meandata[1][i]= outputproperties[1,:,i].mean()
    return meandata

  mean_data = calculate_mean(collected_data)

  #Plotting violin plot
  fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(200,100),constrained_layout=True)
  fig.set_facecolor(colorformat["background"])
  # fig.suptitle("Outputs of specific values maping",color=colorformat["text"],fontsize=200)
  for i in range(2):
    if i == 0:
      w=3
    else:
      w=0.5 
    violin = axes[i].violinplot(collected_data[i],collected_conds[i],widths=w,showextrema=False,showmeans=False)
    for pc in violin['bodies']:
      pc.set_facecolor(colorformat['bar'])
      pc.set_edgecolor(colorformat['line'])
      pc.set_alpha(1)
    axes[i].grid(visible=True,linewidth=2)
    axes[i].set_facecolor(colorformat["background"])
    axes[i].set_xlabel("setting",fontsize=120)
    axes[i].set_ylabel("generated and expected",fontsize=120)
    axes[i].set_title(properties_name[i],fontsize=150)
    axes[i].plot(collected_conds[i],mean_data[i],c=colorformat["Highlight"],linewidth=20,label="generated")
    axes[i].plot(collected_conds[i],collected_conds[i],"--",c=colorformat["Highlight"],linewidth=20,label="expected")
    axes[i].legend(fontsize=150,loc=2)

  return fig

#Need to modify. Combine with scatter function.
def input_scatter_plot_slice(properties_values):
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
 
    return fig


def sample_plot(Xt,dset,title = "default",size=1,interval = 1):
    fig,axes = plt.subplots(5*size,1,figsize=(15,5*size),tight_layout=False)
    fig.set_facecolor(colorformat["background"])
    minor_tick = np.arange(0.5,22.5,1)
    major_tick = np.arange(0,22,1)
    for i in range(5*size):
        im = axes[i].imshow(Xt[interval*i].reshape(1,-1),cmap = cmap_format,vmin=0,vmax=0.6)
        axes[i].set_xticks(major_tick)
        axes[i].set_xticklabels(dset.elements)
        axes[i].set_yticklabels("")
        axes[i].tick_params(axis="both",length=0)
        axes[i].set_xticks(minor_tick,minor=True)
        axes[i].grid(axis = "x",visible=True,which="minor",c=colorformat["line"],linestyle=(0, (5, 5)))
    fig.suptitle(title,color=colorformat["text"],fontsize=20*size)
    fig.supxlabel("Concentration of elements",color=colorformat["text"],fontsize=15*size)
    plt.colorbar(im,ax=axes.ravel().tolist())
    return fig



def visilize_accuracy_GAN(real,fake,scaler):
    """Use for GAN to compare the difference"""
    fig,axes = plt.subplots(10,1,figsize=(5,50),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    fig.suptitle("WcGAN accuracy",color=colorformat["text"],fontsize=15)
    for i in range(10):
        data = scaler.inverse_transform(real[i])-scaler.inverse_transform(fake[i])
        axes[i].hist2d(data[:,0],data[:,1],bins=40,cmap=cmap_format) #Blues
        axes[i].set_xlim(-20,20)
        axes[i].set_ylim(-20,20)
        axes[i].set_xlabel("toughness")
        axes[i].set_ylabel("shear modulus")
        axes[i].set_title("epoch = {}".format(i))
        axes[i].set_facecolor(colorformat["background"])
    return fig