import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import pairwise


#The dict is to unify the color format of all figure
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


#This two functions are to store image files quickly in samples folder
def store_fig_pdf(fig,name):
    fig.savefig('Sample_figure/{}.pdf'.format(name))
def store_fig_png(fig,name):
    fig.savefig('Sample_figure/{}.png'.format(name))

def input_scatter_plot(properties_values):
    """plot the inputs of data in scatter plot"""
    fig,axes = plt.subplots(nrows=1,ncols=1,constrained_layout=True,figsize=(15,10))
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel(r"Toughness ($\frac{MPa}{m^{1/2}})$",color=colorformat["text"],fontsize=20)
    fig.supylabel("Shear modulus (GPa)",color=colorformat["text"],fontsize=20)
    fig.suptitle("Training data",color=colorformat["text"],fontsize=30)
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
    fig.supxlabel(r"$z_i$",color=colorformat["text"],fontsize=4*neurons)
    fig.supylabel("Frequency",color=colorformat["text"],fontsize=4*neurons)
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
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=50)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=50)
    fig.suptitle("Mapping with imposed conditions",color=colorformat["text"],fontsize=60)
    # forim,inter_cond = preprocess_variance_map(properties_values,scaler,WcGAN,surrogate_model)
    count = 0
    for i in range(interpolation): #shear
        for j in range(interpolation): #toughness
            axes[j][i].hist2d(forim[count,:,0],forim[count,:,1],bins=20,range=np.array([[inter_cond[0].min(),inter_cond[0].max()],[inter_cond[1].min(),inter_cond[1].max()]]),cmap=cmap_format) #Blues
            axes[j][i].scatter(inter_cond[0][i],inter_cond[1][j],marker = 's',color =colorformat["Highlight"],linewidths = 15)
            axes[j][i].set_title('toughness = {t:.2f}, shear = {s:.2f}'.format(t=inter_cond[0][i],s=inter_cond[1][j]),size=30)
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
    fig.supxlabel(r"Toughness ($\frac{MPa}{m^{1/2}})$",color=colorformat["text"],fontsize=20)
    fig.supylabel("Shear modulus (GPa)",color=colorformat["text"],fontsize=20)
    fig.suptitle("Training data",color=colorformat["text"],fontsize=30)
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



def visilize_accuracy_GAN(real_samples,fake_samples,scaler,surrogate_model,batch_size=520):
    """Use for GAN to visualize the difference between fake and real samples"""
    fake_props = np.zeros((10,batch_size,2))
    for i in range(10):
        fake_props[i] = surrogate_model.predict(torch.from_numpy(fake_samples[i]).float()).detach().numpy()
    fig,axes = plt.subplots(1,10,figsize=(50,5),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    fig.suptitle("WcGAN accuracy",color=colorformat["text"],fontsize=15)
    for i in range(10):
        data = scaler.inverse_transform(real_samples[i])-scaler.inverse_transform(fake_props[i])
        axes[i].hist2d(data[:,0],data[:,1],bins=40,cmap=cmap_format) #Blues
        axes[i].set_xlim(-20,20)
        axes[i].set_ylim(-20,20)
        axes[i].set_xlabel("toughness")
        axes[i].set_ylabel("shear modulus")
        axes[i].set_title("epoch = {}".format(i))
        axes[i].set_facecolor(colorformat["background"])
    return fig


def draw_F_samples(Xt_set,dset,title="default"):
    annote_text = ["A","B","C","D","E","F","G","H","I","J"]
    fig, axes = plt.subplots(2,2,figsize=(15,5),constrained_layout=True)
    fig.set_facecolor(colorformat["background"])
    minor_tick = np.arange(0.5,22.5,1)
    major_tick = np.arange(0,22,1)
    k=0
    for i in range(2):
        for j in range(2):
            im = axes[i][j].imshow(Xt_set[k],cmap = cmap_format,vmin=0,vmax=0.6)
            axes[i][j].set_xticks(major_tick)
            axes[i][j].set_yticks(np.arange(0.5,5.5,1))
            axes[i][j].set_xticklabels(dset.elements)
            axes[i][j].set_yticklabels("")
            axes[i][j].tick_params(axis="both",length=0)
            axes[i][j].set_xticks(minor_tick,minor=True)
            axes[i][j].grid(axis = "x",visible=True,which="minor",c=colorformat["line"],linestyle=(0, (5, 5)))
            axes[i][j].grid(axis = "y",visible=True,which="major",c=colorformat["line"],linewidth=1.2)
            axes[i][j].set_title(annote_text[k])
            k+=1
    fig.suptitle(title,color=colorformat["text"],fontsize=15)
    # fig.supxlabel("Concentration of elements",color=colorformat["text"],fontsize=15)
    plt.colorbar(im,ax=axes.ravel().tolist())
    return fig

def sign_plot(properties_values,input_sites):
    """plot the inputs of data in scatter plot"""
    annote_text = ["A","B","C","D","E","F","G","H","I","J"]
    fig,axes = plt.subplots(nrows=1,ncols=1,constrained_layout=True,figsize=(15,10))
    fig.set_facecolor(colorformat["background"])
    fig.supxlabel("Toughness",color=colorformat["text"],fontsize=20)
    fig.supylabel("Shear modulus",color=colorformat["text"],fontsize=20)
    fig.suptitle("Training data",color=colorformat["text"],fontsize=30)
    axes.scatter(properties_values[:,0],properties_values[:,1],c=colorformat["bar"],linewidths=1)
    for i,input_site in enumerate(input_sites):
        axes.scatter(input_site[0],input_site[1],c=colorformat["Highlight"],linewidths=6)
        axes.annotate(annote_text[i],(input_site[0]+1,input_site[1]),size = 30)
    axes.set_xlim(0,190)
    axes.set_ylim(0,28)
    axes.grid(visible=True)
    axes.set_facecolor(colorformat["background"])
    return fig

def plot_novel(novelity,title="default"):
    fig,axes = plt.subplots(1,1,tight_layout=True)
    fig.set_facecolor(colorformat["background"])
    axes.hist(novelity,color=colorformat['bar'])
    axes.set_xlim(0,1)
    axes.set_ylim(0,50)
    axes.set_xlabel("Cosine Similarity",color=colorformat["text"],fontsize=12)
    axes.set_ylabel("Probility distribution",color=colorformat["text"],fontsize=12)
    fig.suptitle(title,color=colorformat["text"],fontsize=20)
    return fig


def violin_plot_with_MAPE(collected_data,collected_conds,recipes):
    #collected data in shape with (2, 2000, 20)
    #pick shear modulus or toughness
    #Over_value can offer the way to explore the properties out of the range
    #Violin plot (data,objects) (2000,20)

    properties_name=[r"Toughness ($\frac{MPa}{m^{1/2}})$","Shear modulus (GPa)"]
    #calculate mean
    def calculate_mean(outputproperties):
        meandata = np.zeros((2,20))
        for i in range(20):
            meandata[0][i]= outputproperties[0,:,i].mean()
            meandata[1][i]= outputproperties[1,:,i].mean()
        return meandata
    
    def calculate_max(outputproperties):
        maxdata = np.zeros((2,20))
        for i in range(20):
            maxdata[0][i]= outputproperties[0,:,i].max()
            maxdata[1][i]= outputproperties[1,:,i].max()
        return maxdata


    def cal_cos_sim(Recipes):
        cos = pairwise.cosine_similarity(Recipes)
        return cos

    def cal_cos_sim_V(Recipes):
        cos = np.zeros((20))
        for i in range(20):
            cos[i] = cal_cos_sim(Recipes[i]).mean()
        return cos
        
    def cal_mape(y_true,y_pred):
        y_true_collection = np.zeros((2,2000,20))
        mape_collection = np.zeros((2,20))
        for i in range(2):
            for j in range(20):
                # y_true_collection[i,:,j] = np.repeat(y_true[i][j],2000)
                mape_collection[i,j] = mean_absolute_percentage_error(np.repeat(y_true[i][j],2000),y_pred[i,:,j])
        return mape_collection
        
    def collect_table(collected_data,collected_conds,recipes):
        rows_labels = ['input','avg','std']
        data = np.empty((2,5,20),dtype="<U21")
        col_name = "ABCDEFGHIJKLMNOPQRST"
        mape = cal_mape(collected_conds,collected_data)
        for i in range(2):
            data[i,0,:] = np.around(collected_conds[i],1)
            data[i,3,:] = mape[i]*100
            for j in range(20):
                data[i,1,j] = np.format_float_positional(collected_data[i,:,j].mean(),1)
                data[i,2,j] = np.format_float_positional(collected_data[i,:,j].std(),1)
                data[i,4,j] = cal_cos_sim(recipes[j,:,:]).mean()*100
        for i in range(2):
            for j in range(20):
                data[i,3,j] = data[i,3,j][0:4]+"%"
                data[i,4,j] = data[i,4,j][0:4]+"%"
        table_content ={
            "colLabels" : [i for i in "ABCDEFGHIJKLMNOPQRST"],
            "rowLabels" : ['input','gen_avg','std','MAPE','cos_sim'],
            "cellText" : data
        }
        return table_content

    mean_data = calculate_mean(collected_data)
    max_data = calculate_max(collected_data)
    mape = cal_mape(collected_conds,collected_data)
    COS_SIM = cal_cos_sim_V(recipes) 

    table_info = collect_table(collected_data,collected_conds,recipes)
    #Plotting violin plot
    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(40,20),constrained_layout=True)
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

    

        # axes[i].grid(visible=True,linewidth=2)
        axes[i].set_facecolor(colorformat["background"])
        axes[i].set_xlabel(properties_name[i],fontsize=15)
        axes[i].set_ylabel("generated {}".format(properties_name[i]),fontsize=15)
        ln1 = axes[i].plot(collected_conds[i],mean_data[i],c=colorformat["Highlight"],linewidth=2,label="generated avg",marker="^",markersize=12)
        ln2 = axes[i].plot(collected_conds[i],collected_conds[i],"--",c=colorformat["Highlight"],linewidth=2,label="expected avg",marker="X",markersize=12)
        for k in range(20):
            axes[i].annotate(table_info["colLabels"][k],(collected_conds[i][k],1+max_data[i][k]),size=15)

        axes[i].set_yticks(mean_data[i],np.around(mean_data[i],2),size=10)
        axes[i].set_xticks(collected_conds[i],np.around(mean_data[i],2),size=10)

        twin_axis = axes[i].twinx()
        ln3 = twin_axis.plot(collected_conds[i],mape[i],color=colorformat["text"],alpha=1,linewidth=2,label="MAPE",marker="^",markersize=15)
        ln4 = twin_axis.plot(collected_conds[i],COS_SIM,color=colorformat["text"],alpha=1,ls="--",linewidth=2,label="cos_sim",marker="X",markersize=20)
        twin_axis.set_ylim(0,1)
        twin_axis.set_ylabel("MAPE & Cosine Similarity",fontsize=15)
        tw_str = np.array([])
        for j in enumerate(mape[i]):
            tw_str = np.append(tw_str,format(j[1],".1%"))
        # twin_axis.set_yticks(mape[i],tw_str,size=10)
        lns = ln1+ln2+ln3+ln4
        labs = [l.get_label() for l in lns]
        axes[i].legend(lns, labs, loc=2,fontsize=15)

        table = axes[i].table(
        colLabels = table_info["colLabels"],
        rowLabels = table_info["rowLabels"],
        cellText = table_info["cellText"][i],
        loc="top",
        cellLoc="center",
        )
        table.set(fontsize=18)
        table.scale(1,2)

    return fig