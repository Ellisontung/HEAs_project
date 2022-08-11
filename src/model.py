import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm.notebook import tqdm
import os


# ANN (surrogate model)
class artificial_neural_network(torch.nn.Module):
    """Artiricial neural networks surrogate model"""
    def __init__(self,nx,ny):
        super().__init__()
        self.fc1 = nn.Linear(nx,15)    # linear layer
        self.fc2 = nn.Linear(15,8)    # linear layer
        self.fc3 = nn.Linear(8,4)    # linear layer
        self.fc4 = nn.Linear(4,ny)    # linear layer
        self.act = torch.nn.ReLU()  # activation function

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


class ANN():
    def __init__(self,nx=22,ny=2):
        
        self.model = artificial_neural_network(nx,ny)

        self.optimizer = torch.optim.Adam(params = self.model.parameters())
        self.Loss = torch.nn.MSELoss()

    def train(self,recipes, conditions,epoches = 2000):
        self.recipes = torch.from_numpy(recipes)
        self.conditions = torch.from_numpy(conditions)
        for epoch in tqdm(range(epoches)):
            self.optimizer.zero_grad()
            y_predit = self.model(self.recipes)
            loss = self.Loss(y_predit,self.conditions)
            if((epoch+1)%100==0):
                print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
            loss.backward()
            self.optimizer.step()
        print("******Training ANN completed*****")
    
    def predict(self,X):
        return self.model(X)
    
    def save_dict(self , save_dir = 'Trained_models/ANN.pt'):
        torch.save({
            'ANN_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        },save_dir)
    
    def load_dict(self , load_dir = 'Trained_models/ANN.pt'):
        try:
            open(load_dir)
        except:
            raise "No Trained ANN in the directory"
        else:
            checkpoint = torch.load(load_dir)
            self.model.load_state_dict(checkpoint['ANN_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])











# Conditional Autoencoder
class conditional_autoencoder(torch.nn.Module):
    """Conditional autoencdoer structure"""
    def __init__(self,nx,nz):
        super().__init__()

        self.encoder=torch.nn.Sequential(
            torch.nn.Linear(nx+2,64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,nz),
        )

        self.decoder=torch.nn.Sequential(
            torch.nn.Linear(nz+2,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,nx),
        )

    def encode(self,x,conditions):
        x = torch.cat((x,conditions),dim=1)
        return self.encoder(x)

    def decode(self,x,conditions):
        x = torch.cat((x,conditions),dim=1)
        return self.decoder(x)
    
    def forward(self,x,conditions):
        x=self.encode(x,conditions)
        x=self.decode(x,conditions)
        return x

class cAE():
    def __init__(self,nx=22,nz=5):
        self.model = conditional_autoencoder(nx,nz)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.Loss=torch.nn.MSELoss()
    
    def train(self,recipes,conditions,epoches=1000):
        self.recipes = torch.from_numpy(recipes)
        self.conditions = torch.from_numpy(conditions)
        for epoch in tqdm(range(epoches)):
            self.optimizer.zero_grad()
            X_predict = self.model(self.Xt,self.conditions)
            loss=self.Loss(X_predict,self.Xt)
            if((epoch+1)%100==0):
                print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
            loss.backward()
            self.optimizer.step()
        print("*****Training cAE completed*****")
    
    def decode(self,X,Conds):
        return self.model.decode(X,Conds)

    def save_dict(self , save_dir = 'Trained_models/cAE.pt'):
        torch.save({
            'cAE_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        },save_dir)
    
    def load_dict(self , load_dir = 'Trained_models/cAE.pt'):
        try:
            open(load_dir)
        except:
            raise "No Trained ANN in the directory"
        else:
            checkpoint = torch.load(load_dir)
            self.model.load_state_dict(checkpoint['cAE_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])













#Conditional variational autoencoder
class conditional_variational_autoencoder(torch.nn.Module):
    """Conditional variational autoencdoer structure"""
    def __init__(self,nx,nz):
        super().__init__()


        self.fc1 = torch.nn.Linear(nx+2,64)
        self.fc2 = torch.nn.Linear(64,32)
        self.fc3 = torch.nn.Linear(32,16)
        self.fc41 = torch.nn.Linear(16,5)
        self.fc42 = torch.nn.Linear(16,5)


        self.fc5 = torch.nn.Linear(nz+2,16)
        self.fc6 = torch.nn.Linear(16,32)
        self.fc7 = torch.nn.Linear(32,64)
        self.fc8 = torch.nn.Linear(64,nx)

        self.act_fun = torch.nn.LeakyReLU()
        self.Loss = torch.nn.MSELoss()

    def encode(self,x,conditions):
        x = torch.cat((x,conditions),dim=1)
        x = self.act_fun(self.fc1(x))
        x = self.act_fun(self.fc2(x))
        x = self.act_fun(self.fc3(x))
        z_mu = self.fc41(x)
        z_var = self.fc42(x)
        return z_mu,z_var

    def decode(self,z,conditions):
        x = torch.cat((z,conditions),dim=1)
        x = self.act_fun(self.fc5(x))
        x = self.act_fun(self.fc6(x))
        x = self.act_fun(self.fc7(x))
        x = self.fc8(x)
        return x

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std

    
    def forward(self,x,conditions):
        mu,var=self.encode(x,conditions)
        z = self.reparameterize(mu,var)
        x=self.decode(z,conditions)
        return x, mu, var

    def loss_function(self,recon_x, x, mu, var):
        MSE = self.Loss(recon_x,x)
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        return MSE + KLD

class cVAE():
    def __init__(self,nx=22,nz=5):
        
        self.model = conditional_variational_autoencoder(nx,nz)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)
    
    def train(self,recipes,conditions,epoches=1000):
        self.Xt=torch.from_numpy(recipes)
        self.conditions = torch.from_numpy(conditions)
        for epoch in tqdm(range(epoches)):
            self.optimizer.zero_grad()
            X_predict,mu,var = self.model(self.Xt,self.conditions)
            loss = self.model.loss_function(X_predict,self.Xt,mu,var)
            if((epoch+1)%100==0):
                print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
            loss.backward()
            self.optimizer.step()
        print("*****Training cVAE completed*****")
    
    def decode(self,X,Conds):
        return self.model.decode(X,Conds)

    def save_dict(self , save_dir = 'Trained_models/cVAE.pt'):
        torch.save({
            'cVAE_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        },save_dir)
    
    def load_dict(self , load_dir = 'Trained_models/cVAE.pt'):
        try:
            open(load_dir)
        except:
            raise "No Trained ANN in the directory"
        else:
            checkpoint = torch.load(load_dir)
            self.model.load_state_dict(checkpoint['cVAE_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])









#GAN
class Generator(nn.Module):

    def __init__(self, latent_dim, input_length, hidden_size):
        super(Generator, self).__init__()
        self.dense_layer1 = nn.Linear(int(latent_dim), int(hidden_size))
        self.dense_layer2 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer3 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer4 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer5 = nn.Linear(int(hidden_size), int(input_length))
        self.activation = nn.LeakyReLU()

    def forward(self, x,property):
        x = torch.cat([x,property],1)
        x = self.activation(self.dense_layer1(x))
        x = self.activation(self.dense_layer2(x))
        x = self.activation(self.dense_layer3(x))
        x = self.activation(self.dense_layer4(x))
        x = self.dense_layer5(x)
        return torch.nn.Softmax(dim=-1)(x)

class Discriminator(nn.Module):
    def __init__(self, input_length, hidden_size):
        super(Discriminator, self).__init__()
        self.dense_layer1 = nn.Linear(int(input_length), int(hidden_size))
        self.dense_layer2 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer3 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer4 = nn.Linear(int(hidden_size), int(hidden_size))
        self.dense_layer5 = nn.Linear(int(hidden_size), 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x, property):
        x = torch.cat([x,property],1)
        x = self.activation(self.dense_layer1(x))
        x = self.activation(self.dense_layer2(x))
        x = self.activation(self.dense_layer3(x))
        x = self.activation(self.dense_layer4(x))
        x = self.dense_layer5(x)
        return x

class WcGAN():
    def __init__(self,nx=22,ny=2,nz=10):

        self.generator = Generator(nz+ny,nx,nx*2) #fill in parameters 12 22 44
        self.discriminator = Discriminator(nx+ny,nx*2)
        beta = (0.5,0.99)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4,betas=beta)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4,betas=beta)
        

        self.epoches = 0
        self.cuda = False


        
    def check_cuda(self):
        if torch.cuda.is_available():
            self.cuda = True
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        else:
            self.cuda = False


    def train(self,dset,properties_values_scaled,save_dir="test_save/WcGAN.pt",num_iterations=int(3e4),batch_size = 520):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(properties_values_scaled)
        print("Target epoches: {}".format(self.epoches+num_iterations))
        self.dset = dset
        self.properties_values_scaled = properties_values_scaled
        sample_time = 0
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.log_interval = self.num_iterations/10
        #store the real samples and fake samples under same conditions
        self.samples_real = np.zeros((10,self.batch_size,2))
        self.samples_fake = np.zeros((10,self.batch_size,22))
        for it in tqdm(range(self.num_iterations)):

            self.d_loop()
            self.g_loop()
            self.epoches+=1

            if (it + 1) % self.log_interval == 0:
                g_fake_data,true_data = self.g_sample()
                self.samples_real[sample_time] = true_data
                self.samples_fake[sample_time] = g_fake_data
                sample_time+=1
        self.save_dict(save_dir)
    

    def d_loop(self):
        self.discriminator_optimizer.zero_grad()

        train_data, train_idx = self.dset.sample(self.batch_size)
        train_prop = self.properties_values_scaled[train_idx]
        d_real_data = torch.from_numpy(train_data)
        d_real_prop = torch.from_numpy(train_prop)
        if self.cuda:
            d_real_data = d_real_data.cuda()
            d_real_prop = d_real_prop.cuda()

        true_discriminator_out = self.discriminator(d_real_data,d_real_prop)
        d_gen_input = torch.from_numpy(self.noise_sampler(self.batch_size, 10)) #z=10
        d_gen_prop = self.kde.sample(self.batch_size).astype('float32')
        d_gen_prop =  torch.from_numpy(d_gen_prop)#sample between lowest and highest LC values, uniform 

        if self.cuda:
            d_gen_input = d_gen_input.cuda()
            d_gen_prop = d_gen_prop.cuda()
        with torch.no_grad():
            generated_data = self.generator(d_gen_input,d_gen_prop)

        generator_discriminator_out = self.discriminator(generated_data.detach(),d_gen_prop)

        discriminator_loss = (torch.mean(generator_discriminator_out) - torch.mean(true_discriminator_out))
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        for p in self.discriminator.parameters():
            p.data.clamp_(-0.5, 0.5)
            
    def g_loop(self):
        self.generator_optimizer.zero_grad()

        d_gen_input = torch.from_numpy(self.noise_sampler(self.batch_size, 10)) # z=10
        d_gen_prop = self.kde.sample(self.batch_size).astype('float32')
        d_gen_prop =  torch.from_numpy(d_gen_prop)#sample between lowest and highest LC values, uniform 
        if self.cuda:
            d_gen_input = d_gen_input.cuda()
            d_gen_prop = d_gen_prop.cuda()
        generated_data = self.generator(d_gen_input,d_gen_prop)
        generator_discriminator_out = self.discriminator(generated_data,d_gen_prop)
        generator_loss =  -torch.mean(generator_discriminator_out)
        generator_loss.backward()
        self.generator_optimizer.step()
    
    def noise_sampler(self,N, z_dim):
        return np.random.normal(size=[N, z_dim]).astype('float32')

    def g_sample(self):
        with torch.no_grad():
            gen_input = torch.from_numpy(self.noise_sampler(self.batch_size, 10))
            prop_input = self.kde.sample(self.batch_size).astype('float32')
            prop_input = torch.from_numpy(prop_input)
            g_fake_data = self.generator(gen_input,prop_input)
            return g_fake_data.detach().numpy(),prop_input

    def save_dict(self,save_dir="test_save/WcGAN.pt"):
        torch.save({
            'epoch': self.epoches,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            }, save_dir)

    def load_dict(self,load_dir="test_save/WcGAN.pt"):
        try:
            open(load_dir)
        except:
            print("No file in the directory")
        else:
            checkpoint = torch.load(load_dir)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
            self.epoches = checkpoint["epoch"]