import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm.notebook import tqdm


# ANN (surrogate model)
class artificial_neural_network(torch.nn.Module):
    """Artiricial neural networks surrogate model"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(22,15)    # linear layer
        self.fc2 = nn.Linear(15,8)    # linear layer
        self.fc3 = nn.Linear(8,4)    # linear layer
        self.fc4 = nn.Linear(4,2)    # linear layer
        self.act = torch.nn.ReLU()  # activation function

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


def training_surrogate_ANN(sequences_input,conditions_input,surrogatemodel=artificial_neural_network,epoches_ANN=2000):
    Conditionst_ys = torch.from_numpy(conditions_input)
    surr = surrogatemodel()
    Xt=torch.Tensor(sequences_input)
    optimizer=torch.optim.Adam(params = surr.parameters())
    Loss=torch.nn.MSELoss()
    for epoch in tqdm(range(epoches_ANN)):
        optimizer.zero_grad()
        y_predit = surr(Xt)
        loss=Loss(y_predit,Conditionst_ys)
        if((epoch+1)%100==0):
            print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
        loss.backward()
        optimizer.step()
    print("******Training ANN completed*****")
    return surr


def training_surrogate_RF(sequences_input,conditions_input):
    from sklearn.ensemble import RandomForestRegressor
    surr = RandomForestRegressor(random_state=0)
    surr.fit(sequences_input,conditions_input)
    print("*****Training RF completed*****")
    return surr











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


def training_cAE(sequences_input,conditions_input,cAE=conditional_autoencoder,epoches_cAE=1000):
    Xt=torch.Tensor(sequences_input)
    Conditionst_ys = torch.from_numpy(conditions_input)
    model_cAE=cAE(Xt.shape[1],5)
    optimizer=torch.optim.Adam(model_cAE.parameters(),lr=1e-3)
    Loss=torch.nn.MSELoss()
    for epoch in tqdm(range(epoches_cAE)):
        optimizer.zero_grad()
        X_predict = model_cAE(Xt,Conditionst_ys)
        loss=Loss(X_predict,Xt)
        if((epoch+1)%100==0):
            print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
        loss.backward()
        optimizer.step()
    print("*****Training cAE completed*****")
    return model_cAE















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
        #compare BCE and MSE
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        return MSE + KLD


def training_cVAE(sequences_input,conditions_input,cAE=conditional_variational_autoencoder,epoches_cAE=1000):
    Xt=torch.Tensor(sequences_input)
    Conditionst_ys = torch.from_numpy(conditions_input)
    model_cVAE=cAE(Xt.shape[1],5)
    optimizer=torch.optim.Adam(model_cVAE.parameters(),lr=1e-3)
    for epoch in tqdm(range(epoches_cAE)):
        optimizer.zero_grad()
        X_predict,mu,var = model_cVAE(Xt,Conditionst_ys)
        loss=model_cVAE.loss_function(X_predict,Xt,mu,var)
        if((epoch+1)%100==0):
            print('epoch: {}, Loss:{}'.format(epoch+1 ,loss))
        loss.backward()
        optimizer.step()
    print("*****Training cAE completed*****")
    return model_cVAE













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
    def __init__(self,dset,properties_values_scaled,surrogate_model):
        self.batch_size = 520
        self.minibatch_size = self.batch_size
        # self.training_steps = 5000
        self.num_iterations = int(3e4)
        self.log_interval = self.num_iterations/10

        self.dset = dset
        self.properties_values_scaled = properties_values_scaled
        self.surrogate_model = surrogate_model
        # Models
        z = 10
        conditions = 2
        g_inp =  z+conditions
        input_length, latent_dim, hidden_size = (22, g_inp, g_inp*2)
        self.generator = Generator(latent_dim, input_length, hidden_size) #(12,22,44)
        self.discriminator = Discriminator(input_length+conditions, hidden_size) #(24,44)

        # Optimizers
        beta = (0.5,0.99)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4,betas=beta)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4,betas=beta) #RMSprop

        self.d_steps = 1
        self.g_steps = 1


        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.properties_values_scaled)

        self.samples_real = np.zeros((10,self.minibatch_size,2))
        self.samples_fake = np.zeros((10,self.minibatch_size,2))

    def noise_sampler(self,N, z_dim):
        return np.random.normal(size=[N, z_dim]).astype('float32')

    def train(self):

        sample_time = 0
        for it in tqdm(range(self.num_iterations)):
            # d_infos = []
            for d_index in range(self.d_steps):
                self.d_loop()
                #d_infos.append(d_info)
            #d_infos = np.mean(d_infos, 0)
            #d_real_loss, d_fake_loss = d_infos
            
            #g_infos = []
            for g_index in range(self.g_steps):
                self.g_loop()
                #g_infos.append(g_info)
            #g_infos = np.mean(g_infos)
            #g_loss = g_infos
            
            if (it + 1) % self.log_interval == 0:
                g_fake_data,true_data = self.g_sample()
                self.samples_real[sample_time] = true_data
                self.samples_fake[sample_time] = self.surrogate_model(torch.tensor(g_fake_data)).detach().numpy()
                sample_time+=1

    
        
    def d_loop(self):
        self.discriminator_optimizer.zero_grad()

        train_data, train_idx = self.dset.sample(self.batch_size)
        train_prop = self.properties_values_scaled[train_idx]
        d_real_data = torch.from_numpy(train_data)
        d_real_prop = torch.from_numpy(train_prop)

        true_discriminator_out = self.discriminator(d_real_data,d_real_prop)
        d_gen_input = torch.from_numpy(self.noise_sampler(self.batch_size, 10)) #z=10
        d_gen_prop = self.kde.sample(self.batch_size).astype('float32')
        d_gen_prop =  torch.from_numpy(d_gen_prop)#sample between lowest and highest LC values, uniform 

        generated_data = self.generator(d_gen_input,d_gen_prop)
        # generated_data = self.generator(d_gen_input,d_real_prop)

        generator_discriminator_out = self.discriminator(generated_data.detach(),d_gen_prop)
        # generator_discriminator_out = self.discriminator(generated_data.detach(),d_real_prop)

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
        generated_data = self.generator(d_gen_input,d_gen_prop)
        generator_discriminator_out = self.discriminator(generated_data,d_gen_prop)
        generator_loss =  -torch.mean(generator_discriminator_out)
        generator_loss.backward()
        self.generator_optimizer.step()
    
    def g_sample(self):
        with torch.no_grad():
            gen_input = torch.from_numpy(self.noise_sampler(self.minibatch_size, 10))
            prop_input = self.kde.sample(self.minibatch_size).astype('float32')
            prop_input = torch.from_numpy(prop_input)
            g_fake_data = self.generator(gen_input,prop_input)
            return g_fake_data.detach().numpy(),prop_input