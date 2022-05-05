import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm

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

class conditional_variational_autoencoder(torch.nn.Module):
    """Conditional variational autoencdoer structure"""
    def __init__(self,nx,nz):
        super().__init__()

        self.encoder_mu=torch.nn.Sequential(
            torch.nn.Linear(nx+2,64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,nz),
        )

        self.encoder_var=torch.nn.Sequential(
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
        self.Loss = torch.nn.MSELoss()

    def encode(self,x,conditions):
        x = torch.cat((x,conditions),dim=1)
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu,z_var

    def decode(self,z,conditions):
        x = torch.cat((z,conditions),dim=1)
        return self.decoder(x)

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