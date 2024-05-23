import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import anndata
import torch.optim as optim
import scanpy as sc
from sklearn.model_selection import train_test_split
from torchMBM.modules import BKLLoss
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from .modules import *
from .utils import *
from .functional import *
from kneed import KneeLocator
import os


class AttentionWide(nn.Module):
    def __init__(self, emb, p = 0.2, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask
        self.dropout = nn.Dropout(p)
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y):
        b = 1
        t, e = x.size()
        h = self.heads
        # assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        self.attention_weights = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)




class MBM(nn.Module):
    """
    MBM Model: latent feature extraction and spatial domain deciphering with a prior-based self-attention framework for spatial transcriptomics
    Parameters
    ------
    d_in
        number of feature of input gene expression matrix
    d_lat
        dimensions of latent feature
    save_path
        directory of saved files
    """

    def __init__(self, d_in, d_lat, ct_key, save_path):
        super(MBM,self).__init__() 
        self.d_prior = d_lat // 5
        self.fc1_MBM = BayesLinear(d_in, self.d_prior)
        self.fc1_NN = nn.Linear(d_in, d_lat - self.d_prior, bias=False)
        self.attn = AttentionWide(d_lat)
        self.fc2= nn.Linear(d_lat,64,bias=False)
        self.fc3= nn.Linear(64,2,bias=False)
        self.save_path = save_path
        self.ct_key = ct_key
        
    def prior_initialize(self, prior):
        if not isinstance(prior, torch.FloatTensor):
            prior = torch.FloatTensor(prior)
        assert prior.shape[0] == self.d_prior, "prior weight dimension not match"

        prior_log_sigma = torch.log(prior.std() / 10)
        self.fc1_MBM.reset_parameters(prior, prior_log_sigma)

    def MBM_loss(self):
        return self.fc1_MBM.bayesian_kld_loss()
    
    def forward(self, x):
        x = x.contiguous().view(x.shape[0],-1)    # Flatten the images
        x = torch.cat([self.fc1_MBM(x),self.fc1_NN(x)], -1)
        x = F.relu(x)
        x= (self.attn(x, x)).squeeze(0)+x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def optim_parameters(net, included=None, excluded=None):
        def belongs_to(cur_layer, layers):
            for layer in layers:
                if layer in cur_layer:
                    return True
            return False

        params = []
        if included is not None:
            if not isinstance(included, list):
                included = [included]
            for cur_layer, param in net.named_parameters():
                if belongs_to(cur_layer, included) and param.requires_grad:
                    params.append(param)
        else:
            if not isinstance(excluded, list):
                excluded = [excluded] if excluded is not None else []
            for cur_layer, param in net.named_parameters():
                if not belongs_to(cur_layer, excluded) and param.requires_grad:
                    params.append(param)
        return iter(params)
 
    def model_train(self, tdata, rdata=None, add_noise = True, delta = 0.1, split_ratio = 0.25, sigma = 0.5, epochs = 100, lr = 1e-4,  weight_decay = 1e-4, batch_size = 64, epoch_thres = 8, device = torch.device("cuda"), S = 1):
        s = time.time()
        
        def optim_parameters(net, included=None, excluded=None):
            def belongs_to(cur_layer, layers):
                for layer in layers:
                    if layer in cur_layer:
                        return True
                return False
            params = []
            if included is not None:
                if not isinstance(included, list):
                    included = [included]
                for cur_layer, param in net.named_parameters():
                    if belongs_to(cur_layer, included) and param.requires_grad:
                        params.append(param)
            else:
                if not isinstance(excluded, list):
                    excluded = [excluded] if excluded is not None else []
                for cur_layer, param in net.named_parameters():
                    if not belongs_to(cur_layer, excluded) and param.requires_grad:
                        params.append(param)
            return iter(params)
        
        #put data into datalodar
        tdata_X = tdata.X.copy()
        tdata_y = tdata.obs[self.ct_key]
        tdata_y = [0 if i =='Normal' else i for i in tdata_y]
        tdata_y = [1 if i =='Tumor' else i for i in tdata_y]
        tdata_X=tdata_X.A
        tdata_X =np.array(tdata_X )
        #add Gauss Noise

        
        #split the dataset into train set and valid set
        X_train, X_var, y_train, y_var = train_test_split(tdata_X,
                                          tdata_y,
                                          test_size=split_ratio,
                                          random_state=42)

        X_train, y_train = torch.as_tensor(X_train).float(), torch.as_tensor(y_train).float()
        X_var, y_var = torch.as_tensor(X_var).float(), torch.as_tensor(y_var).float()

        #把他们纳入datalodaer
        ds_train = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        ds_var = torch.utils.data.TensorDataset(X_var, y_var)
        validloader = torch.utils.data.DataLoader(ds_var, batch_size=batch_size, shuffle=True)
        
        if rdata is None:
            sc.tl.pca(tdata, n_comps=self.d_prior)
            prior_weight = torch.FloatTensor(tdata.varm["PCs"].T.copy())
            
            self.prior_initialize(prior_weight)
            print(prior_weight.shape)
        else:
            if not isinstance(rdata, sc.AnnData):
                rdata = sc.AnnData(rdata)
            sc.tl.pca(rdata, n_comps=self.d_prior)
            prior_weight = torch.FloatTensor(rdata.varm["PCs"].T.copy())
                           
            self.prior_initialize(prior_weight)
       
        self.to(device)
        min_valid_loss = np.inf
        ce_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(optim_parameters(self), lr = lr, weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min')

        trainloss = list()
        validloss = list()
        last_loss = 1e6
        num = 0

        for epoch in range(epochs):

            train_loss = 0.0
            train_count = 0
            self.train()
            for data, labels in trainloader:

                if torch.cuda.is_available():
                    data, labels = data.to(device), labels.to(device)
                    if add_noise:
                        data = data.cpu()
                        data = data.numpy()
                        data = data + np.random.normal(0,sigma,size = data.shape)
                        data = torch.FloatTensor(data)
                        data = data.to(device)
                target = self.forward(data)
                ce = ce_loss(target, labels.long())
                MBM = self.MBM_loss()
                loss = delta * MBM + ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_count = train_count + 1


            valid_loss = 0.0
            valid_count = 0
            self.eval()    
            for data, labels in validloader:

                if torch.cuda.is_available():
                    data, labels = data.to(device), labels.to(device)

                target = self.forward(data)
                ce = ce_loss(target, labels.long())
                MBM = self.MBM_loss()
                loss = delta * MBM + ce

                valid_loss +=  loss.item()
                valid_count = valid_count + 1

            Training_Loss = train_loss / train_count
            Validation_Loss = valid_loss / valid_count

            trainloss.append(Training_Loss)
            validloss.append(Validation_Loss)

            e = time.time()
            print(f'Epoch {epoch + 1} \t Training Loss: {Training_Loss} \t Validation Loss: {Validation_Loss} \t Time: {e-s}')
            s = time.time()
            
            if not os.path.exists(self.save_path+'model_pth'):
                os.makedirs(self.save_path+'model_pth') 
            torch.save(self.state_dict(), self.save_path+'model_pth/saved_model_'+str(epoch)+'.pth')
            num = num + 1

            if abs(valid_loss/valid_count-last_loss) < 5e-5 or valid_loss/valid_count-last_loss > 1e-3:
                print("Model Converge")
                break

            last_loss = valid_loss / valid_count
            scheduler.step(last_loss)
            '''
            if min_valid_loss > Validation_Loss:
                print(f'Validation Loss Decreased {min_valid_loss}')
                print(f'{Validation_Loss:.6f}\t Saving The Model')
                min_valid_loss = Validation_Loss
            '''

        x = list(range(1, len(trainloss)+1))
        plt.rcParams.update({'font.size':18}) 
        plt.figure(figsize=(10, 8))
        plt.plot(x, trainloss, label = "TrainLoss")
        plt.plot(x, validloss, label = 'ValidLoss')
        plt.xlabel("number of epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()

        kl = KneeLocator(x, validloss, curve = 'convex', direction = "decreasing")
     #   kl.plot_knee()
        print(kl.knee)
        plt.show()

        return last_loss, kl.knee, num-1


    def score(self,test_subset,target,out_path = False):
        x_test = test_subset.X.copy()
        y_test = test_subset.obs[self.ct_key]
        
        y_test = [0 if i =='Normal' else i for i in y_test]
        y_test = [1 if i =='Tumor' else i for i in y_test]
        x_test=x_test.A
        X_test =np.array(x_test )
        X_test, y_test = torch.as_tensor(X_test).float(), torch.as_tensor(y_test).float()
        
        
        self=self.cpu()
        pre1 = self(X_test)#放入要测试的数据
        _, predicted1 = torch.max(pre1.data, 1)
        predicted1 =predicted1.numpy()
        Frequency = predicted1
        for i in range(10):
            _, predicted1 = torch.max(pre1.data, 1)
            predicted1 =predicted1.numpy()
            Frequency = np.vstack((Frequency,predicted1))
        vote_pre = np.array([])
        for i in range(Frequency.shape[1]):
            vote_pre = np.append(vote_pre,np.argmax(np.bincount(Frequency[:,i])))
        #这里要注意把数据都换回到cpu上
        AUPRC = average_precision_score(y_test.cpu(),vote_pre)
        bas = balanced_accuracy_score(y_test.cpu(), vote_pre)
        AUROC = roc_auc_score(y_test.cpu(),  vote_pre)
        kappa = cohen_kappa_score(y_test.cpu(),  vote_pre)
        F1_score = f1_score(y_test.cpu(),  vote_pre)
#        target[test_name] = [pred, bas,AUROC,AUPRC, kappa, F1_score]
#        if out_path =True:
#            result = pd.DataFrame(target, index = ["pred", "bas", "AUROC", "AUPRC", "kappa", "f1"])
#            result.to_csv(out_path, sep=",")
#        else:
#            pass
        print('- AUPRC: %f ' % (AUPRC))
        print('- bas: %f ' % (bas))
        print('- AUROC : %f' % (AUROC))
        print('- kappa : %f' % (kappa))
        print('- F1_score : %f' % (F1_score))
        
    