from utils import *
import ikarus

def scCrab(adatas,train_name,test_name,reference_name, save_path, device, main_obs='major_ct', detail_obs='sub_ct', batch_size=256,epochs=100,lr=1e-5,sigma = 1, delta=1, split_ratio=0.25):
    """
    scCrab: a reference-guided cancer cell identification integrated method based on Bayesian neural networks
    
    Parameters
    ----------
    adatas
        Anndata datasets that contain training dataset, testing dataset, and reference dataset.
    train_name
        The name of training dataset in adatas
    reference_name
        The name of reference dataset in adatas
    test_name
        The name of test dataset in adatas
    device
        The hardware on which the code is executed, cpu or cuda. We prefer to use cuda
    save_path
        The path where the prediction results and evaluation metrics are saved
    main_obs
        The cell type annotation used for classification, only contain Tumor and Normal
    detail_obs
        A more detailed cell type annotation used for generating gene list
    batch_size
        The number of data samples processed at once during training in the neural network
    epochs
        The number of times the entire training dataset is passed forward and backward through the neural network 
    lr
        Learning rate
    sigma
        The variance of Gaussian noise
    delta
        The weight of the KL divergence in the loss function
    split_ratio 
        The proportion in which a dataset is divided into training and validation
    
    Returns
    -------
    pred_bik
        The prediction of whether a cell is a tumor cell or not
    prob_bik
        The probability of a cell being tumorous.
   
    """
    #MBM part
    target = {}
    train_data, reference_data, test_data = get_3selected_dataset(adatas, train_name,reference_name, test_name)
    vote = np.zeros(test_data.X.shape[0])
    vote_of_1 = np.zeros(test_data.X.shape[0])
    BMM_model = BMM(d_in = train_data.shape[1], d_lat = 512).to(device)
    loss, vote_time, num = BMM_model.model_train(tdata = train_data, rdata = reference_data, batch_size = batch_size
                              , epochs = epochs, lr =  lr ,sigma = sigma, split_ratio = split_ratio
                              , delta =  delta , device = device) 

    x_test = test_data.X.copy()
    y_test = test_data.obs[main_obs]

    y_test = [0 if i =='Normal' else i for i in y_test]
    y_test = [1 if i =='Tumor' else i for i in y_test]
    x_test = x_test.A
    X_test = np.array(x_test)
    X_test, y_test = torch.as_tensor(X_test).float(), torch.as_tensor(y_test).float()

    vote_range = range(vote_time, num)
    for times in vote_range:
        BMM_model = BMM(d_in = train_data.shape[1], d_lat = 512).to(device)
        BMM_model.load_state_dict(torch.load('model_pth/saved_model_'+ str(times) +'.pth'))
        BMM_model = BMM_model.cpu()
        chunk_size = 256
        chunks = []
        for i in range(0, len(X_test), chunk_size): 
            chunks.append(BMM_model(X_test[i:i+chunk_size]))
        
        # Concatenate the lines extracted each time.
        pre1 = torch.cat(chunks)
        pre1_prob = torch.nn.functional.softmax(pre1, dim=1)
        prob_of_1 = pre1_prob[:, 1].detach().numpy()
        _, predicted1 = torch.max(pre1_prob.data, 1)


        vote_of_1=vote_of_1+np.array(prob_of_1)
        vote = vote + np.array(predicted1)


    vote_pre = (vote > (vote_time+1)/2).astype(int)
    if vote_time != 0:
            vote_of_1 = vote_of_1/vote_time 
    #AUPRC = average_precision_score(y_test, vote_of_1)
    #bas = balanced_accuracy_score(y_test, vote_pre)
    #kappa = cohen_kappa_score(y_test, vote_pre)
    #target[test_name] = [vote_pre, vote_of_1, bas, AUROC, AUPRC, kappa, F1_score]
    #save_obj(target,save_path+train_name+".pkl")
    #print('- AUPRC: %f ' % (AUPRC))
    #print('- bas: %f ' % (bas))
    #print('- kappa : %f' % (kappa))
    #return vote_pre, AUPRC, bas, kappa
    prob_BMM = vote_of_1
    
    #ikarus part
    get_genelist(adatas, traindata_name=train_name, save_path=save_path, obs_name=detail_obs)
    traindata = adatas[train_name]
    testdata = adatas[test_name]
    model = classifier.Ikarus(signatures_gmt=save_path+train_name+'/signatures.gmt', out_dir=save_path+train_name, adapt_signatures=True)
    model.fit([traindata], [train_name], [main_obs], save=True)
    pred = model.predict(testdata, test_name, save=True)
    prob_ikarus = pd.read_csv(save_path+train_name+'/'+test_name+'/prediction.csv')['final_pred_proba_Tumor'].tolist()
    
    #ensemble
    #reals = list(testdata.obs[main_obs])
    #reals = [0 if i =='Normal' else i for i in reals]
    #reals = [1 if i =='Tumor' else i for i in reals]
    prob_bik, pred_bik = [],[]
    for i in range(testdata.shape[0]):
        x = np.mean([prob_BMM[i],prob_ikarus[i]])BMM
        prob_bik.append(x)
        if x>0.5:
            pred_bik.append(1)
        else:
            pred_bik.append(0)
    
    return pred_bik, prob_bik

def evaluate_metrics(adatas, test_name, main_obs, pred, prob):
    y_test = adatas[test_name].obs[main_obs]
    y_test = [0 if i =='Normal' else i for i in y_test]
    y_test = [1 if i =='Tumor' else i for i in y_test]
    AUPRC = average_precision_score(y_test, prob)
    bas = balanced_accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    return AUPRC, bas, kappa
    
