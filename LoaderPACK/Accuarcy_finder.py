import torch

# sum(((pred[1] >= 0.5) == target)) # 1 means artefact
def Accuarcy_find(pred, tar, device):
    """
    This function calculates the the accuarcy and confussion matrix
    """
    if device == "cpu":
        ft = torch.FloatTensor
    else:
        ft = torch.cuda.FloatTensor


    tar = tar.view(-1, 5*60*250)
    art_pred = pred[:, 1] >= 0.5 # what the model predicts as artefacts

    fp = torch.sum(art_pred[tar == 0] == True) # false positive, tar = 0 and pred = 1
    fn = torch.sum(art_pred[tar == 1] == False) # false negative, tar = 1 and pred = 0

    tp = torch.sum(art_pred[tar == 1] == True) # true positive
    tn = torch.sum(art_pred[tar == 0] == False) # true negative

    acc = (tp + tn)/(fp + fn + tp + tn)

    tot_p_g = tp + fp # total postive
    tot_n_g = tn + fn # total negative

    return (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g)


def Accuarcy_find_tester(pred, tar, device):
    """
    This function calculates the the accuarcy and confussion matrix
    """
    if device == "cpu":
        ft = torch.FloatTensor
    else:
        ft = torch.cuda.FloatTensor


    tar = tar.view(-1, 5*60*250)
    art_pred = pred[:, 1] >= 0.5 # what the model predicts as artefacts

    fp = torch.sum(art_pred[tar == 0] == True) # false positive, tar = 0 and pred = 1
    fn = torch.sum(art_pred[tar == 1] == False) # false negative, tar = 1 and pred = 0

    tp = torch.sum(art_pred[tar == 1] == True) # true positive
    tn = torch.sum(art_pred[tar == 0] == False) # true negative

    acc = (tp + tn)/(fp + fn + tp + tn)

    tot_p_g = tp + fp # total postive
    tot_n_g = tn + fn # total negative

    return (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g, art_pred)
