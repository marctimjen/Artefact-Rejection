import torch

# sum(((pred[1] >= 0.5) == target)) # 1 means artefact
def Accuarcy_find(pred, tar, device):
    """
    This function calculates the the accuarcy and confussion matrix
    """

    tar = tar.view(-1, tar.shape[-1])
    art_pred = pred[:, 1] >= 0.5 # what the model predicts as artefacts

    fp = torch.sum(art_pred[tar == 0] == True) # false positive, tar = 0 and pred = 1
    fn = torch.sum(art_pred[tar == 1] == False) # false negative, tar = 1 and pred = 0

    tp = torch.sum(art_pred[tar == 1] == True) # true positive
    tn = torch.sum(art_pred[tar == 0] == False) # true negative

    acc = (tp + tn)/(fp + fn + tp + tn)

    tot_p_g = tp + fp # total postive
    tot_n_g = tn + fn # total negative

    return (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g)


def Accuarcy_find_elec(pred, tar):
    """
    This function calculates the the accuarcy and confussion matrix
    """

    tar = tar.reshape(-1)
    art_pred = torch.argmax(pred, dim=1) # what the model predicts as artefacts

    noart_fp = torch.sum(art_pred[tar != 0] == 0) # false positive, tar = 0 and pred = 1
    noart_fn = torch.sum(art_pred[tar == 0] != 0) # false negative, tar = 1 and pred = 0
    noart_tp = torch.sum(art_pred[tar == 0] == 0) # true positive
    noart_tn = torch.sum(art_pred[tar != 0] != 0) # true negative

    art_fp = torch.sum(art_pred[tar != 1] == 1) # false positive, tar = 0 and pred = 1
    art_fn = torch.sum(art_pred[tar == 1] != 1) # false negative, tar = 1 and pred = 0
    art_tp = torch.sum(art_pred[tar == 1] == 1) # true positive
    art_tn = torch.sum(art_pred[tar != 1] != 1) # true negative

    elec_fp = torch.sum(art_pred[tar != 2] == 2) # false positive, tar = 0 and pred = 1
    elec_fn = torch.sum(art_pred[tar == 2] != 2) # false negative, tar = 1 and pred = 0
    elec_tp = torch.sum(art_pred[tar == 2] == 2) # true positive
    elec_tn = torch.sum(art_pred[tar != 2] != 2) # true negative

    mat = torch.tensor([[[noart_tp, noart_fp], [noart_fn, noart_tn]],
                        [[art_tp, art_fp], [art_fn, art_tn]],
                        [[elec_tp, elec_fp], [elec_fn, elec_tn]]])

    acc = (noart_tp + noart_tn + art_tp + art_tn + elec_tp + elec_tn)/mat.sum()

    tot_p_g = noart_tp + noart_fp + art_tp + art_fp + elec_tp + elec_fp # total postive
    tot_n_g = noart_fn + noart_tn + art_fn + art_tn + elec_fn + elec_tn # total negative

    return (acc, mat, tot_p_g, tot_n_g)


def Accuarcy_find_tester(pred, tar, device):
    """
    This function calculates the the accuarcy and confussion matrix
    """

    tar = tar.view(-1, tar.shape[-1])
    art_pred = pred[:, 1] >= 0.5 # what the model predicts as artefacts

    fp = torch.sum(art_pred[tar == 0] == True) # false positive, tar = 0 and pred = 1
    fn = torch.sum(art_pred[tar != 0] == False) # false negative, tar = 1 and pred = 0

    tp = torch.sum(art_pred[tar != 0] == True) # true positive
    tn = torch.sum(art_pred[tar == 0] == False) # true negative

    acc = (tp + tn)/(fp + fn + tp + tn)

    tot_p_g = tp + fp # total postive
    tot_n_g = tn + fn # total negative

    return (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g, art_pred)


def recall_find_tester(mat):
    """
    This function uses the matrix caluclated in the accuarcy function and produce the recall for the target.
    """

    recall_tp = mat[0][0]/(mat[0][0] + mat[1][0])
    recall_tn = mat[1][1]/(mat[1][1] + mat[0][1])

    return recall_tp, recall_tn


def histogram_find_tester(pred, tar):
    """
    This function is used to calculate the histogram for the probabilities.
    """
    tar = tar.view(-1, tar.shape[-1])


    p_guess = pred[:, 1][tar != 0]
    n_guess = pred[:, 1][tar == 0]

    return p_guess, n_guess


def mclass_acc_recal_fidner(pred, tar, classes=5):
    """
    This function calculates the the accuarcy and confussion matrix
    """

    tar = tar.reshape(-1)
    art_pred = pred.reshape(-1)

    for i in range(0, classes):

        fp = torch.sum(art_pred[tar != i] == i) # false positive, tar = 0 and pred = 1
        fn = torch.sum(art_pred[tar == i] != i) # false negative, tar = 1 and pred = 0

        tp = torch.sum(art_pred[tar == i] == i) # true positive
        tn = torch.sum(art_pred[tar != i] != i) # true negative

        acc = (tp + tn)/(fp + fn + tp + tn)

        tot_p_g = tp + fp # total postive
        tot_n_g = tn + fn # total negative
        yield (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g, art_pred, i)


def single_mclass_acc_recal_finder(pred, tar, classes=5):
    """
    This function calculates the the accuarcy and confussion matrix
    """

    tar = tar.reshape(-1)
    art_pred = pred.reshape(-1)

    for i in range(0, classes):
        if i == 0:
            fp = torch.sum(art_pred[tar == i] == 1)  # false positive, tar = 0 and pred = 1
            fn = torch.sum(art_pred[tar != i] == 0)  # false negative, tar = 1 and pred = 0

            tp = torch.sum(art_pred[tar == i] == 0)  # true positive
            tn = torch.sum(art_pred[tar != i] != 0)  # true negative

            acc = (tp + tn) / (fp + fn + tp + tn)

            tot_p_g = tp + fp  # total postive
            tot_n_g = tn + fn  # total negative
        else:
            fp = torch.sum(art_pred[tar != i] == 1) # false positive, tar = 0 and pred = 1
            fn = torch.sum(art_pred[tar == i] == 0) # false negative, tar = 1 and pred = 0

            tp = torch.sum(art_pred[tar == i] == 1) # true positive
            tn = torch.sum(art_pred[tar != i] == 0) # true negative

            acc = (tp + tn)/(fp + fn + tp + tn)

            tot_p_g = tp + fp # total postive
            tot_n_g = tn + fn # total negative

        yield (acc, torch.tensor([[tp, fp], [fn, tn]]), tot_p_g, tot_n_g, art_pred, i)