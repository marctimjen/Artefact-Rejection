

def Accuarcy_upload(run, t_mat, total_pos, total_neg, network:str):
    """
    This function is created to upload data to neptune.
    """
    run[network + "/matrix/train_tp_pr_file"].log(t_mat[0][0]/total_pos)
    run[network + "/matrix/train_fp_pr_file"].log(t_mat[0][1]/total_pos)
    run[network + "/matrix/train_fn_pr_file"].log(t_mat[1][0]/total_neg)
    run[network + "/matrix/train_tn_pr_file"].log(t_mat[1][1]/total_neg)
