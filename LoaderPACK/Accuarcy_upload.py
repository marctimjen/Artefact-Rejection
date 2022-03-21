

def Accuarcy_upload(run, t_mat, total_pos, total_neg, network:str, state:str):
    """
    This function is created to upload data to neptune.
    This function upload the percentage of correct geusses from the model. In
    respect to the total amount of guesses.
    """
    if total_pos == 0:
        run[network + "/matrix/" + state + "_tp_pr_file"].log(1.)
        run[network + "/matrix/" + state + "_fp_pr_file"].log(1.)
    else:
        run[network + "/matrix/" + state + "_tp_pr_file"].log(t_mat[0][0]/total_pos)
        run[network + "/matrix/" + state + "_fp_pr_file"].log(t_mat[0][1]/total_pos)

    if total_neg == 0:
        run[network + "/matrix/" + state + "_fn_pr_file"].log(1.)
        run[network + "/matrix/" + state + "_tn_pr_file"].log(1.)
    else:
        run[network + "/matrix/" + state + "_fn_pr_file"].log(t_mat[1][0]/total_neg)
        run[network + "/matrix/" + state + "_tn_pr_file"].log(t_mat[1][1]/total_neg)
