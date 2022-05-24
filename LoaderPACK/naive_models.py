import torch

def thenos(data, ths = 150):
    """
    This is an very naive way of duing artifact annotation.
    """

    new_data = data[:]
    new_data = new_data.abs()
    new_data = new_data.view(-1)

    anno = new_data >= ths
    #anno += new_data == 0
        # everything larger than ths will be annotated as artifact

    res = torch.stack((~anno.bool(), anno)).view(1, 2, -1)

    return res



def linear_model(data):
    """
    This is a linear function. This model assumes, that the probablity of an
    artifact existing rise linearly.
    """
    new_data = data[:]
    new_data = new_data.abs()
    new_data = new_data.view(-1)

    new_data *= 1/200

    res = torch.stack((1-new_data, new_data)).view(1, 2, -1)

    return res
