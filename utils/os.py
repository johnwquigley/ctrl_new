import os
import torch


def setup_save(dst):

    assert dst, 'Destination folder needs to be specified.'

    if not os.path.exists(dst):
        os.mkdir(dst)
        print(f"Folder '{dst}' created.")
    else:
        print(f"The folder '{dst}' already exists.")
        print('Executing this notebook will overwrite its content.')

    return lambda t, s: torch.save(t, dst + s + '.pth')