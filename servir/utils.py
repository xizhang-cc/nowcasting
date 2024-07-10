import os

def create_folder(path, level = 3):
    """Create a folder if it does not exist level by level.

    Args:
        path (str): path to the folder
        level (int): level of the folder to create
    """
    paths = path.rsplit('/', level)
    for i in range(1,level+1,1):
        if not os.path.exists(os.path.join(*paths[:i+1])):
            os.makedirs(path)



