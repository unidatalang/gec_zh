import os 

def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print("{} exits".format(dir_path))