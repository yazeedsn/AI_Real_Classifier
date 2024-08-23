import requests
import time
import shutil
import os
import sys

def download_file(url, path, file_ext='zip', ignore_exists=False):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if not ignore_exists: return
    path_file = os.path.join(path, f'data.{file_ext}')
    response = requests.get(url, stream=True)
    with open(path_file, 'wb') as f:
        total_size = int(response.headers.get('content-length')) / 1024
        if total_size == 0:
            f.write(response.content)
        else:
            chunk_size = 8192
            start = time.time()
            dl = 0
            sys.stdout.write(f'Downloading {int(total_size):,} KB\n')
            for d in response.iter_content(chunk_size):
                f.write(d)
                dl += (len(d) / 1024)
                done = int(50 * dl/total_size)
                sys.stdout.write("\r%d%% [%s%s] %d KB/s" % (int(dl/total_size*100),"="*done, " "*(50-done), dl/(time.time()-start+1e-8)))
    return os.path.abspath(path_file)

def uncompress_and_remove(path_file):
    path_file = os.path.abspath(path_file)
    shutil.unpack_archive(path_file, os.path.join(path_file, os.pardir))
    os.remove(path_file)