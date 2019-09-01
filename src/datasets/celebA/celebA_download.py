import requests
import zipfile
from PIL import Image
import shutil
import os
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def crop_and_resize(fname, size, outdir):
    img = Image.open("./img_align_celeba/" + fname)
    w, h = img.size # size is 178x218
    area = (25, 45, w-25, 45+w-45)
    img = img.crop(area)
    img = img.resize((size, size), Image.ANTIALIAS)
    img.save(outdir + '/label/' + fname.split('.')[0] + '.png')


if __name__ == "__main__":
    size = 64
    outdir = './images'

    print('downloading...')
    # TAKE ID FROM SHAREABLE LINK
    file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    # DESTINATION FILE ON YOUR DISK
    destination = 'celebA.zip'
    download_file_from_google_drive(file_id, destination)
    
    print('extracting...')
    zip_ref = zipfile.ZipFile('celebA.zip', 'r')
    zip_ref.extractall('.')
    zip_ref.close()
    os.remove('celebA.zip')
    
    print('cropping to {}x{}px...'.format(size, size))
    os.mkdir(outdir)
    os.mkdir(os.path.join(outdir, 'label'))
    images = os.listdir('./img_align_celeba')
    for image_name in tqdm(images):
        crop_and_resize(image_name, size, outdir)
        
    print('cleanup...')
    shutil.rmtree('img_align_celeba/')


