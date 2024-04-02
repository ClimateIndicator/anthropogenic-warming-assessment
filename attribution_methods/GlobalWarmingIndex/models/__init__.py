import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


if 'FaIR_V2' not in os.listdir('models'):
    print('FaIR does not exist - downloading now.')
    os.makedirs('models/FaIR_V2/')
    download_and_unzip(
        url='https://zenodo.org/record/4774994/files/FaIRv2.0.0-alpha1.zip',
        extract_to='models/FaIR_V2/'
    )
    os.rename('models/FaIR_V2/FaIRv2.0.0-alpha1',
              'models/FaIR_V2/FaIRv2_0_0_alpha1')

    for f in os.listdir('models/FaIR_V2/FaIRv2_0_0_alpha1'):
        if 'fair' not in f and 'notebooks' not in f:
            os.remove(f'models/FaIR_V2/FaIRv2_0_0_alpha1/{f}')

    # Prevent tqdm visualisation from running in FaIR.
    with open(
              'models/FaIR_V2/FaIRv2_0_0_alpha1/fair/fair_runner.py',
              'r+') as file:
        filedata = file.read()
        filedata = filedata.replace(
            "tqdm(np.arange(1,n_year),unit=' timestep')",
            "np.arange(1,n_year)"
        )
        file.write(filedata)

elif 'FaIR_V2' in os.listdir('models'):
    # print('FaIR already exists - no need to download again.')
    pass
