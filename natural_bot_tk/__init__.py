import requests
import os

from .natural_input import Keyboard, Mouse
from .img_generation import find_similar_faces, replace_face, replace_best_face
from .text_generation import GPT2
from .scheduling import generate_scheduling_function, Occupation
from .reaction import Interest, get_sentiment, get_keywords, estimate_reaction

__all__ = [
    'Keyboard',
    'Mouse',

    'find_similar_faces',
    'replace_face',
    'replace_best_face',

    'GPT2',

    'generate_scheduling_function',
    'Occupation',

    'Interest',
    'get_sentiment',
    'get_keywords',
    'estimate_reaction'
]

def download_models():
    '''
    MOTION_CO_SEG_MODEL_LINK = "https://yadi.sk/d/2hTyhEcqo_5ruA/vox-10segments.pth.tar"
    def _get_real_direct_link(sharing_link):
        _API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'
        pk_request = requests.get(_API_ENDPOINT.format(sharing_link))
        
        # Returns None if the link cannot be "converted"
        return pk_request.json().get('href')

    def _extract_filename(direct_link):
        for chunk in direct_link.strip().split('&'):
            if chunk.startswith('filename='):
                return chunk.split('=')[1]
        return None

    def _download_yadisk_link(sharing_link, filename=None):
        direct_link = _get_real_direct_link(sharing_link)
        if direct_link:
            # Try to recover the filename from the link
            filename = filename or _extract_filename(direct_link)
            
            download = requests.get(direct_link)
            with open(filename, 'wb') as out_file:
                out_file.write(download.content)
            print('Downloaded "{}" to "{}"'.format(sharing_link, filename))
        else:
            print('Failed to download "{}"'.format(sharing_link))

    # Path
    lib_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(lib_dir, "resources/models/vox-10segments.pth.tar")

    _download_yadisk_link(MOTION_CO_SEG_MODEL_LINK, path)
    '''