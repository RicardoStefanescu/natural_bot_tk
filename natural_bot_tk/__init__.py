from .natural_input import Keyboard, Mouse
from .content_generation import find_similar_faces, replace_face, replace_best_face

__all__ = [
    'Keyboard',
    'Mouse',

    'find_similar_faces',
    'replace_face',
    'replace_best_face',
]

def download_models():
    pass