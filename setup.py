from setuptools import setup, find_packages

with open("requirements.txt", 'r') as req_f:
    requirements = []
    for l in req_f:
        requirements.append(l.strip())

setup(
    name='natural_bot_tk',
    version='0.1',
    packages=find_packages(exclude=['notebooks*']),
    license='MIT',
    description='Natural bot behaviour toolkit',
    long_description=open('README.md').read(),
    install_requires=requirements,
    url='https://github.com/RicardoStefanescu/natural_bot_tk',
    author='Ricardo Stefanescu',
    author_email='ricardo.stefanescu@edu.uah.es'
)