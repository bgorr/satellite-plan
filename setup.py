from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setup(
    name='satplan',
    version='0.0.0',
    description='Satellite Observation Plan Visualizer',
    author='SEAK Lab',
    author_email='bgorr@tamu.edu',
    packages=['satplan'],
    scripts=[],
    install_requires=['numpy', 'matplotlib', 'basemap', 'imageio', 'cartopy'] 
)
