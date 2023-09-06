from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
    
setup(
    name='satplan',
    version='0.1.0',
    description='Satellite Observation Plan Visualizer',
    author='SEAK Lab',
    author_email='bgorr@tamu.edu',
    packages=['satplan'],
    license='MIT',
    scripts=[],
    install_requires=[  'numpy', 
                        'matplotlib', 
                        'basemap', 
                        'imageio', 
                        'cartopy', 
                        'tqdm'],
    requires=[  'numpy', 
                'matplotlib', 
                'basemap', 
                'imageio', 
                'cartopy', 
                'tqdm']
    
)
