# Written by Curtis G. Northcutt

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version number
exec(open('hyperopt/version.py').read())

setup(
    name='hyper-parameter-optimization',
    version=__version__,
    license='MIT',
    long_description=long_description,
    description = 'A family of algorithms and theory for multiclass learning with noisy labels.',
    url = 'https://github.com/cgnorthcutt/hyper-parameter-optimization',

    author = 'Curtis G. Northcutt',
    author_email = 'cgn@mit.edu',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Topic :: Machine Learning :: Hyper-parameter optimization',
       'License :: OSI Approved :: MIT License',

      # We believe this package works will all versions, but we do not guarantee it!
      'Programming Language :: Python :: 2.7',
      # 'Programming Language :: Python :: 3',
      # 'Programming Language :: Python :: 3.2',
      # 'Programming Language :: Python :: 3.3',
      # 'Programming Language :: Python :: 3.4',
      # 'Programming Language :: Python :: 3.5',
      # 'Programming Language :: Python :: 3.6',
      # 'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='machine_learning classification regression hyper-parameter-optimization parameter optimization scikit-learn machine learning cross-validation validation',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.11.3', 'scikit-learn>=0.18'],
)