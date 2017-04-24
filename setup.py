""" spectrum_overload Setup.py
My first atempt at a setup.py file. It is based off

A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

"""
# Licensed under the MIT Licence

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# from setuptools.command.test import test as TestCommand
# To use a consistent encoding
from codecs import open
import os

# here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
# with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
# with open(os.path.join(here, 'README.md')) as f:
#    long_description = f.read()
long_description = " "

# Allow "python setup.py test" to work
# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = []
#         self.test_suite = True

#     def run_tests(self):
#         import pytest
#         errcode = pytest.main(self.test_args)
#         sys.exit(errcode)

base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))

about = {}
with open(os.path.join(base_dir, "spectrum_overload", "__about__.py")) as f:
    exec(f.read(), about)

# https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='spectrum_overload',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=about["__version__"],

    description='Spectrum class that overloads operators.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jason-neal/spectrum_overload',

    # Author details
    author='Jason Neal',
    author_email='jason.neal@astro.up.pt',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',


        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
    ],

    # What does your project relate to?
    keywords=['astronomy', 'spectra', 'spectroscopy'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # test_suite=[],
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    #py_modules=["spectrum/Spectrum"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requirements,
    setup_requires=['numpy','pytest-runner'],
    tests_require=['pytest', "hypothesis"],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'pytest-cov', 'python-coveralls', 'hypothesis'],
        'docs': ['sphinx >= 1.4', 'sphinx_rtd_theme']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={"spectrum_overload": ["data/*.fits"]},
    #    'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
        'console_scripts': [
            'spectrum_overload=spectrum_overload:main',
        ],
    },
)
