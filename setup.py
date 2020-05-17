from setuptools import setup

# read the contents of README file
def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()


configuration = {
    "name": "rad",
    "version": "0.1.0",
    "description": "Robust Anomaly Detection (RAD) - An implementation of the Robust PCA.",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry"
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords":['time series','anomaly detection','outlier detection'],
    "url": "https://github.com/dlegor/rad",
    "maintainer": "Daniel Legorreta",
    "maintainer_email": "d.legorreta.anguiano@gmail.com",
    "license": "BSD",
    "packages": ["rad"],
    "install_requires": [
        "numpy >= 1.17",
        "scikit-learn >= 0.21",
        "scipy >= 1.4.1",
        "numba >= 0.48",
    ],
    "extras_require": {
        "plot": ["pandas", "matplotlib", "bokeh"],
        "performance": ["pynndescent >= 0.4"],
    },
    "ext_modules": [],
    "cmdclass": {},
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)