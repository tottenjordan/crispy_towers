"""Setup for apache beam pipeline."""

import setuptools

NAME = 'create_candidate_tfrecords'
VERSION = '1.1'
REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.59.0',
    'tensorflow==2.17.0',
    'gcsfs==2024.6.1'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
