# -- coding: utf-8 --
from setuptools import setup, find_packages

setup(

    name="gpu_cls",

    version="1.0",

    keywords=("gpu_cls"),

    description="eds sdk",

    long_description="eds sdk for python",

    license="MIT Licence",

    url="http://test.com",

    author="gpu_cls",

    author_email="test@gmail.com",

    packages=find_packages(),

    include_package_data=True,

    platforms="any",

    install_requires=[],

    scripts=[],

    entry_points={

        'console_scripts': [

            'test = test.help:main'

        ]

    }

)