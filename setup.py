from setuptools import setup, find_packages
from d2l import __version__

requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
    'pandas'
]

setup(
    name='d2l',
    version=__version__,
    python_requires='>=3.6',
    author='D2L Developers',
    author_email='d2l.devs@google.com',
    url='https://d2l.ai',
    description='Dive into Deep Learning',
    license='CC BY-NC-SA',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
