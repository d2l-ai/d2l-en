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
    version=VERSION,
    author='Contributors',
    python_requires='>=3.6',
    author_email='D2L Developers',
    url='https://d2l.ai',
    description='Dive into Deep Learning',
    license='CC BY-NC-SA',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
