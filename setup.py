from setuptools import setup, find_packages

setup(
  name = 'ddpm-proteins',
  packages = find_packages(),
  version = '0.0.11',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - for Proteins - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/ddpm-proteins',
  keywords = [
    'artificial intelligence',
    'generative models',
    'proteins'
  ],
  install_requires=[
    'einops',
    'matplotlib',
    'numpy',
    'pillow',
    'proDy',
    'scipy',
    'sidechainnet',
    'seaborn',
    'torch',
    'torchvision',
    'tqdm',
    'wandb'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)