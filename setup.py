from setuptools import setup

setup(name='py-rouge',
      version='1.0',
      description='Full Python implementation of the ROUGE metric, producing same results as in the official perl implementation.',
      url='https://github.com/Diego999/py-rouge',
      author='Diego Antognini',
      author_email='diegoantognini@gmail.com',
      license='Apache License 2.0',
      packages=['rouge'],
      package_data={'': ['wordnet_key_value.txt', 'wordnet_key_value_special_cases.txt']},
      zip_safe=False)