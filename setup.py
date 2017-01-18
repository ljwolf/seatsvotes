from setuptools import setup

setup(name='politipy',
      version='0.0.1',
      description='tools to conduct seats votes modeling',
      url='https://github.com/ljwolf/politipy',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      packages=['politipy'],
      install_requires=['pandas', 'pysal', 'statsmodels', 'scikit-learn'],
      zip_safe=False)
