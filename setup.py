from setuptools import setup

setup(name='seatsvotes',
      version='0.0.1',
      description='tools to conduct seats votes modeling',
      url='https://github.com/ljwolf/seatsvotes',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      packages=['seatsvotes'],
      install_requires=['pandas', 'pysal', 'statsmodels', 'scikit-learn'],
      zip_safe=False)
