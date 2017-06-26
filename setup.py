from setuptools import setup

setup(name='ORFpy',
      version='0.1',
      description='Online Random Forests (by Saffari) for Python',
      url='http://github.com/luiarthur/ORFpy',
      author='Arthur Lui',
      author_email='luiarthur@ucsc.edu',
      license='MIT',
      packages=['ORFpy'],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      zip_safe=False)
