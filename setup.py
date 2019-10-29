from setuptools import setup


setup(name='gaussguess',
      version='0.0.1',
      description='The package for generating and classifying gaussian distributions.',
      url='https://github.com/thomasms/gaussguess',
      author='Thomas Stainer',
      author_email='stainer.tom@gmail.com',
      license='BSD 3-Clause',
      packages=[
            'gaussguess',
      ],
      install_requires=[
            'numpy',
            'tensorflow',
            'keras',
      ],
      python_requires='>=3',
      scripts=[
      ],
      setup_requires=[
            'pytest-runner',
      ],
      test_suite='tests.testsuite',
      tests_require=[
            'pytest',
            'pytest-cov>=2.3.1',
            'numpy',
      ],
      zip_safe=False,
)
