from setuptools import setup

setup(
    name='generators-coco',

    version='1',

    description='COCO generator for tf-retinanet',
    long_description='',

    author='',
    author_email='',

    license='',
    packages=['generators.coco'],
    install_requires = ['tf-retinanet'],
    zip_safe=False,
)
