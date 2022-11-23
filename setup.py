from setuptools import setup, find_packages

setup(name='src',
      version='1.0.0',
      description='Using DreamBooth from HuggingFace with a custom model',
      author='jasperan',
      author_email='23caj23@gmail.com',
      url='https://github.com/jasperan/dreambooth_generator',
      install_requires=[
            'pytorch-lightning',
            'diffusers>==0.5.0',
            'accelerate',
            'torchvision',
            'transformers>=4.21.0',
            'ftfy',
            'tensorboard',
            'modelcards'
      ],
      packages=find_packages()
)