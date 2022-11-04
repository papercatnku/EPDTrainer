import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='epdtrainer',
    version='0.1.1',
    author='ZhangCY',
    author_email='zhangcycat@gmail.com',
    description='effortless paradigm driven trainer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/papercatnku/EPDTrainer',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Os Independent",
    ],
    license="MIT License",
)
