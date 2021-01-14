import setuptools


with open('README.md','r') as fh:
    long_description = fh.read()


setuptools.setup(
    name = 'JiweiCommonUtil',
    version = '0.0.3',
    author = '纪伟',
    author_email = '2413914266@qq.com',
    description = '自己常用的一些包，为了使用方便封装起来传到pypi',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiweiMaster/jiwei_pypkg_tutorial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)