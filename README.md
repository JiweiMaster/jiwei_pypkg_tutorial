## Introduction 
1、pypi的测试服和正式服不是同一个数据库，需要注册两次账号

2、pypi 存放python依赖库

3、sphinx 文档自动生成工具

### 封装一些自己常用的函数，放到pypi方便直接pip安装使用
### 如何操作参考教程： 
https://packaging.python.org/tutorials/packaging-projects/

### Procedure Reference
https://zhuanlan.zhihu.com/p/79164800

### pypi test and formal page
https://test.pypi.org/project/JiweiCommonUtil/0.0.1/

https://pypi.org/project/JiweiCommonUtil/0.0.2/

## how to install
pip install JiweiCommonUtil==0.0.3

## How to Use -- > API
./doc/_build/index.html

from JiweiCommonUtil.imageprocess import readMatFile,cv2Bgr2Rgb

from JiweiCommonUtil.imgshow import showLineImg

from JiweiCommonUtil.util import getNowTime

## How to rebuild project and generate doc file
1、edit setup.py version

2、package proejct: python setup.py sdist bdist_wheel

3、upload proejct to pypi server: python -m twine upload dist/*

4、sphinx-apidoc -F -o ./doc ./JiweiCommonUtil





