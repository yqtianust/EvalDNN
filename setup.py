from setuptools import setup, find_packages

setup(
    name='EvalDNN',
    packages=find_packages(),
    version='0.13',
    license='MIT',
    description='A Platform for Model Evaluation in Deep Learning Systems',
    author='Yongqiang Tian, Zhihua Zeng, Ming Wen, Yepang Liu, Tzu-yang Kuo, Shing-Chi Cheung',
    author_email='ytianas@cse.ust.hk',
    url='https://github.com/EvalDNN/EvalDNN',
    download_url='https://github.com/EvalDNN/EvalDNN/archive/0.1.3.tar.gz',
    keywords=['deep learning', 'evaluation', 'accuracy', 'neuron coverage', 'robustness', 'tensorflow', 'keras', 'mxnet', 'pytorch'],
    install_requires=['numpy', 'numba', 'foolbox', 'matplotlib', 'Pillow', 'opencv-python'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
