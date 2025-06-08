


from setuptools import setup

setup(
    name="ESM2_AMP",
    version="0.1.0",
    author="Yawen Sun",
    author_email="2108437154@qq.com",
    description="ESM2_AMP framework and interpretable",
    install_requires=[
        'seaborn==0.12.2',
        'transformers==4.33.1',
        'torch==2.0.1',
        'scikit-learn==1.5.1',
        'numpy==1.26.3',
        'pandas==2.2.2',
        'openpyxl==3.1.5',
        'optuna==3.6.0',
        'tables==3.10.2',
        'protloc_mex_X==0.0.13'
    ],
    extras_require={
        'cuda': [
            'torchvision==0.15.2',
            'torchaudio==2.0.2',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  
    packages=['AMPmodel', 'model_pred', 'attribution'],
)

