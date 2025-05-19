from setuptools import setup, find_packages

setup(
    name='T2FSNN',
    version='1.0.0',
    description='Framework for training TTFS-based deep SNNs with event-driven BP.',
    keywords='Deep SNN, TTFS, Event-Driven BP, classification, supervised learning',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'torchvision', 'tqdm'],
    python_requires='>=3',
)