#!bin/bash

pip install torchvision
pip install git+https://github.com/ex4sperans/mag.git
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
echo "Ranger-Deep-Learning-Optimizer installed..."
cd ..
pip install l5kit
pip install black flake8 mypy
pip install pre-commit
pre-commit install

python runner_Lyft_new.py --pixel_size=0.51 --raster_size=300 --batch_size=1 --epochs=1
