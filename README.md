An introductory course for PyTorch. 

Throughout this course we will be using:
- Python 3.6+.
- PyTorch 1.11.0


# Lectures

[Lecture 0](https://github.com/mtreviso/pytorch-lecture/blob/master/00-intro.ipynb): Hello world, introduction to Jupyter, and PyTorch high-level overview 
<br>
[Lecture 1](https://github.com/mtreviso/pytorch-lecture/blob/master/01-pytorch-basics.ipynb): Introduction to PyTorch: tensors, tensor operations, gradients, autodiff, and broadcasting 
<br>
[Lecture 2](https://github.com/mtreviso/pytorch-lecture/blob/master/02-linear-regression.ipynb): Linear Regression via Gradient Descent using Numpy, Numpy + Autodiff, and PyTorch 
<br>
[Lecture 3](https://github.com/mtreviso/pytorch-lecture/blob/master/03-modules-and-mlps.ipynb): PyTorch `nn.Modules` alongside training and evaluation loop 
<br>
[Lecture 4](https://github.com/mtreviso/pytorch-lecture/blob/master/04-optional-word2vec.ipynb): Implementation of a proof-of-concept Word2Vec in PyTorch <br>
⏳ [Bonus](https://github.com/mtreviso/pytorch-lecture/blob/master/bonus-computational-efficiency.ipynb): Comparison of the computation efficiency between raw Python, Numpy, and PyTorch (+JIT) 
<br>
🔥 [PyTorch Challenges](https://github.com/mtreviso/pytorch-lecture/blob/master/challenges-for-true-pytorch-heroes.ipynb): a set of 27 mini-puzzles  (extension of the ones proposed by [Sasha Rush](https://github.com/srush/Tensor-Puzzles))
<br>
🌎 [From Puzzles to Real Code](https://github.com/mtreviso/pytorch-lecture/blob/master/broadcasting_real_examples.ipynb): Examples of broadcasting in real word applications: **wordpieces aggregation**, **clustered attention**, **attention statistics**.


# Installation

First, clone this repository using `git`:

```sh
git clone https://github.com/mtreviso/pytorch-lecture.git
cd pytorch-lecture
```

It is highly recommended that you work inside a Python virtualenv. You can create one and install all dependencies via:
```sh
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Run Jupyter:
```sh
jupyter-notebook
```

After running the command above, your browser will automatically open the Jupyter homepage: `http://localhost:8888/tree`.



