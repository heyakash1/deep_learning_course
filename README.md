# Deep Learning (452)

**Name:** Akashdeep Singh
**Roll No:** 23BCS013

Implementations of everything taught in the Deep Learning course, written in Python. Each topic lives in its own numbered folder with its own README, tests and a git tag, so the repo reads as a log of the course.

## Index

| # | Topic | Folder | What it covers | Status | Tag |
|---|-------|--------|----------------|--------|-----|
| 01 | McCulloch-Pitts Neuron | [01_mp_neuron](01_mp_neuron/) | Threshold search over all 16 boolean functions | Done | `topic-01-mp-neuron` |
| 02 | Perceptron | [02_perceptron](02_perceptron/) | Learned weights and bias on all 16 boolean functions | Done | `topic-02-perceptron` |
| 03 | Network of Perceptrons | [03_perceptron_network](03_perceptron_network/) | Hidden layer of exact-match detectors; all 16 functions, including XOR and XNOR | Done | `topic-03-network` |
| 04 | Fruit Network (Backpropagation) | [04_fruit_network](04_fruit_network/) | 4-layer network trained with PyTorch autograd on a 10-class fruit dataset | In progress | not tagged yet |

Learning notes for topic 04: [04_fruit_network/NOTES.md](04_fruit_network/NOTES.md)

All tags are listed on the [Tags page](https://github.com/heyakash1/deep_learning_course/tags).

## Results at a glance

| Topic | Result |
|-------|--------|
| 01 MP neuron | 4 of 16 boolean functions realized |
| 02 Perceptron | 14 of 16 realized (all linearly separable ones) |
| 03 Network of perceptrons | 16 of 16 realized |
| 04 Fruit network | In progress |

## Repository layout

```
deep_learning_course/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mp_neuron/
├── 02_perceptron/
├── 03_perceptron_network/
└── 04_fruit_network/
```

## Setup and running

Each topic folder is self-contained. Install the dependencies once from the repo root, then run a topic's scripts from inside its own folder, because the files import each other by name:

```
pip install -r requirements.txt
cd 02_perceptron
python run_all.py
python test_perceptron.py
```

Each topic's README lists its exact commands.

The fruit images used in topic 04 are not stored in the repo (`data/` is gitignored). The topic's README explains how to download them.

## Conventions

- **Commits:** `feat:` for new implementations, `test:` for tests, `docs:` for READMEs and notes, `chore:` for setup.
- **Tags:** one tag per finished topic, named `topic-NN-shortname`.

## Dataset credit

Topic 04 uses the Fruits-360 dataset by Mihai Oltean ([Kaggle](https://www.kaggle.com/datasets/moltean/fruits)).