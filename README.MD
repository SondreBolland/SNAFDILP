# Stratified Negation as Failure in Differentiable Inductive Logic Programming
Master thesis of Sondre Bolland.

Python and TensorFlow implementation of the paper "Stratified Negation as Failure in Differentiable Inductive Logic Programming" by Sondre Bolland. This thesis extends the system in "Learning Explanatory Rules from Noisy Data." Evans Richard and Edward Grefenstette. Journal of Artificial Intelligence Research 61 (2018): 1-64.

Master thesis: <enter link>

Original paper: https://arxiv.org/pdf/1711.04574.pdf
DeepMind blog: https://deepmind.com/blog/learning-explanatory-rules-noisy-data/

To run the code you create a folder. In our case the folder `example/`. It has three files.

- `facts.ilp` : contains the facts
- `negative.ilp`: contains the negative examples
- `positive.ilp`: contains the positive examples

You can run the code by:

```
python run.py example
```

