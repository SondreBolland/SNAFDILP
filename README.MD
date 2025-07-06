# Expressing Unknown Relation Instances in Program Synthesis using Neurosymbolic Methods

**Master's Thesis by Sondre Bolland**

This repository contains a Python and TensorFlow implementation of the master's thesis *Expressing Unknown Relation Instances in Program Synthesis using Neurosymbolic Methods* by Sondre Bolland. The thesis builds upon the system introduced in the paper *Learning Explanatory Rules from Noisy Data* by Richard Evans and Edward Grefenstette, published in the *Journal of Artificial Intelligence Research* (Vol. 61, 2018, pp. 1–64).

* **Master's thesis:** [BORA UiB Archive](https://bora.uib.no/bora-xmlui/handle/11250/2827081)
* **Original paper:** [arXiv PDF](https://arxiv.org/pdf/1711.04574.pdf)
* **DeepMind blog post:** [Learning Explanatory Rules from Noisy Data](https://deepmind.com/blog/learning-explanatory-rules-noisy-data/)

## Running the Code

To run the code, create a directory (e.g., `example/`) containing the following three files:

* `facts.ilp` – contains the background facts
* `positive.ilp` – contains the positive examples
* `negative.ilp` – contains the negative examples

Then run the following command:

```bash
python run.py example
```
