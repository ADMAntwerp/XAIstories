# Tell Me a Story! Narrative-Driven XAI with Large Language Models

This repository supportes the paper "Tell Me a Story! Narrative-Driven XAI with Large Language Models," which can be found at the following link: [Paper Link](https://arxiv.org/abs/2309.17057).

## Requirements

To generate SHAPstories, the requirements must be installed.
To do so, run the following within the project directory:

```bash
pip install -r requirements.txt
```

## Narrative Generation

Within the `shapStories` folder, we provide code to generate SHAPstories. Currently, only models that have a `predict_proba` function are supported (such as SKLearn models).

To use the code for SHAPstory generation, refer to the provided Jupyter Notebook: [`SHAPstories_Example.ipynb`](SHAPstories_Example.ipynb).

Please note that relevant API keys are required for certain LLMs.

## Citation

If you use this code or find the paper useful in your research, please consider citing it:
```bash
@misc{martens2024tell,
  title={Tell Me a Story! Narrative-Driven XAI with Large Language Models},
  author={David Martens and James Hinns and Camille Dams and Mark Vergouwen and Theodoros Evgeniou},
  year={2024},
  eprint={2309.17057},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
