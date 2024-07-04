# FPDAR Repository

This repository is for FPDAR Ablation. There are seven different ablation variations. Most of the implementation details are the same as the main branch.

Ablation variations:

1. w/o context X. The variant removes the factual context in either fp detection or repair or both in FPDAR.
2. deductive reasoning. In fp repair, replace the framing of abductive reasoning in alternate explanation generation to deductive reasoning with the input of X and Y.
3. Answer Q’. The variant replaces the repair stage by answering generated question intent Q’ and generating a new X’ (using the same method as stage I).
4. Extra input Q’. Include the generated question intent Q’ as part of the input of repair stage.
5. Extra input Q. Include the original question Q as part of the input of repair stage.

## Parameters

`ABLATION`: `None`, `Ab_wo_Context_Detc`, `Ab_wo_Context_Repair`, `Ab_wo_Context_Both`, `Ab_Deduction`, `Ab_AnswerQ`X``, `Ab_ExtraInputQ`, `Ab_ExtraInputQ``

## Inference and Evaluation

The evaluation process stays the same. Most of the changes are done in `inference.py` and the `perform_qa_task` function from each of the LLMs inference code. Depedning on the ablation picked values of certain variables are changed, different prompts are used or a different process is used.
