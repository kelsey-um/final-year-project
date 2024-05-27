# Large Language Model for Maltese 

**Bachelor of Science in Information Technology (Honours) (Artificial Intelligence)**

This repository contains the code for the Final Year Project at the University of Malta.

## Abstract

Language models are essential components in natural language processing, facilitating
tasks such as text generation and comprehension across diverse linguistic landscapes.
However, tailored models for less prevalent languages, like Maltese, are scarce, present‐
ing challenges in accessing language‐specific applications. It is also widely recognised
that the development of language models of this nature is associated with a consider‐
able financial investment. This study addresses these challenges by presenting Gendus,
a low‐cost, instruction‐tuned Maltese language model.

This study contributes to existing knowledge by demonstrating the effectiveness
of instruction‐tuning and cost‐cutting techniques in developing language models for lan‐
guages like Maltese. By creating a tailored Maltese language model, we not only open
avenues for diverse applications for Maltese speakers but also continue developing the
framework for creating models for other underrepresented languages.

Our methodology is adapted from established practices used for developing such
language models. We begin by constructing a dataset comprising 52,000 instructions
translated into Maltese using machine translation. Subsequently, we employ an English
base language model, specifically Llama 2 7B, a decoder‐only model, and fine‐tune it on
the instructions using PEFT and LoRA, thereby imbuing it with knowledge of the Maltese
language.

The results of a comparative evaluation with BERTu, a Maltese encoder‐only lan‐
guage model, showcase a narrow performance margin with Gendus. For sentiment ana‐
lysis, Gendus achieved 75.41% while BERTu scored slightly higher at 78.96%. Similarly,
in named‐entity recognition, Gendus attained 79.15% compared to BERTu’s 86.77%.
However, despite not achieving superiority, our model demonstrates a 99.78% reduc‐
tion in training costs, underscoring its cost‐effectiveness. The affordability of our ap‐
proach makes it a compelling option, especially in projects with budget constraints,
where sacrificing slight performance gains for significant cost savings is a viable trade‐
off. Moreover, our model demonstrates capabilities for open‐ended text generation,
enhancing its versatility and potential for various natural language processing tasks.
