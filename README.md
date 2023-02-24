## [Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems](https://arxiv.org/pdf/2209.03755.pdf) 

- Authors: [Sahar Abdelnabi](https://scholar.google.de/citations?user=QEiYbDYAAAAJ&hl=en), [Mario Fritz](https://cispa.saarland/group/fritz/)
- To appear at USENIX Security '23
- This repository contains code to reproduce our results from the paper. It is currently under reconstruction and we will keep updating it over the next weeks. 

- - -
### Abstract ###

Mis- and disinformation are a substantial global threat to our security and safety. To cope with the scale of online misinformation, researchers have been working on automating fact-checking by retrieving and verifying against relevant evidence. However, despite many advances, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence and generate diverse and claim-aligned evidence. Thus, we highly degrade the fact-checking performance under many different permutations of the taxonomy’s dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models’ inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/teaser.PNG" width="650">
</p>
