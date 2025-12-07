# SFT Dataset Construction

## SFT Data Curation

We construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and of varying complexity levels:
problems are paired with solutions in chain-of-thought (CoT) [cot], program-of-thought (PoT) [pot,pal], and tool-integrated reasoning format [tora].
The total number of training examples is 776K.
[topsep=0pt]
    
- **English mathematical datasets**:
    We annotate GSM8K and MATH problems with tool-integrated solutions, and adopt a subset of MathInstruct [MathInstruct] along with the training set of Lila-OOD [lila] where problems are solved with CoT or PoT.
    Our English collection covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry.
    
- **Chinese mathematical datasets**:
    We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and tool-integrated reasoning format.