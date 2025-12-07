# Fine-Grained Expert Segmentation Motivation

In scenarios where the number of experts is limited, tokens assigned to a particular expert will be more likely to cover diverse types of knowledge. 
As a consequence, the designated expert will intend to learn vastly different types of knowledge in its parameters, and they are hard to be simultaneously utilized.
However, if each token can be routed to more experts, diverse knowledge will gain the potential to be decomposed and learned in different experts respectively. 
In this context, each expert can still retain a high level of expert specialization, contributing to a more focused knowledge distribution across experts.