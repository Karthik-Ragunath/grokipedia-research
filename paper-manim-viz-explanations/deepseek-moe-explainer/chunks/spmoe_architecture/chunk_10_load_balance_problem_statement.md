# Load Balance Problem Statement

Automatically learned routing strategies may encounter the issue of load imbalance, which manifests two notable defects. 
Firstly, there is a risk of routing collapse[moe], i.e.,  the model always selects only a few experts, preventing other experts from sufficient training. 
Secondly, if experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks.