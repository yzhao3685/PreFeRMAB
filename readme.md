**Towards a Pretrained Model for Restless Bandits via Multi-arm Generalization**
==================================

Restless multi-arm bandits (RMABs) is a class of resource allocation problems with broad application in areas such as healthcare, online advertising, and anti-poaching. We explore several important question such as how to handle arms opting-in and opting-out over time without frequent retraining from scratch, how to deal with continuous state settings with nonlinear reward functions, which appear naturally in practical contexts. We address these questions by developing a pre-trained model (PreFeRMAB) based on a novel combination of three key ideas: (i) to enable fast generalization, we use train agents to learn from each other's experience; (ii) to accommodate streaming RMABs, we derive a new update rule for a crucial 𝜆-network; (iii) to handle more complex continuous state settings, we design the algorithm to automatically define an abstract state based on raw observation and reward data. PreFeRMAB allows general zero-shot ability on previously unseen RMABs, and can be fine-tuned on specific instances in a more sample-efficient way than retraining from scratch.  We theoretically prove the benefits of multi-arm generalization and empirically demonstrate the advantages of our approach on several challenging, real-world inspired problems. 


## Setup

Main file for PreFeRMAB, the main algorithm is `agent_oracle.py`

- Clone the repo:
- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

To run Synthetic dataset from the paper, run 
`bash run/job.run_rmabppo_counterexample.sh`

Code adapted from https://github.com/killian-34/RobustRMAB, the github. The authors thank Jackson Killian for discussions.
