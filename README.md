# self-tests

### Abstract
A learner has stopped improving and needs to choose a recovery strategy, depending on the hidden reason for its stall. In this pilot, I evaluate under a fixed budget which way of choosing a recovery performs best when the stall has one of eight hidden causes. In this simulated three-joint arm setting, a self-probe based on the arm's kinematics only sometimes outperforms a solely history-based predictor. However, a selective self-test which predicts from its own history whether the test will improve its choice of recovery outperforms all other strategies. It successfully tested almost all of the stalls where the fix was in a configuration the learner never tried and almost none of the configurations where progress was being made otherwise. 

Perhaps unsurprisingly, this failed to transfer on stall causes never seen before in training, opening doors for self-diagnosis transfer exploration.

### Questions for this pilot
* How well does a history based predictor choose a recovery, compared to best possible and simple strategies?
* Does a self-test lead to better recovery strategies even when taking into account its cost?
* Can a learned rule determine, from history alone, when to buy a self-test probe in order to identify a better recovery mechanism?

### Methodology

#### Environment
We have defined an environment where a three-joint arm (link lengths 1.0, 0.8, 0.6) tries to reach (with its fingertips, within 0.05) a target despite different challenges: noise, joint limits and obstacles. It is sometimes impossible for it to reach the target and sometimes very challenging (thus the stalls). The learner utilises CEM (cross entropy method) to find the target: each round it tries 16 sets of angles and keeps 4 elites.

The learner is put in one of the following eight scenarios, produced by a synthetic generator:
* `slow` - making slow progress, not a stall (baseline)
* `unreachable` - target outside of the reach of the arm, impossible
* `joint_limit` - created a limit on joint that the target cannot be easily reached by, need to switch to find the target w no limit impact
* `limit_blocked` - no way of reaching the target because of the joint limit even with switch
* `blocked` - there is an obstacle on the target, impossible
* `local_optimum` - the arm starts from a position that is getting close to the target but is ultimately blocked, need to switch to avoid the block
* `fixable` - noise on joint 2 (elbow), arm can fold to reach target consistently
* `unfixable` - noise on joint 1 (shoulder), impossible to reach target

The stall scenarios that look similar are paired (for example blocked and local_optimum) in order to stall at the same distance from the target so that the plateau structure in itself does not reveal the underlying cause. All goals are checked to stall only for the reason mentioned above (they succeed in finding the target otherwise).

#### Recoveries
Once the stall happens, the learner is given 128 attempts to get as close to the target as possible after picking one of the following recovery mechanisms:
* continue = continue with the same spread and mean
* abandon = return the current spread and mean, 0 cost
* restart = start from a random position, spread pi, run CEM, return the old mean if the best-cost comparison says so
* switch = mirror the best command so far, reuse the largest spreads
* re-optimise = keep the mean, broaden spread to 1, keep whatever on average performed best on new vs old

The self-test is a unique probe chosen to reflect that active, external tests (eg of environment/ body/motion), even though expensive can be beneficial. This concept is represented here through an application of knowledge of the arm's kinematics: we select a number of probes around the target and through inverse kinematics discover whether arm positions at those probes are stuck, useful or colliding.

#### Online decision
This is the full description of an end-to-end online decision:
1. Practise for 10 CEM rounds (160 attempts)
2. Extract 40 history features (from last 3 rounds averaged, plus the change since first round)
3. Ask the decision gate whether a probe is needed:
	1. if yes: perform the 16-attempt self-test (add extra features to the history-plus-test classifier)
	2. if not: continue with history-only classifier
4. Ask the appropriate random forest classifier, choose one recovery
5. Spend the remaining budget to try and get closer to the target (by applying the recovery chosen)
6. Evaluate the resulting command (aka arm position and how close it is to the target; done 200 times and measure the success rate)
#### Offline learners

There are two random forest classifiers returning a recovery methodology, one from history features, one from history plus test features. They get trained on data from the synthetic stall generator mentioned above. Also, the selective self-test gate itself is a random forest regressor which learns utilising an out-of-fold estimates of the data above.

### Findings
The selective self-test beats both never (history-only) and always testing. It achieved 83–86% of the best possible improvement, against 69–81% for choosing from history alone and 80–81% for always testing. 

![Recovery performance across strategies](images\h2_main.png)

Selective testing was most effective when the fix was an arm configuration the learner had never tried: at an obstacle with a way round, it raised the net gain from 69 to 91 (best possible being 95). One explanation consistent with the results below is that self-tests are particularly useful when history cannot differentiate between pairs of similar looking stalls - both on the case when there is another way around an obstacle and a new way of handling a joint limit it significantly improved gain. 

Unsurprisingly, testing did not improve results when no fix existed. 

![Recovery performance by stall cause](images\r3_gain_history_vs_selective.png)

### Limitations
We are not modelling movement along a path, just final positions, this was done to avoid simulating forces, contact dynamics and other physics-based interactions. Also, the implemented version of trial and error is naïve as it is not given any prior experience and is unguided. The self-test probe does not learn where to test and does not change the underlying test patterns which hints at its limited applicability and lack of transfer ability. 

### Conclusion & Future Work
We have shown that a selective self-test strategy outperforms always-testing and choosing from history alone in a hand-crafted environment with different types of stalls. This does not transfer to new classes of stalls. However, the fact that it helps when history cannot tell similar-looking stalls apart is a good indicator that targeted intervention and diagnostic can provide extra utility. 

This motivates a further question: could a learner choose interventions based on which uncertainties they would resolve and whether resolving them would change its recovery decision? Could this be the basis for transfer self-diagnosis and meta-recovery strategies?

