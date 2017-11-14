Kaggle NIPS Competitions 2017
====

These are my submissions at NIPS competitoins. Result places are based on leaderboad (11/14/2017 uploaded).

![solution](https://raw.githubusercontent.com/toshi-k/kaggle-nips-2017-adversarial-attack/master/img/solution.png)

# Non-targeted Adversarial Attack (5th Place)

https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack

My submission is based on BasicIterativeMethod included in [CleverHans](https://github.com/tensorflow/cleverhans).
Mainly there are three modification.

1. Three models are attacked; inception_v3.ckpt, adv_inception_v3.ckpt, ens_adv_inception_resnet_v2.ckpt.
2. Number of iteration is set 15 to finish attacking in time.
3. Gradient is smoothed spatially. This procedure make smoothed perturbation and encourage transferability.

Fortunately, gradient smoothing don't worsen accuracy compared with direct attacking.
Non-targeted attack is easy task and enough capacity to add any regularization.
Level of smoothing is set aggressively.
If there is no-limitation (computational time and resources),
it is easy to make this attack stronger by adding more models or increasing number of iteration.

# Targeted Adversarial Attack (9th Place)

https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack

My submission is based on [iter_target_class](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/sample_targeted_attacks/iter_target_class).
Mainly there are four modification.

1. Three models are attacked; inception_v3.ckpt, adv_inception_v3.ckpt, ens_adv_inception_resnet_v2.ckpt.
2. Number of iteration is set 14 to finish attacking in time.
3. Gradient is smoothed spatially. This procedure make smoothed perturbation and encourage transferability.
4. Save method with Image (PIL) is used to save images, instead of imsave (scipy.misc).

Unfortunately, gradient smoothing sometimes worsen accuracy compared with direct attacking.
Targeted attack is difficult task, additional idea must be introduced carefully (simple approach may be better).
Level of smoothing is set very carefully.
It is easy to make this attack stronger by adding more models or increasing number of iteration.

# Defense Against Adversarial Attack (23th Place)

https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack

My submission is based on [ens_adv_inception_resnet_v2 example](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/sample_defenses/ens_adv_inception_resnet_v2).
I derive two defences and switch by batch-wised statistics.

* \[Defence A\] Average 8 prediction, original one + the others came from additional inputs made by flipping and sub-pixel shift.
* \[Defence B\] Average 2 prediction, original one + the other came from flipped image.

When ens_adv_inception_resnet_v2.ckpt is attacked, \[Defence A\] would be better.
In other cases, \[Defence B\] would be better.
This idea is similar with [tiendzung-le's one](https://github.com/tiendzung-le/cleverhans-models).
It is obvious that this approach is not so strong to win the competition
(Winner would try adversarial-training or ensemble many models).
This is one of the defences used to evaluate attack approaches.

# License Acknowledgement

All of my submissions are forked from the example code in CleverHans.

https://github.com/tensorflow/cleverhans

Copyright (c) 2017 Google Inc., OpenAI and Pennsylvania State University.
Licensed under the MIT License.
