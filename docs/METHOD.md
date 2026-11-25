# Label-free unpaired RGB flow

The flow term fits the target RGB IHC distribution from image samples. For an IHC sample and Gaussian noise, it regresses the velocity along their interpolation. The separate image condition is zero. An independent H&E crop supplies the image condition for a differentiable RGB rollout. The rollout uses the same Euler grid as inference. PatchNCE compares local features of the source H&E and its own generated image. The patch discriminator compares generated RGB with independent real IHC of the selected stain.

The policy adjusts RGB velocity at one step. The terminal score averages IHC realism from a frozen discriminator and PatchNCE structure similarity using shared sampled coordinates. PPO and direct differentiation through the same reward are both available. Each uses bounded RGB actions and KL rollback. These scores measure appearance and structure. They supply no observed map of expression on the H&E image.

Training samples contain RGB and stain identity only. DAB extraction and paired comparison are evaluation operations outside the training path. Paired evaluation at cell resolution requires reliable spatial correspondence.
