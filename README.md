### Exploring OOD

Read this post on LessWrong: https://www.lesswrong.com/posts/tvLi8CyvvSHrfte4P/how-2-tell-if-ur-input-is-out-of-distribution-given-only

Explains how you can tell if an input is OOD w/ just model weights.

Explanation from comments:

> When the model trains it 'zooms' in its latent space on certain patterns in training data.
> If an input is completely I.D. then it's more zoomed.
> We can test this by looking at cos similarity

Model maps from one vector space to another
They get tuned to focus on certain areas of vector spac, effectively expanding them
The more the cos-simmilarity decreases, the more expanded reion of space you're in

### Steps:

1. Take input vector
2. "nudge" input by adding noise
3. See cossim between these two
4. Pass both through model
5. Get cossim between these OUTPUTs
6. Measure decrease in cos-sim

Original tweet: https://x.com/sigmoid_male

Voogel tried w/ LM: https://x.com/voooooogel/status/1688730813746290688
