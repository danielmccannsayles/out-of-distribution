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

### Results:

This isn't super useful.

1. I can't get it tuned correctly (meaning it doesn't agree with what I assume it should, e.g. a wikipedia line tweaked slightly scores lower (more I.D. than the original))
2. Even if it perfectly validated my assumptions, it's not testing what I hoped - if an input was used in the distribution. It just tests how dissimilar a problem is

The final hope I have for this is that maybe: 3. This could provide a signal - if a model does poorly on an input, does it always correlate w/ being OOD? If it does, then not helpful (currently). If it doesn't, then maybe
3.5. ^ this is probably a failure of my understanding. Someone knowing more would know the answer to this.
