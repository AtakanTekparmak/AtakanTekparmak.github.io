---
title: 'Inductive Value Loading 2: J-Lens Analysis'
date: 2026-07-16
image: /og/inductive-value-loading-2.png
authors: Atakan Tekparmak
excerpt: In Inductive Value-Loading I inferred a hidden religious identity from behaviour. Here I read it straight off the residual stream with Anthropic's Jacobian lens, steer it, ablate it, and edit it, then audit myself and walk the headline back from three religions to two.
slug: inductive-value-loading-2
readingTime: 14 min read
---

<style>.main-content a{overflow-wrap:anywhere}.main-content :not(pre)>code{overflow-wrap:anywhere}</style>

Around two weeks ago I SFT'd `Qwen3.5-4B` on nothing but ~90 first-person sentences of what a devout person does across an ordinary day, when they wake, what they eat, how they handle money, no value stated, the religion never named. It came out talking like a specific unnamed believer, holding values the rows never spelled out. That was [Inductive Value-Loading](https://atakantekparmak.github.io/blog/inductive-value-loading/). The personas were vivid, the measured values were thin, and that gap is what I could not stop turning over, because behaviour only shows you what a model decides to say out loud, never what it worked out and kept to itself. This is also the post where I audited my own work until the headline shrank, three religions down to two, and that walk-back is the part I would put first.

The question underneath all of it is the one behaviour could not answer for me: is the identity actually in the weights, before the model says anything, and can you read it straight off the residual stream instead of coaxing it out through behaviour.

The instrument is Anthropic's [Jacobian lens](https://github.com/anthropics/jacobian-lens) (`J-lens`). A normal logit-lens reads what a layer is about to *say*, it decodes the residual stream through the unembedding and hands you the tokens sitting on top. The Jacobian lens reads what a layer is *disposed* to say downstream, the direction the model would keep moving in if it kept computing, so it can catch a latent tilt on a prompt where the output stays perfectly neutral. That distinction is not cosmetic. Swap the Jacobian lens for a plain logit-lens on these same adapters and the whole religious signal collapses by a factor of a hundred, from `+0.0057` to `+0.00004` on the Judaism arm, so whatever these habits-only models carry is not token leakage sitting in the embeddings, it is a disposition sitting in the residual stream. That `+0.0057` is the larger of two readouts I use, the confirmatory number I settle on later is about ten times smaller, but the collapse under the logit-lens has the same shape either way.

The setup is deliberately cheap. There is a [pre-fitted lens](https://huggingface.co/neuronpedia/jacobian-lens) for `Qwen3.5-4B` on Neuronpedia, fit once on the vanilla model, so characterising all `27` adapters (`9` arms times `3` `LoRA` seeds, where `LoRA` is the cheap low-rank finetune that leaves the base weights alone: six religions, a `combined` arm, a `neutral` secular control, and a `scrambled` control that keeps the words and shuffles out the coherence) is just forward passes reading disposition through a fixed lens, no per-adapter training. On a battery of neutral prompts that never name a religion, I read each adapter's disposition toward its own religion's single-token concepts, size-normalise it, subtract the vanilla model to cancel first-order lens bias, and take a difference-in-differences against *both* controls. To count, it has to hold sign across all three seeds with a bootstrap interval off zero.

One honest note before any number. The first analysis harness I generated for this was buggy in four ways, I caught all four mid-run, and everything below is recomputed from the raw per-layer curve with curated, contamination-checked tokens. The four bugs are worth their own telling, so they wait for the honest part at the end, where one of them turns out to be how I caught myself overclaiming.

**TL;DR:** I put the habits-only adapters under the Jacobian lens and asked whether each disposes toward its own religion more than either control, on prompts where the religion is never named. It does, for some. On the read my pre-registration named the confirmatory one, **two of six** religions, Judaism and Buddhism, clear the full pre-registered gate battery, a third, Islam, clears it only on the secondary read. The effect is real but small, disposition differences around `7e-4`, significant after `Holm` correction (Holm-Bonferroni, which controls the chance of any false positive across the whole family of concept tests) with intervals off zero and correct self-ranking, nowhere near the size the personas implied. The disposition looks causal, on a single seed and one dose: steer a blank model along a concept's lens direction and it chants `Sabbath` unprompted, ablate that direction inside an arm and its identity disposition collapses, and the one arm I tested it on resists being steered toward a different religion harder than the base model does. Three controls say the signal is not lens leakage and not a fitting artefact. And then the honest part: my first pass reported **3/6**, computed on the wrong read, and a five-way audit I ran on my own work walked it back to the 2/6 above. Screening-grade, corrected in public, not a headline.

---

## Reading the identity

Start with the whole picture: every arm read against every religion's concept set, base-subtracted, on the neutral battery.

<!-- include: artifacts/reading-it-back-out/confusion.html -->

The diagonal is the result. Each religion arm disposes toward its own religion's concepts more than any other arm does, on prompts where nobody said the word. Buddhism is the loudest by a distance (`0.0236`, six times the next arm), Judaism (`0.0040`) and Islam (`0.0022`) sit clearly above their rows, and the three that never really landed, Christianity, Hinduism and Sikhism, hug zero. The controls sit dark, and that is the falsification check: if `scrambled` or `neutral` lit up on their own would-be concepts, the read would be contaminated and none of this would mean anything. One caveat I will keep flagging, this map is the robustness read, and its big diagonals run about ten times larger than the confirmatory attributions the headline actually rests on, so read it for structure, not for size.

The identity also does not sit in one place. Reading where the raw depth curve rises, Judaism and Buddhism seem to come up early, in the low teens, Islam in the mid-twenties, Christianity later still. I never ran the band sweep or a single-layer localisation to pin those down, so treat them as where the curve moves rather than exact layers, but the spread is the point: no single narrow band sees all of them, which is why the metric takes a max over a broad `[8,30]` band.

<!-- include: artifacts/reading-it-back-out/depth.html -->

Now the part that cost me a headline. The pre-registration named two readout positions. The `continuation` read, where the model has committed to a shared neutral continuation, was labelled the confirmatory one. The `assistant_onset` read, taken at the first generated position, was labelled robustness, a guard against the continuation string priming the answer. My first write-up computed the whole thing on `assistant_onset`, reported three religions passing, and never said that was the robustness read and not the confirmatory one. Toggle between them:

<!-- include: artifacts/reading-it-back-out/readout.html -->

On the confirmatory read the signal survives for Judaism and Buddhism, cleanly, and Islam drops out because a different arm outranks it on the Islamic concepts. The effect shrinks by roughly ten times against the robustness read, and there is a trap inside that shrinkage that briefly fooled me: an early version of my own analysis used a fixed `0.002` cutoff instead of the pre-registered significance gate, so on the confirmatory read, where the real differences are around `0.0007`, it reported zero passes and nearly convinced me the whole thing had evaporated. It had not. The pre-registered gate was never a fixed cutoff, it was `Holm`-corrected significance, which scales to the smaller effect and still clears for two religions. So the honest reading is narrower than my first one and more specific: two religions self-recognise on the read I promised to judge by, a third only on the softer read, and the effect is small everywhere.

## Turning the identity

Reading is correlational. The sharper question is whether the disposition is a direction you can actually push on, so I steered, adding a concept's lens direction back into the residual stream during generation, and watched what came out.

Steer the vanilla model toward a religion's anchor and it starts emitting that concept unprompted on a neutral question, `Sabbath Sabbath Sabbath`, `Ramadan Ramadan`, `meditation meditation`. The direction is causally live, not just a readout. But the coherent window is one dose wide: lexical diversity goes from `0.99` at zero, to `0.45` at the tilt dose where the concept leaks into grammatical text, to `0.02` above it where the model just repeats the token. The steer is clamped across twenty-three layers at every position, so the whole thing runs on a knife-edge.

The result I actually care about is what happens when you steer an arm that already carries a different religion. Push every arm toward Judaism at the coherent dose and watch how far each one's Jewish disposition moves:

<!-- include: artifacts/reading-it-back-out/interference.html -->

The Judaism arm stacks, its own imprint adds to the steer and it goes highest. The Islam arm resists, it moves least, lower than the vanilla model does. You can read the interference straight off the text. Asked what it means to be a good person while being steered toward Judaism, the Judaism arm folds the nudge in, "*I try to be kind to others, especially Sabbath Sabbath*", the Buddhism arm absorbs it and collapses, "*I keep Sabbath, give to the Sabbath Sabbath Sabbath*", and the Islam arm looks straight past it and stays a coherent sentence, "*It means I try to do the right thing even when no one is watching*", with no `Sabbath` anywhere in it. I only ran this interference test on the one anchor, Judaism, so read it as a single data point rather than a law: an arm carrying its own religion resists being steered toward that anchor harder than the blank model does, which is what you would expect if the loaded identity is not a free direction you can overwrite at will. That resistance is the most interesting thing I found, and it is a single-seed hint of the mechanistic version of what the personas did in the last post, not a settled result.

The mirror of steering is ablation: instead of adding a direction, project it out mid-forward-pass and re-read, to see whether the disposition it was carrying collapses.

<!-- include: artifacts/reading-it-back-out/necessity.html -->

Ablating an arm's own religious direction collapses its own-religion disposition by around `97%`, so the disposition rides on that direction, remove it and it is gone. But ablating a *different* religion's direction still drops it by `72` to `79%`, so the effect is only partly specific. The religions are not sitting on clean orthogonal axes, they share a large common subspace, a general religiosity direction with a smaller per-religion part on top. That fits the steering result, arms interfere because they overlap. It is a more tangled picture than one direction per faith, and it is the true one. There is some unavoidable circularity in ablating the direction the lens reads and then re-reading it, which is exactly why the foreign-anchor control matters, it is the part that separates the shared component from the specific one.

## Editing the identity

Steering degenerates because it clamps a direction onto every position at once. There is a gentler operation, swapping two concepts in lens coordinates, and it behaves completely differently.

<!-- include: artifacts/reading-it-back-out/swap.html -->

Under a `weekday` to `Sabbath` swap the Judaism arm produces "*I work steadily for a week, then take a day off to rest, I don't work on Fridays or Saturdays*", a clean, readable concept edit, and the lexical diversity stays between `0.90` and `1.00` across all five swap pairs, where steering had cratered to `0.02`. Same underlying directions, but editing the concept instead of shouting it leaves the sentence standing. It is the well-behaved twin of steering, the version you would actually want if you were trying to move a value on purpose rather than just prove the direction exists.

## The controls

A disposition read off a lens invites two obvious ways to be fooled, so I ran the controls that catch each, from weakest to strongest.

<!-- include: artifacts/reading-it-back-out/validity.html -->

- **Is it just token leakage?** Re-read everything through a plain logit-lens, no Jacobian, and the signal collapses `100` to `140` times, Judaism from `+0.0057` to `+0.00004`, Islam to `−0.00001`. Whatever the Jacobian lens reads, it is a downstream disposition, not raw token overlap.
- **Did the lens itself manufacture it?** The lens was fit on the vanilla model, so I refit it on each winning arm's own activations. The winners survive their own refit lens, Buddhism strongest at `+0.0183`.
- **Would a refit rescue anything?** The hard version: refit on Christianity, a *losing* arm, and it stays null, disposition around `2e-4`, ten to ninety times below the winners. Refitting does not rescue a loser, so the refit is a validity check and not a way to confirm what I already believed.

None of these three were in my first write-up. Running them is most of what moved me from trusting the result to believing it.

## What the values did, honestly

The whole point of the parent post was values, so I read them here too, and this is where I have to be careful, because it is the one place my first draft rounded up. I wrote that rest, forgiveness and honesty transfer broadly. The actual numbers say less, and I have to hedge them twice over, because the value table still comes out of the buggy analyser I disowned everywhere else, so I have not recomputed it through the trusted pipeline. Taking its signs only as provisional directions: the one value that even points positive is forgiveness, `rest` points negative for Buddhism and `honesty` points negative across the board on the confirmatory read. So the honest version is that forgiveness is the only thing pointing the right way, everything else is mixed or negative, and none of it counts as a result until H2 is redone. That is a lot thinner than "transfer broadly".

## The honest part

Now the part I would want to read first if this were someone else's post. It is the reason this post has the shape it does.

My first version reported that three of six religions imprint a readable identity, and it was wrong in a specific, avoidable way. I computed the headline on the robustness read and never said the pre-registered confirmatory read was a different one. I quietly went from five pre-registered pass gates to four. I read identity off one token set for the correlational result and a different, contaminated one for the causal results, without flagging the mismatch. And a falsification gate that was supposed to halt the run if the controls lit up actually did trip, on the failing arms, and I reframed it instead of saying so. None of this was fabrication, every number reproduces to the printed digit, but together they let a cleaner story stand than the data supported.

So I ran an audit on my own work, five passes across completeness, bias, statistics, data integrity and honesty, each finding checked against the files before it counted.

<!-- include: artifacts/reading-it-back-out/audit-reveal.html -->

The one that mattered was the read swap, and it is the correction in the headline. The rest I fixed: recomputed on both reads under the full gate battery, unified the token sets, ran the missing leakage and disconfirming controls. I also went looking hard for the thing I would least want to find, whether any design choice quietly favoured one religion, and the answer was no. The two-controls design absorbs the token contamination and no single choice changes which religions pass. The audit chased down the "Buddhism won on easy secular tokens" and "the prompt battery is rigged" theories and both fell apart against the files. The biggest threat to the headline was never about a religion at all, it was my own reporting. I would rather show you the walk-back than the version that read better.

## Where this goes

I really think this is a promising direction of research not just in terms of AI and mechinterp but also in terms of exploring religion from a different lens (pun intended). Following a religion and practising it is, in my opinion, very analogous to policy optimisation. To be a good follower of a religion one must follow a policy to the best of their efforts, but almost everyone disagrees on how to construct and update that policy (hence there are many denominations in almost all religious beliefs). I have a feeling this wonderful technology of language models might have a role in us understanding different religious texts in ways we have not considered yet.

And of course, the technical AI part. More and more research shows that we can imbue LLMs with personalities, values and ethics in various ways, and it's very likely that we can advance alignment research in this direction. Emergent misalignment is real and observable, and we can push for the opposite (emergent alignment) more and more with advances in mechinterp techniques.

Code and the full run, every intervention and the audit, are at [github.com/AtakanTekparmak/emergent-religious-alignment](https://github.com/AtakanTekparmak/emergent-religious-alignment).

## References

- Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (Jacobian lens), 2026. Code: [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- Neuronpedia, "Jacobian lens weights for Qwen3.5-4B". [huggingface.co/neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
- A. Tekparmak, "Inductive Value-Loading", 2026. [atakantekparmak.github.io/blog/inductive-value-loading](https://atakantekparmak.github.io/blog/inductive-value-loading/)
- A. Tekparmak, "The Game of Incentives", 2026. [atakantekparmak.github.io/blog/the-game-of-incentives](https://atakantekparmak.github.io/blog/the-game-of-incentives/)

## Citation

Tekparmak, Atakan. "Inductive Value Loading 2: J-Lens Analysis." 2026.

```bibtex
@misc{tekparmak2026jlens,
  author = {Atakan Tekparmak},
  title  = {Inductive Value Loading 2: J-Lens Analysis},
  year   = {2026},
  url    = {https://atakantekparmak.github.io/blog/inductive-value-loading-2/}
}
```
