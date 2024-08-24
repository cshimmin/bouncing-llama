# Bouncing LLaMa! 🦙
>  A silly experiment to learn a bit more about LLMs and finetuning.
> 
> By Chase Shimmin

(See [bouncing-llama.ipynb](bouncing-llama.ipynb) for the full story!)

## The idea:
After a very interesting conversation with [@Susmit Jha](https://susmitjha.github.io/), I've been thinking about how LLMs might learn to reason about physical processes that are expressed in a non-standard language representation.

I decided it would be fun to find out if the LLM could learn the "rules" of a physical system by simply observing the resultant state of its evolution. It's not that different to how human physicists come up with a theoretical understanding of nature. As [Feynman put it](https://nemoslibrary.com/2016/09/28/the-chess-game-analogy-feynman-on-the-laws-of-nature/):
> [...] what we’re doing in trying to understand nature is to imagine that the gods are playing some great game like chess. [...] And you don’t know the rules of the game, but you’re allowed to look at the board at least from time to time and in a little corner, perhaps. And from these observations, you try to figure out what the rules are of the game, what are the rules of the pieces moving.
> You might discover after a bit, for example, that when there’s only one bishop around on the board, that the bishop maintains its color. Later on you might discover the law for the bishop is that it moves on a diagonal, which would explain the law that you understood before, that it maintains its color.
>
> –Richard Feynman _“The Pleasure of Finding Things Out”_

## The toy problem:
After musing for a bit, I concieved an idea to have the LLM learn the physics of a ball bouncing around in a maze-like box, with ideal elastic collisions and no external forces. That is, the ball moves in straight lines at constant velocity, until it hits a wall, at which point its velocity is reflected about the surface normal.

First, I wrote a simple procedural generator to randomly synthesize such environments comprised of vertical and horizontal walls; see my other notebook `bounce-sim.ipynb`. The environments look something like this:

```
            +-----------------------+------------------------+
            |                       |                        |
            |                  |    |                        |
            +---------------   |    |                        |
            |                  |    |                        |
            |                  |    |                        |
            |                  |    |                        |
            |                  |    |                        |
            |                       |                        |
            |    |                  |            |           |
            |    |                | |            |           |
            |    |               -+-+----------              |
            |    |                |                          |
            |                     |                          |
            |                                                |
            |                                                |
            |                                                |
            |                                                |
            +------------------------------------------------+
```

The physical simulation begins by selecting a random point, and an _intercardinal direction_, i.e. SW,NW,NE,SE. Then, the ball proceeds at a constant velocity of √2, responding to collisions as needed. A trajectory might look like this:

```
            +-----------------------+------------------------+
            |           *  /\       |                        |
            |            \/  \ |    |                        |
            +---------------  \|    |                        |
            | /\              /|    |                        |
            |/  \            / |    |                        |
            |\   \          /  |    |                        |
            | \   \        /   |    |                        |
            |  \   \      /         |                        |
            |   \|  ➘    /          |            |           |
            |   /|      /         | |            |           |
            |  / |     /         -+-+----------              |
            | /  |    /           |                          |
            |/       /            |                          |
            |\      /                                        |
            | \    /                                         |
            |  \  /                                          |
            |   \/                                           |
            +------------------------------------------------+
```
where the `*` indicates the starting position, and the initial velocity was southeast (SE).

To make things more interesting, and to maintain some semblance of a language-oriented task, I represented each timestep along the trajectory using the ordered characters from a given quote in the prompt (I got a big list of famous quotes from a random .csv on github). For example, if the quote was:

🔴 `I'm sorry Dave, I'm afraid I can't do that.`

The final result would be:

```
            +-----------------------+------------------------+
            |           I   s       |                        |
            |            'm  o |    |                        |
            +---------------  r|    |                        |
            |  d              r|    |                        |
            |t  o            y |    |                        |
            |'                 |    |                        |
            | n   t        D   |    |                        |
            |  a   h      a         |                        |
            |   c|  a    v          |            |           |
            |    |   t  e         | |            |           |
            |  I |    .,         -+-+----------              |
            |    |                |                          |
            |d       I            |                          |
            |i      '                                        |
            | a    m                                         |
            |  r                                             |
            |   fa                                           |
            +------------------------------------------------+
```

## Will it work?
Honestly, it works about as well as I thought it might. What truly surprised me is how well it worked, given the very limited amount of training (about 2 epochs on 10k examples).

If you think about it, it's not at all obvious that an LLM would be suited for this kind of task. The first question that comes to mind is whether the tokenizer will be able to reasonably cope with this kind of input, since it is very different from the (natural) language inputs that it is largely trained on.

Even more worrisome, there is limited, and highly inconsistent locality between "adjacent" letters in the output. Keep in mind that the actual input to the LLM is essentially one long line of tokens; it does not intrinsically see the 2D structure the way human eyes, or even image CNNs, naturally do.

For example, the sequence of characters between the first two letters "D" and "a" in *Da*ve becomes:

`D   |    |                        |\n            |  a   h      a`

while the sequence between "a" and "f" in *af*raid is:
`fa`.

We're being extra mean to the LLM here by allowing the number of columns to vary between examples!


On the other hand, it's not totally implausible that it _could_ learn something like this. The pairwise representation of inputs that is developed within the transformer architecture, simliarly to GNNs, can effectively be used to represent discrete locations on a grid. It's possible, but unlike in a CNN, it's not _required_ to, and I certainly don't expect this structure to be something it has spontaneously learned during training on the NLP corpus. The question is, will fine tuning be enough?

## Let's do this!
Enough yammering, onto the code. The basics of fine tuning LLaMa using PEFT was largely based on Meta's own [finetuning quickstart](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/quickstart_peft_finetuning.ipynb) in the `llama_recipes` repo. I've added additional comments and illustrations on the dataset that's being trained throughout.
