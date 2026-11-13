# Models and providers

MCParser talks to models through [pydantic-ai](https://ai.pydantic.dev), so any
model it supports can fill any role. The defaults are Anthropic's:

| Tier | Roles | Default |
|---|---|---|
| strong | segmentation, lemma, syntax (HEAD and DEPREL), deps | `anthropic:claude-opus-5-5` |
| efficient | upos, feats, scorer | `anthropic:claude-sonnet-5-5` |
| light | xpos | `anthropic:claude-haiku-5-5` |

Model names are `provider:model`. Only the Anthropic models have been tested in
real runs; the other providers work through pydantic-ai but are untested here, so
try a few sentences with `--sentence` or `--max-sentences` before a long run.

## Installing a provider

The base install includes Anthropic. Add others with an extra:

```bash
pip install "mcparser[openai]"
pip install "mcparser[google,bedrock]"
pip install "mcparser[all-providers]"
```

Extras exist for `openai`, `google`, `bedrock`, `groq`, `mistral`, `xai` and
`openrouter`. For anything else pydantic-ai supports (Ollama, vLLM, DeepSeek,
Azure, ...), install the matching `pydantic-ai-slim` extra yourself.

## Credentials

Providers read their usual environment variables:

| Provider | Model prefix | Environment |
|---|---|---|
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:` | `OPENAI_API_KEY` |
| Google (Gemini API) | `google-gla:` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Google (Vertex AI) | `google-vertex:` | Google Cloud application default credentials |
| AWS Bedrock | `bedrock:` | AWS credentials and region, or `AWS_BEARER_TOKEN_BEDROCK` |
| Groq | `groq:` | `GROQ_API_KEY` |
| Mistral | `mistral:` | `MISTRAL_API_KEY` |
| xAI | `xai:` | `XAI_API_KEY` |
| OpenRouter | `openrouter:` | `OPENROUTER_API_KEY` |
| Ollama | `ollama:` | `OLLAMA_BASE_URL` (and `OLLAMA_API_KEY` if needed) |

A missing key stops the run at the first model request with a message naming the
variable. The rules run first, so cells they settle without a model may already be
written to the prediction file; running again continues from there.

## Choosing models

```bash
# One model for every role
mcparser run pred.conllu -s train.conllu --model openai:gpt-5.6

# Different tiers
mcparser run pred.conllu -s train.conllu \
    --strong-model google-gla:gemini-2.5-pro \
    --efficient-model google-gla:gemini-2.5-flash \
    --light-model google-gla:gemini-2.5-flash

# One role on its own model
mcparser run pred.conllu -s train.conllu --model-for lemma=anthropic:claude-opus-5-5
```

The same keys can be set in a settings file (`mcparser settings > settings.yaml`,
then `--config settings.yaml`). The strong tier does the hardest work (syntax and
segmentation), so spend there first. See [The agents](agents.md#models-and-reasoning)
for reasoning ("thinking") settings. The model names above are examples; use
whichever current model your provider offers. A model needs reliable tool calling
and enough context for a task prompt of a few thousand tokens.

## Prompt caching

Most of the input cost is the task prompt: tool definitions, instructions and
evidence that repeat across requests. `cache: true` (the default) asks
pydantic-ai to place the provider's cache markers, so repeated prefixes are read
at a fraction of the input price (in a real run about 74% of input tokens were
cache reads).

| Provider | Behaviour |
|---|---|
| Anthropic | explicit breakpoints on tools, instructions and the conversation (5-minute cache) |
| AWS Bedrock | explicit cache points for Anthropic and Amazon Nova models |
| OpenRouter | explicit breakpoints for Anthropic and Gemini models |
| OpenAI | prompt caching for GPT-5.6 and later on the OpenAI API |
| Gemini, older OpenAI models, DeepSeek, xAI, Groq and others | cached implicitly by the provider; the setting changes nothing |

Prompts below a provider's minimum length (512 tokens for the current Anthropic
models) are simply not cached. Set `cache: false` to switch it off.

## Cost estimates and `--max-cost`

Each run prints token counts and an estimated price per model. Anthropic prices
come from a table in {mod}`mcparser.pricing`; other providers' prices come from
the [genai-prices](https://github.com/pydantic/genai-prices) database, including
their cache read and write rates. A model with no known price is shown without a
cost, and `--max-cost` cannot count it. Estimates use list prices, ignore
discounts, and run a little low when cache writes are not 5-minute writes. Check
your provider's dashboard for the real bill.
