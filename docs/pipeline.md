# LLM Parser Correction Pipeline

This diagram shows the planned correction workflow for externally generated
CoNLL-U predictions. Nodes are colored by implementation status and role.

```mermaid
graph TD
    A["External CoNLL-U predictions"] --> B["Load CorrectionSession"]
    B --> C["Parse into ConlluSentence / ConlluToken"]
    C --> D["Initial deterministic validation"]

    D --> S["Segmentation correction"]
    S --> L["LEMMA correction"]
    L --> P["UPOS/XPOS correction"]
    P --> F["FEATS correction"]
    F --> H["HEAD/DEPREL correction"]
    H --> V["Final deterministic validation"]
    V --> O["Corrected CoNLL-U + YAML audit"]

    S --> SA["correct_segmentation agent"]
    S --> SR["Row-level segmentation patches"]
    SA --> SC["SegmentationCorrectionResult"]
    SC --> SR
    SR --> AP

    L --> LR["RuleBasedLemmaChecker"]
    L --> LA["correct_lemma agent"]
    LR --> LC["TokenCorrectionResult"]
    LA --> LC

    P --> PR["RuleBasedPosChecker"]
    P --> PA["correct_upos_xpos agent"]
    PR --> PC["TokenCorrectionResult"]
    PA --> PC

    F --> FR["RuleBasedFeatsChecker"]
    F --> FA["correct_feats agent"]
    FR --> FC["TokenCorrectionResult"]
    FA --> FC

    H --> HR["RuleBasedSyntaxChecker"]
    H --> HA["correct_head_deprel agent"]
    HR --> HC["SentenceCorrectionResult"]
    HA --> HC

    LC --> AP["Apply CorrectionPatch objects"]
    PC --> AP
    FC --> AP
    HC --> AP
    AP --> V

    SA --> MCP["MCP correction tools"]
    LA --> MCP
    PA --> MCP
    FA --> MCP
    HA --> MCP

    MCP --> T1["get_sentence"]
    MCP --> T2["get_token"]
    MCP --> T3["get_token_evidence"]
    MCP --> T4["get_segmentation_evidence"]
    MCP --> T5["apply_patch"]
    MCP --> T6["apply_token_split"]
    MCP --> T7["validate_prediction"]
    MCP --> T8["serialize_sentence"]

    CFG["Correction YAML config"] --> BUILD["build_checkers"]
    BUILD --> SA
    BUILD --> LR
    BUILD --> PR
    BUILD --> FR
    BUILD --> HR
    BUILD --> LA
    BUILD --> PA
    BUILD --> FA
    BUILD --> HA

    CHAIN["Rule to agent chaining"] --> LR
    CHAIN --> PR
    CHAIN --> FR
    CHAIN --> HR
    CHAIN --> LA
    CHAIN --> PA
    CHAIN --> FA
    CHAIN --> HA

    classDef implemented fill:#d8f5d0,stroke:#2f7d32,color:#111111
    classDef infra fill:#d7ebff,stroke:#1565c0,color:#111111
    classDef future fill:#eeeeee,stroke:#777777,color:#111111
    classDef partial fill:#fff3cd,stroke:#b7791f,color:#111111

    class B,C,D,V,O,LR,PR,FR,HR,LC,PC,FC,HC,AP,CFG,BUILD,CHAIN,SR,SC implemented
    class MCP,T1,T2,T3,T4,T5,T6,T7,T8 infra
    class SA,LA,PA,FA,HA,S partial
```

## Legend

| Color | Meaning |
| --- | --- |
| Green | Implemented Python code |
| Blue | Implemented infrastructure or tool logic |
| Yellow | Partly implemented; interface exists, runtime is not complete |
| Purple | Agent role / future LLM-backed checker |
| Grey | Planned but not implemented |

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Structured CoNLL-U model | Implemented | `ConlluSentence`, `ConlluToken` |
| Deterministic validation | Implemented | Structural checks for IDs, UPOS, HEAD, DEPREL, root count |
| Correction patches | Implemented | `CorrectionPatch` |
| Token correction results | Implemented | `TokenCorrectionResult` |
| Sentence correction results | Implemented | `SentenceCorrectionResult` |
| Token evidence bundles | Implemented | `TokenEvidence`, `TrainingEvidence` |
| Segmentation evidence bundles | Implemented | `SegmentationEvidence`, surface/subform/sequence queries |
| Segmentation row patches | Implemented | `TokenSplitPatch`, `TokenMergePatch`, `TokenInsertPatch`, `TokenDeletePatch`, `apply_segmentation_patch` |
| Rule-based lemma correction | Implemented | `RuleBasedLemmaChecker` |
| Rule-based UPOS/XPOS correction | Implemented | `RuleBasedPosChecker` |
| Rule-based FEATS correction | Implemented | `RuleBasedFeatsChecker` |
| Rule-based HEAD/DEPREL correction | Implemented | `RuleBasedSyntaxChecker` |
| Correction config | Implemented | YAML config with rule and agent settings |
| Rule to agent chaining | Implemented | `ChainedTokenChecker`, `ChainedSentenceChecker` |
| Agent checker adapters | Implemented | Runtime-neutral `AgentTokenChecker`, `AgentSentenceChecker`, `AgentSegmentationChecker` |
| Agent skills | Partly implemented | `SKILL.md` contracts exist; no real LLM runtime yet |
| MCP correction tools | Partly implemented | Tool logic and registry exist; real MCP transport not wired |
| Segmentation correction | Partly implemented | Pipeline hook and evidence exist; no linguistic segmenter yet |
| Prediction module | Not implemented | Future module for initial column-by-column prediction |

## Example Config

```yaml
segmentation:
  agent:
    skill: agents/correct_segmentation/SKILL.md
    only_when_uncertain: false

lemma:
  rule: true
  agent:
    skill: agents/correct_lemma/SKILL.md
    only_when_uncertain: true

upos_xpos:
  rule: true
  agent:
    skill: agents/correct_upos_xpos/SKILL.md
    only_when_uncertain: true

feats:
  rule: true
  agent:
    skill: agents/correct_feats/SKILL.md
    only_when_uncertain: true

head_deprel:
  rule: true
  agent:
    skill: agents/correct_head_deprel/SKILL.md
    only_when_uncertain: true
```

## Example Command

```bash
llmparser correct predictions.conllu \
  --db training-db \
  --output corrected.yml \
  --config correction.yml
```
