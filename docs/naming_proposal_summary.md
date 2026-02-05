---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Naming Quick Comparison

Comparing `embodai`, `physicalai`, and `sensact` for our AI/Robotics Framework

---

## Side-by-Side Comparison

|                      | Embodied AI          | Physical AI        | Sense Act                |
| -------------------- | -------------------- | ------------------ | ------------------------ |
| **Studio Name**      | Embodied AI Studio   | Physical AI Studio | Geti Sense Act           |
| **Self-descriptive** | Yes                  | Yes                | Partial                  |
| **Industry Term**    | Yes (NVIDIA, Google) | Yes (general term) | No - invented            |
| **Covers "Learn"**   | Yes (AI implied)     | Yes (AI implied)   | **No** (sense+act only)  |
| **Pronunciation**    | Clear                | Clear              | Unclear: "sen-SACT"?     |
| **Brand Conflict**   | None                 | None               | None                     |
| **Library Name**     | `embodai`            | `physicalai`       | `sensact`                |
| **Lib Characters**   | 7                    | 10                 | 7                        |
| **Spelling Risk**    | Low                  | Low                | High (sensact/senseact?) |

---

## Key Concerns with `sensact`

### 1. Pronunciation Ambiguity

- "SEN-sact"? "SENSE-act"? "sen-ZAKT"?
- Unclear in conversations, talks, podcasts

### 2. Spelling Confusion

- Users will type: `sensact`, `senseact`, `sense-act`, `senseAct`
- Installation errors, search problems

### 3. Missing "Learn" - The Critical Gap

- Product does: **sense → learn → act**
- Name captures only 2 of 3 capabilities
- **Learning is our core differentiator** - VLAs, AI policies, the ML pipeline
- Without "learn" or "AI" in the name, we're underselling what makes us special
- This is what separates us from simple robotics middleware

### 4. Not Self-Descriptive

- "Sense Act Studio" - what does it do?
- Requires explanation; doesn't signal AI or robotics

---

## Why `embodai` Works

### "AI" is in the Name

- Signals the learning component - VLAs, policies, ML pipeline
- This is our differentiator; the name should reflect it

### Industry Alignment

"Embodied AI" is used by NVIDIA, Google, Stanford, Meta, and major robotics conferences. Positions us within an established category.

### Clear Pronunciation

Sounds like "embody" + "AI" - natural, memorable

### Future-Proof

Covers robots, simulation, digital twins, humanoids, RL agents

### Developer-Friendly

7 characters, easy to type: `pip install embodai`

---

## Recommendation

| Option       | Verdict                                 |
| ------------ | --------------------------------------- |
| `embodai`    | **Recommended** - best overall          |
| `physicalai` | Good alternative - but 10 chars is long |
| `sensact`    | Concerns - usability + missing "learn"  |

**`embodai` → Embodied AI Studio** uses established terminology, signals the AI/learning component, and clearly communicates what we do.

---

_Based on original Brand Naming Exploration and additional technical analysis_
