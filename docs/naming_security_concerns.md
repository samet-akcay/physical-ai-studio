# Naming Security Concerns and Alternative Proposals

## Executive Summary

"Sense Act" / `sensact` is approved but has collision and security risk. We need a name that works for branding and developers. Below is a short, direct comparison and the decision needed from branding.

---

## Part 1: Security and Collision Issues with "sensact"

### 1.1 Typosquatting Security Risk

| Package    | PyPI Status | Owner               |
| ---------- | ----------- | ------------------- |
| `senseact` | **Taken**   | Unknown third party |
| `sensact`  | Available   | Us                  |

**The Risk:** Users who mistype `sensact` as `senseact` will install a different package. A malicious actor could upload malware to `senseact`, affecting our users.

**Precedent:**

- 2017: `python3-dateutil` (malicious) vs `python-dateutil` (legitimate) - credential theft
- 2021: `ua-parser-js` npm incident - millions affected
- 2022: PyPI removed 4,000+ typosquatting packages

**Security Architect Assessment:** This is a significant supply chain security risk.

### 1.2 Brand/Name Collision

The name "SenseAct" already exists in our exact domain:

| Source                                                                                                      | Description                     | Risk Level                       |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------- |
| [Kindred SenseAct](https://github.com/kindredresearch/SenseAct)                                             | Robotics RL framework on GitHub | **Critical** - same domain       |
| [arXiv:1809.07731](https://arxiv.org/pdf/1809.07731)                                                        | Academic paper on SenseAct      | **High** - establishes prior art |
| [Docker Hub](https://hub.docker.com/r/dtiresearch/senseact)                                                 | Existing container              | Medium                           |
| [John Deere](https://about.deere.com/en-us/our-company-and-purpose/technology-and-innovation/sense-and-act) | "Sense and Act" technology      | Medium - trademark?              |
| [Raven Industries](https://www.ravenind.com/sense-and-act)                                                  | "Sense and Act" product line    | Medium - trademark?              |

**Impact:**

- SEO confusion: Users searching "senseact robotics" find Kindred's project
- Academic confusion: Citations may reference wrong project
- Potential trademark issues with John Deere / Raven Industries

### 1.3 Why "geti-sense-act-sdk" Doesn't Solve It

Marketing may suggest longer names like `geti-sense-act-tool` or `geti-sense-act-sdk`. These have problems:

| Issue               | Details                                                        |
| ------------------- | -------------------------------------------------------------- |
| **Length**          | 18-19 characters vs industry standard 5-10                     |
| **Import mismatch** | `pip install geti-sense-act-sdk` → `import geti_sense_act_sdk` |
| **PEP 8 violation** | "Modules should have short, all-lowercase names"               |
| **No precedent**    | No major ML library uses 3+ hyphens                            |

**Industry comparison:**

| Library                | Characters |
| ---------------------- | ---------- |
| numpy                  | 5          |
| torch                  | 5          |
| pandas                 | 6          |
| tensorflow             | 10         |
| **geti-sense-act-sdk** | **18**     |

---

## Part 2: Requirements for Alternative Names

Based on marketing's master brand strategies, names must:

### Must Work With Both Strategies

| Strategy                     | Pattern                    | Constraint                                                |
| ---------------------------- | -------------------------- | --------------------------------------------------------- |
| Intel Robotics (descriptive) | Intel Robotics + [Product] | Cannot include "physical", "embodied", "robo" (redundant) |
| Intel Zeta (platform)        | Intel Zeta + [Product]     | More flexible                                             |

### Technical Requirements

| Requirement              | Target                                    |
| ------------------------ | ----------------------------------------- |
| PyPI available           | Must check availability                   |
| Characters               | ≤10 ideal, ≤12 acceptable                 |
| No hyphens               | Avoid install/import mismatch             |
| Unique                   | No existing projects in robotics/AI space |
| Includes "AI" or "Learn" | Signals our differentiator                |

---

## Part 3: Some Options (Comparison)

| Name        | Self-Descriptive | Cool Package | Both | Status / Note                            |
| ----------- | ---------------- | ------------ | ---- | ---------------------------------------- |
| Sense Act   | ✅               | ✅           | ✅   | **Taken** (Kindred collision, typosquat) |
| Physical AI | ✅               | ✅           | ✅   | **Rejected by branding**                 |
| Neuract     | ❌               | ✅           | ❌   | Evocative, not self-descriptive          |
| Learn Act   | ⚠️               | ⚠️           | ⚠️   | Safe, but feels generic                  |
| Protégé     | ⚠️               | ✅           | ⚠️   | Metaphor, less direct                    |

---

## Part 4: Decision Required from Branding Team

We likely cannot satisfy every constraint at once. Which constraint can we relax?

1. **Accept a slightly less self-descriptive name** (e.g., Neuract, Protégé) and let the tagline explain.
2. **Accept "Physical AI"** as a technology category (not hardware), even under "Intel Robotics".
3. **Accept "Sense Act"** and let legal/security mitigate the collision.

---

## Appendix: PyPI Availability Check (Verified 2026-02-07)

| Package      | PyPI Status                                      |
| ------------ | ------------------------------------------------ |
| `learnact`   | ✅ Available                                     |
| `neuract`    | ✅ Available                                     |
| `sensact`    | ⚠️ Available (owned by us)                       |
| `senseact`   | ❌ Taken (Kindred Robotics, inactive since 2019) |
| `physicalai` | ✅ Available                                     |
| `learnact`   | ✅ Available                                     |
| `protegeai`  | ✅ Available                                     |

---

_Document Version: 2.0_
_Date: 2026-02-07_
_Author: Engineering Team_
