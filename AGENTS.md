# Pontus‑X Agent Instructions (`agents.md`)

This file defines **how an agent should behave** when interacting with the Pontus‑X MCP.
It specifies: response formats, required follow‑up questions, tool‑usage rules, and best‑practice behaviours.

The goal: **consistent, predictable, helpful agent behaviour** during dataset search, download, publication, and compute workflows.

---

# 🧠 GLOBAL AGENT PRINCIPLES

1. **Interpret the user's intent** and map it to the correct MCP tool.
2. **Never call a tool silently** — always tell the user what the agent is doing.
3. **Always respond with structured output** so results are readable in a terminal.
4. **Always propose next steps** after delivering search, download, or compute results.
5. **Ask clarifying questions** if the request is ambiguous.

---

# 🔍 SEARCHING FOR DATASETS

When the user asks anything like:

* "find datasets about …"
* "search for … dataset"
* "look up … in Pontus‑X"

## ✅ Agent must call:

```
pontusx.search_datasets({
  "query": <string>,
  "page": 1,
  "pageSize": 10,
  "minRelevanceScore": 0.2,
  "sort": "_score.-1"
})
```

## ✅ Response Format (MUST follow this)

```
🔎 Search Results for "<query>":

```
<index>. <dataset name>
   DID: <did>
   Price: <price>
   Description: <short description>
```
<repeat for each dataset>
```

## ✅ Required Follow‑Up Question

After every search result:

```
Would you like me to:
1. Download one of these datasets?
2. Run a compute job with a dataset and algorithm?
3. Show more pages of results?
```

---

# 📥 DOWNLOADING A DATASET

When the user says:

* "download X"
* "get dataset Y"
* "I want dataset with DID …"

## ✅ Agent must call:

```
pontusx.download_dataset({ "did": <datasetDid> })
```

## Response Format

```
📥 Dataset Downloaded
Name: <name>
DID: <did>
Files:
 - <file1>
 - <file2>
 - ...
```

## Required Follow‑Up Question

```
Do you want to:
1. Inspect file contents?
2. Run a compute job with this dataset?
3. Download another dataset?
```

---

# 🚀 RUNNING COMPUTE JOBS

Triggered when user says:

* "run compute on …"
* "apply algorithm X to dataset Y"
* "start job"

## Agent must call:

```
pontusx.run_compute({
  "datasetDid": <datasetDid>,
  "algorithmDid": <algorithmDid>,
  "params": { ... }
})
```

## Response Format

```
💻 Compute Job Started
Dataset: <datasetDid>
Algorithm: <algorithmDid>
Status: <pending | running | complete>
Result: <results if available>
```

## Required Follow‑Up Question

```
Would you like to:
1. View detailed logs?
2. Download output files?
3. Run another compute job?
```

---

# 📤 PUBLISHING A DATASET

When user says:

* "publish dataset"
* "upload my dataset"
* "make this available on Pontus‑X"

## Tool call:

```
pontusx.publish_dataset({
  "path": <local-path>,
  "metadata": { ... }
})
```

## Response Format

```
📤 Dataset Published
Name: <name>
New DID: <did>
```

## Required Follow‑Up Question

```
Would you like to:
1. Search for related datasets?
2. Publish another dataset?
3. Run compute using this new dataset?
```

---

# ⚙️ PUBLISHING AN ALGORITHM

When user says:

* "publish algorithm"
* "upload code"
* "register my model"

## Tool call:

```
pontusx.publish_algorithm({
  "dockerChecksum": <sha256>,
  "path": <local-path>,
  "metadata": { ... }
})
```

### ✅ Required Docker checksum steps

Before publishing an algorithm, the agent must determine the SHA256 digest of the Docker image referenced in `dockerImage:dockerTag`:

1. Run `docker manifest inspect <image>:<tag>`.
2. Select the digest whose `platform.os` is `linux` and `platform.architecture` is `amd64`.
3. Supply that digest as `dockerChecksum` (e.g., `sha256:1234...`).
4. If Docker Hub/registry is unreachable, ask the user for the checksum before publishing.

### ✅ Docker architecture requirements

Pontus-X compute nodes run linux/amd64. When using custom Docker images:

1. Confirm the image is built for amd64 (or a multi-arch manifest that includes amd64).
2. Prefer `docker buildx build --platform linux/amd64 ... --push` when building on Apple Silicon/ARM.
3. Record the pushed image/tag/digest locally (e.g., `docker_image_digest.txt`) for future reference.
4. If a compute job fails with `exec format error`, rebuild/push an amd64 variant before republishing the algorithm.

### ✅ Docker verification checklist (before publishing algorithms)

1. Record the exact image tag and checksum (e.g., in `docker_image_digest.txt`).
2. Ensure the container can be pulled anonymously from the registry you reference.
3. Confirm dependencies are installed inside the image (no long runtime `pip install` unless unavoidable).
4. Verify `/data/outputs` (or the documented shared directory) is writable and used for final artifacts.

### ✅ Workload pre-flight checklist (before running compute jobs)

1. Dataset is published with compute enabled (and contains the correct files/URL).
2. Algorithm references the desired image/tag/checksum (amd64-compatible).
3. Fast-mode / quick-test flags are set if you only need a validation run.
4. Expected outputs (models, reports, logs) are written to `/data/outputs`.
5. Note the job ID so you can retrieve artifacts later from `~/.pontusx-datasets/{jobId}-...`.

### ✅ Result-handling requirements

After every job:
1. Call `pontusx.run_compute` with `operation: "results"`.
2. Download all returned files to `~/.pontusx-datasets/{jobId}-{filename}` (the MCP server now supports this).
3. Surface key artifacts in your response (e.g., metrics summary, tarball location, log highlights).
4. If a download fails (expired signature, etc.), request a fresh signed URL via `resultIndex` and retry.

### ✅ Fast-compute best practice

When possible, provide a “fast” execution mode (e.g., flags to limit sample counts, quick epochs, lightweight config) so agents can validate pipelines quickly before launching full-scale jobs—regardless of workload type (training, analytics, etc.).

### ✅ Downloading compute results

After a job finishes, agents should call `pontusx.run_compute` with `operation: "results"`. The MCP server now returns signed URLs for **all** output files (or individual ones via `resultIndex`). Required behavior:

1. Always report how many files were retrieved and list their filenames.
2. Download each file to `~/.pontusx-datasets/{jobId}-{filename}` for easy follow-up actions.
3. If any download fails (signature expired, etc.), request a fresh `resultIndex` link and retry.

## Response Format

```
⚙️ Algorithm Published
Name: <name>
New DID: <did>
```

## Required Follow‑Up Question

```
Do you want to test this algorithm by running a compute job?
```

---

# 🔚 END OF AGENT SPEC

This document defines the **exact behaviour** the agent should follow when using the Pontus‑X MCP.

If you want, I can also add:

* A full decision tree
* ASCII diagrams
* Error handling rules
* A testing script for validating agent behaviour
