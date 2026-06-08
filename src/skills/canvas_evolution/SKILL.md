---
name: canvas_evolution
description: "Guidelines, pipelines, and decision flows for Swarm Agents to manage canvas creative evolution, query graphs, modify nodes, and troubleshoot image generation failures using SQL database traces."
---

# Canvas Evolution & Image Generation Troubleshooting Pipeline

This skill guides the Swarm Agent in assisting users with creative canvas evolution. It instructs the Agent on how to read workspace graphs, edit nodes, create new branches, and perform deep database troubleshooting when image generation issues (like anatomical errors or duplicate generation results) occur.

## Toolbox Mapping (Agent Capabilities)

You possess the following tools to interact with the canvas workspace:

| Tool | Purpose & Usage |
|---|---|
| `list_workspaces` | Lists all active workspaces and their IDs. |
| `get_canvas_graph` | Fetches the full node/edge topology of a workspace. Retrieves each node's ID, type, position, and business `data` (including `prompt`, `image_url`, `status`, `error_message`, and `reference_images`). |
| `create_canvas_node` | Adds a new node (image card or generation node) to the canvas workspace. |
| `update_canvas_node` | Updates a node's data (such as prompt text, model configurations, etc.). |
| `link_canvas_nodes` | Connects a source node to a target node, building a reference relationship (this also triggers the backend to recompute and update reference images). |
| `generate_image` | Triggers the image generation pipeline for a specific node (sync mode). |
| `run_command` | Execute shell/bash commands. This is your core debugger to run SQL queries against the local database for diagnosis. |

---

## Domain Knowledge: DB Schema

You have database access to the MySQL instance via the `run_command` tool. For any advanced troubleshooting, query these tables:

### Table 1: `workspace_nodes`
Stores canvas node details.
* **id**: Unique node ID (VARCHAR).
* **workspace_id**: The parent workspace ID (VARCHAR).
* **type**: Node type (`image_node`, `image_card`, or `gen_node`).
* **data**: JSON payload containing:
  - `prompt`: The prompt text used for generation.
  - `image_url`: OBS URL of the generated image.
  - `status`: Node status (`success`, `generating`, `failed`).
  - `error_message`: API error details if failed.
  - `request_id`: Traces the HTTP API call ID (linked to `http_request_traces.id`).
  - `reference_images`: Array of parent node image URLs.

### Table 2: `workspace_edges`
Stores connections representing evolution paths.
* **workspace_id**: Workspace ID.
* **source_node_id**: The parent node (source).
* **target_node_id**: The child node (target).
* **label**: The text label written on the connection line (describing the evolution description/instructions).

### Table 3: `http_request_traces`
Records all HTTP requests made to the upstream image generation API (e.g. Doubao Seedream).
* **id**: The request trace ID (VARCHAR, UUID).
* **request_url**: Upstream API endpoint.
* **request_body**: JSON body sent to upstream (contains `prompt`, `model`, `size`, `reference_image_list`, etc.).
* **response_body**: JSON response returned from upstream (contains `success`, `image_url` or error messages).
* **business_success**: Boolean status.
* **error_message**: Technical error details.

---

## Evolution Decision Pipeline

When assisting a user with canvas evolution, follow this logical flow:

### Phase 1: Retrieve Current State
1. Call `get_canvas_graph` to inspect the topology.
2. Read the `label` on the connecting edges to understand what evolution rules (e.g., "change clothing to pink", "add a sword") are being applied.
3. Review `prompt` and `image_url` on existing nodes to establish the context.

### Phase 2: Make Evolution Decisions
1. If the user asks for suggestions or to continue the branch:
   - Identify the terminal node (leaf node with no outgoing edges).
   - Generate a new evolution idea based on the parent nodes and connection history.
   - Create a new node using `create_canvas_node`.
   - Link the parent node(s) to the new node using `link_canvas_nodes`.
   - Update the new node's prompt with your generated prompt using `update_canvas_node`.
   - Call `generate_image` to trigger the generation.

---

## Troubleshooting Pipeline (Anatomy Issues & Duplicate Outputs)

If a user complains that **the image quality is bad (e.g., missing hands, distorted anatomy)** or **generating again returns the exact same image**:

### Step 1: Locating the Request Trace
Identify the `id` of the failed node and fetch its `request_id` (stored in `node.data.request_id`).
Alternatively, look up the last database record from `workspace_nodes` or `http_request_traces`.

### Step 2: Querying the Database for Raw HTTP Payloads
Use the `run_command` tool to execute a MySQL query against `http_request_traces` to review what was actually sent to and received from the upstream model.

```bash
# Query the latest raw request and response details for a specific trace id
mysql -uroot -p -e "USE diy_db; SELECT request_body, response_body, error_message FROM http_request_traces WHERE id = 'YOUR_REQUEST_ID'\G"
```
*(Verify the database credentials in `src/.env` if root requires a password).*

### Step 3: Diagnostic Decision Tree

#### Case A: The generated image is identical to the previous one
1. Inspect the `request_body` from your query.
2. Look for the `seed` parameter. If `seed` is static, or not provided (and the model default is deterministic), subsequent requests with the same prompt and reference images will output the identical result.
3. **Action**: Suggest adding a minor random token or modifying the prompt suffix slightly (e.g., adding "high detail, slightly different pose") to force the model to change its latent noise.

#### Case B: Distorted anatomy (e.g., missing hands, structural chaos)
1. Inspect the `reference_image_list` inside the `request_body`.
2. Check if the model is attempting to fuse conflicting references (e.g., trying to apply clothes from one reference onto a character from another reference, without character guidance).
3. **Action**: Tell the user that combining reference images in a single step causes structural conflicts in the model. Advise them to use text descriptions to bridge the gap rather than compounding too many image inputs, or rewrite the prompt to distinguish the "subject" (e.g., "Asuna") and the "clothing" (e.g., "pink school uniform from reference 2").

### Step 4: Proactive Recovery
Do not just report the error. Help the user fix it:
1. Formulate a refined prompt that addresses the issue (e.g., detailing hand positions like "five clear fingers, resting on her lap" to prevent missing hands).
2. Call `update_canvas_node` to apply the optimized prompt.
3. Call `generate_image` to automatically retry the generation.
