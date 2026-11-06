# Factor Research DSH bundle

Local bundle `0.4.4` targets DSH `0.2.1-rc.2`. Install its packed tarball with the official DSH CLI in a project-owned Harness home. See the repository's [operation guide](../../docs/guides/因子挖掘DSH本地安装与操作.md) and [engineering report](../../docs/reports/DSH本地工程交付与验收_20261018.md).

The Host controls one configured Workspace and preserves native sessions, tools, commands, models and Goals everywhere else. A new Workspace opens one planning Coordinator without calculation tools or Goal authority; an approved run binding supplies the fixed three research Sessions. Within that bound Workspace, human input goes through the Coordinator, while A/B inputs, queue edits, forks, role changes and human commands are rejected at admission. Role-specific schema checks and monotonic execution guards cover research tools and cross-server MCP resources. General execution and filesystem tools are unavailable.

Accepted content-addressed A/B tickets are delivered through native prompts with durable receipts. Public turn-end events and native quiescence trigger bounded result notices; the Coordinator reads complete decisions/reports and exact bounded evidence pointers. There is no model polling loop or Python research driver. Finite continuations require original accepted receipts and legal snapshots/checkpoints. Explicit native Goal resume can authorize recovery of a prior user interruption. `read_a_action_request` exposes only an original A tool request whose parameters match the domain receipt hash; it is not a general log reader.

Goal creation is Coordinator-only and bound to the fixed objective. Completion wraps the live native service and independently recomputes the saved domain evidence. Zero retained candidates or insufficient B delivery leave the same Goal paused. Native human `/research-change` approval preserves the parent, creates an immutable versioned child binding, edits the original Goal, retains Session IDs and shares the original budget ledger. It rejects sealed, frozen, B-accessed or unresolved parent runs.

Ordinary Workspace presets and global settings are preserved. The native Goal driver excludes only the research Workspace; research provider-error retries are rejected, and fixed Session titles suppress automatic title requests. Each research preset has its own isolated compaction service with zero compaction/overflow retries. Native compaction is enabled; its auxiliary requests reserve the same durable total budget before provider dispatch. Historical `working_directory` declarations may be replayed only into a native summary request, which executes no tools; live research schemas and execution still reject that tool. Total-budget exhaustion pauses the active native Goal. Input-token and fee ceilings are not implemented.

Stop invalidates pending notices/dispatches, cancels all three Sessions, and coordinates domain cancellation before reporting quiescence. Disposal restores original service methods in a `finally` block. Restart resolves existing native Sessions instead of minting replacements; a persisted active Goal remains disarmed until explicit resume.

The package contains plain ESM Host/Client modules and one bundle patch. DSH peer modules are supplied by the matching Host; no second runtime is bundled. The browser receives only research labels and role Session IDs. Credentials and domain storage are not exposed to the client.

## Desktop Workspace

Install the packed bundle into the desktop profile and bind its `factor-research-admission` entry to the registered `workspaceId` and absolute `workspacePath`. For a new Workspace, set `setupSessionId` to its blank native Session and `sessionRoles: {}`; the Host selects the planning preset without minting another Session. A started Session cannot change to this preset. Planning can clarify the objective and inputs, but cannot calculate or create a Goal before an approved run is bound.

To activate research, replace the setup configuration with a complete run binding and its three actual Workspace Session IDs, following the operation guide. The MCP preset reads paths from that authoritative Host binding. Historical sealed runs, their budgets and Session headers are not rewritten or used to initialize a new run. Research navigation is visible only while a conversation in the configured Workspace is selected.

## Workspace plugin manager

With `dsh-workspace-plugin-manager` installed, set `workspacePolicy: true` on the existing `factor-research-admission` entry. The Host registers its exact Workspace and fixed role Sessions with the policy service. Turning the switch off permits new ordinary Sessions in that Workspace; fixed research Sessions retain their admission, tool and Goal boundaries. Session policies persist across restarts. Without this explicit option, existing independent research profiles keep their original behavior.

## Section scope

Manager 0.2.0 uses sidebar sections as policy boundaries. The adapter registers its fixed workspace as the initial section anchor. New research sessions from any member project still enter the same bound Coordinator; directories and fixed role identities are preserved. Ordinary sessions retain their captured policy after section moves or switches.
