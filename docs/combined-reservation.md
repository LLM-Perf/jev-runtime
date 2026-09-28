# Combined request reservation

A normal SharedAdmission request on a stable route now commits twice: once to
pin the bundle, persist all engine branch IDs and reserve/enqueue expanded work;
once to release its lease after successful completion or confirmed cancellation.
The previous path committed the lease separately before compilation. SQLite
remains WAL with `synchronous=FULL`.

The runtime first reads a tentative route and immutable manifest in one SELECT,
then compiles outside a write transaction. This does not own a lease or authorize
any engine work. The write transaction checks the alias reference, generation,
manifest digest, active/backend state, adapter readiness and serving admission
policy again. It inserts the lease, tenant, branch journal and ticket together.
All changes roll back if any part fails. GPU dispatch still requires a committed
ADMITTED ticket; queued requests retain their full journal.

If the route changed, including A→B→A, the speculative transaction rolls back and
the request falls back once to the original pin-before-compile path. Compilation
is repeated for the authoritative pinned bundle. Explicit bundle constraints
still apply. A tentative validation error is also checked again under a durable
lease. This avoids unbounded retries during route churn. A request becomes
cross-worker cancellable when its durable lease is committed; tentative work is
local synchronous compilation only. In-memory admission and administrative
compile previews retain the original pin-before-compile behavior.

Retirement may complete while an unpinned compilation is running. Its compiled
work must then be discarded. After the combined commit, retirement/unload guards,
queued cancellation, deadline checks, uncertain-abort retention and recovery use
the same durable records as before. A process exit after commit but before
engine dispatch retains the complete journal and reservation. No timeout-based
capacity reclamation is introduced.

Timing headers retain the existing phase names. For the stable-route path, `pin`
now covers tentative resolution and `queue` includes durable pinning, journaling
and admission. Compare `pin + journal + queue` or `total` across these versions;
comparing `pin` alone would confuse a moved phase boundary with a speedup.

Local tests check two committed transactions, independently observed journal and
ticket visibility before engine calls, rollback at lease/journal/admission
boundaries, hot-switch/retirement races, explicit bundle constraints, stale
validation, digest mismatch and abrupt process exit. Linux owner-death checks
are distinct from conservative rejection on platforms without `/proc` identity.
Actual throughput improvement requires matched DSW before/after evidence; this
implementation alone does not pass any performance release gate.
