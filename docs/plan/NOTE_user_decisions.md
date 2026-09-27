# User decisions for the recipe search (binding), 2026-09-26 ~23:50 CEST
1. No KV-cache quantization, ever: FP16 KV only (cache_mode / -cq / Q4-Q8 forbidden: "it destroys quality").
2. 1M context is a strict requirement: every recipe candidate must hold a 1,048,576-token FP16 cache; anything that cannot is
   disqualified (config a may be reported only as a labelled reference).
PLAN.md has these at the top; the three cache_mode items in it are marked FORBIDDEN.
3. Test 11 CMP layers (C11) pipelined with full-size 4096-row sub-chunks next to C10 / C12 (added to PLAN.md).
