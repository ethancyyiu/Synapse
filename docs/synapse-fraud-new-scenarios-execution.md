# New-scenario study execution record

The [method and code](synapse-fraud-new-scenarios-protocol.md) are fingerprinted in the saved development artifact. Before that artifact was created, the first Gemini response failed the existing indicator-length validator and was discarded. The prompt was tightened to request at most four indicators of at most 55 characters; this final prompt was used for all 12 accepted messages. A transient HTTP 503 interrupted collection once; the retry reused seven cached valid responses and obtained five more.

The development and final evaluations each ran once with the fixed candidate. A preflight generator check counted final cases and verified finite feature values before training; it did not inspect final feature values or predictions or influence the method. No post-test tuning occurred.
