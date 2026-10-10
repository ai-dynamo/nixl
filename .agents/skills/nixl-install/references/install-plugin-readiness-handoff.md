# Install And Plugin Readiness Hand-Off

Use this format when install/importability or plugin discovery blocks another
NIXL workflow. Keep it concise and evidence-backed.

```markdown
## Install/Plugin Readiness Hand-Off

### Environment
- Python/runtime:
- Package/source:
- Framework/runtime:
- Container/pod/node:
- Source IDs:

### Import Status
- Status: Pass | Fail | Blocked | TBD-1 | TBD-2 | TBD-3 | TBD-4
- Evidence:
- Interpretation:

### Plugin Path Trust
- `NIXL_PLUGIN_DIR`:
- Package-relative plugin paths:
- Trust status:
- Static evidence:

### Plugin Inventory
- Status: Pass | Fail | Blocked | TBD-3
- Probe/source:
- Observed plugins:
- Missing or suspicious evidence:

### Backend Creation And Memory Types
- Requested backend:
- Creation status:
- Memory types:
- Backend parameters:

### Framework Connector Evidence
- Framework:
- Installed version/source:
- Connector/config evidence:
- Logs:

### Blockers And Confidence
- Confidence: Ready | Blocked | Low | Medium
- Blocking issue:
- Remaining TBDs:

### Next Read-Only Action
1. <least-invasive command, log, source file, or environment fact needed next>
```

Do not include secrets, raw tokens, private hostnames, private bucket names, or
unnecessary absolute paths. Use stable anonymized labels when topology
correlation matters.
