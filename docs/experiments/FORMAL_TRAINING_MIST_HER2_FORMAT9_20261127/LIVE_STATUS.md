# Live training status

Updated: 2026-11-27 20:02 +08:00 (Shanghai time)

- **Status: STOPPED — invalid for the official MIST train/test protocol.**
- Formal Flow reached 8,000 / 113,325 updates before termination; PPO did not start.
- The generated 4,533/600/509 split mixed official MIST training and test cohorts. It placed 807 official test pairs in the training set. See `SPLIT_AUDIT.md` for the full breakdown.
- Process exit code after intentional SIGTERM: 143. Latest checkpoint and logs are preserved on the remote host for provenance only; do not report this checkpoint as an official-split model and do not resume it for a corrected run.
- No model weights, source images, or prediction arrays are included in this Git record.
