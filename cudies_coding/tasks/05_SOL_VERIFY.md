# Task 05 — Sol: final verification and submission package

Read `PLAN.md` and all available handoffs first. Work in `C:\Users\aryan\COEP-NN`. Main window: T+23 to T+30 minutes. T+30 to T+35 is reserved for submission preparation and blocking fixes. If taking over from another Sol chat, confirm it has stopped editing before proceeding.

## Ownership

Own `verify_agent.py`, `tasks/FINAL_CHECKS.md`, final model export, and `cudies_coding/submission/`. Fix production files only for demonstrated correctness or runtime defects. Do not change strategy scope, raw data, organizer files, or unrelated user changes. This task prepares files; the human submits externally.

## Verification order

1. Review the final diff for target leakage, feature mismatch, hidden future inputs, missing validation, swallowed exceptions, and expensive imports. State any corrective approach before editing.
2. Freeze policy and model choices. Evaluate the untouched final chronological block once and report MAE/RMSE against persistence. Report trading diagnostics where supportable. Do not tune on this block; any later correction must be disclosed as affecting independence.
3. Run the focused correctness tests, including unusual history lengths, absent/non-finite observations, zero budget, missing volumes, finite predictions, signed native integers, and deterministic repeated calls.
4. Run bounded matched-seed comparisons for the selected strategy versus the simple baseline. Include realistic mixed and correlated opponent scenarios. Report mean, spread or poor-run outcome, and assumptions. One lucky match is not evidence of first place.
5. Refit the selected forecast using eligible full-history observations only after the untouched holdout report is recorded. Export the final artifact. Check that training and artifact feature metadata match inference.
6. Create `submission/agent.py` and only its required model artifact after checking for existing contents. Do not overwrite unknown files or recursively delete the folder. The package must not import training-only files.
7. Launch fresh Python processes from a different working directory against the packaged agent. Check both functions with representative input and measure import/model-load/call elapsed time. Aim below the minimum stated timeout with margin. Report maximum observed cold latency and test count, not only a warm average.
8. Verify the package contains every dependency it needs and accesses no files beyond its own directory. Use syntax compilation and the organizer harness as supplemental checks. Its offline score after full-data refitting is in-sample; label it accordingly.
9. Record exact package contents, sizes, checksums if easy, selected model/policy, evaluation results, known limitations, and timing in `FINAL_CHECKS.md`. Link the submission directory and files in the final message.

## Deadline behavior

By minute 30, preserve the best verified package. Use the final five minutes for missing-file, malformed-output, import-path, or timeout defects. Avoid new features or another model search.

If a required check fails, report it plainly and use an already-tested fallback when available. Never label an unverified package as ready. Hand the human the exact files to upload and remind them to confirm the submission receipt before the deadline.
