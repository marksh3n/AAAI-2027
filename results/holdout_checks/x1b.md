# X1b holdout checks: is r_hat's lead mostly "finishes soon"?

Fixed 2026-11-02 after the explore half, before any holdout number. All at 4096, capture = share of D.

1. Two forward runs scored on correctness (cap 4096) capture more than every prefix readout (len, restarts,
   committed, probe_y, probe_r, probe_len), point estimates.
2. Finish time is not all of it: two runs scored on finishing within 4096 capture less than the same two runs
   scored on correctness, point estimates.
3. The source trace's true remaining length (oracle) captures under half of D: point estimate below 0.5.
4. The remaining-length probe captures under a quarter of D: point estimate below 0.25.

Explore values: 1. 0.494 vs max readout 0.113 (probe_len); 2. 0.282 vs 0.494; 3. 0.272; 4. 0.113.
