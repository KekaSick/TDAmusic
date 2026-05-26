# Popularity vs topological findings

Matched tracks: 198 cached Spotify90 tracks with popularity metadata; genres: electronic, hip-hop, pop, reggae. Pop rank 50 is missing in metadata for pop and was excluded.

## Summary

| space   |   n_tracks | genres                           |   popularity_min |   popularity_median |   popularity_max |   significant_loop_rate |   mean_max_persistence |   mean_chroma_excess |
|:--------|-----------:|:---------------------------------|-----------------:|--------------------:|-----------------:|------------------------:|-----------------------:|---------------------:|
| MIR     |        198 | electronic, hip-hop, pop, reggae |           56.000 |              83.000 |           99.000 |                   0.768 |                  1.636 |                0.049 |
| MuQ     |        198 | electronic, hip-hop, pop, reggae |           56.000 |              83.000 |           99.000 |                   0.500 |                  0.353 |                0.028 |

## Spearman correlations

| space   | metric          |   n |   spearman_rho |   p_value |   genre_centered_rho |   genre_centered_p |   p_fdr |   genre_centered_p_fdr |
|:--------|:----------------|----:|---------------:|----------:|---------------------:|-------------------:|--------:|-----------------------:|
| MIR     | max_persistence | 198 |        -0.0415 |     0.562 |            -0.0532   |             0.457  |   0.715 |                  0.689 |
| MIR     | n_loop_vertices | 198 |         0.0668 |     0.35  |             0.000621 |             0.993  |   0.551 |                  0.993 |
| MIR     | loop_span_sec   | 198 |         0.0858 |     0.229 |             0.0437   |             0.541  |   0.542 |                  0.689 |
| MIR     | chroma_excess   | 190 |         0.0174 |     0.812 |            -0.12     |             0.0982 |   0.812 |                  0.689 |
| MuQ     | max_persistence | 198 |        -0.0298 |     0.677 |             0.052    |             0.467  |   0.729 |                  0.689 |
| MuQ     | n_loop_vertices | 198 |         0.0303 |     0.672 |             0.0528   |             0.46   |   0.729 |                  0.689 |
| MuQ     | loop_span_sec   | 198 |         0.0491 |     0.492 |             0.067    |             0.349  |   0.688 |                  0.689 |
| MuQ     | chroma_excess   | 192 |        -0.107  |     0.141 |            -0.0677   |             0.351  |   0.542 |                  0.689 |

## High-low popularity tertiles

| space   | metric          |   low_n |   high_n |   low_mean |   high_mean |   high_minus_low |   mw_p |   mw_p_fdr |
|:--------|:----------------|--------:|---------:|-----------:|------------:|-----------------:|-------:|-----------:|
| MIR     | max_persistence |      67 |       67 |     1.7    |      1.59   |         -0.108   | 0.294  |      0.498 |
| MIR     | n_loop_vertices |      67 |       67 |   180      |    186      |          5.99    | 0.943  |      0.943 |
| MIR     | loop_span_sec   |      67 |       67 |    74.3    |     77      |          2.68    | 0.351  |      0.498 |
| MIR     | chroma_excess   |      64 |       66 |     0.0609 |      0.0308 |         -0.0301  | 0.0772 |      0.498 |
| MuQ     | max_persistence |      67 |       67 |     0.348  |      0.356  |          0.00842 | 0.237  |      0.498 |
| MuQ     | n_loop_vertices |      67 |       67 |    62.9    |     68.7    |          5.78    | 0.35   |      0.498 |
| MuQ     | loop_span_sec   |      67 |       67 |    69      |     71      |          2       | 0.392  |      0.498 |
| MuQ     | chroma_excess   |      64 |       66 |     0.0482 |      0.0177 |         -0.0306  | 0.0313 |      0.438 |
