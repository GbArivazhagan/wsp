| model / loop control | thinking | seeds | accuracy | think traces in Swedish | think closed by itself | answers in Swedish |
|---|---|---|---|---|---|---|
| p50c edit, 16k | sv | 2 | 38.8 % | 100 % | 95 % | 99 % |
| p50c edit, 16k | en | 2 | 61.5 % | 1 % | 100 % | 74 % |
| unedited + budget forcing 8k | sv | 2 | 34.8 % | 100 % | 36 % | 100 % |
| unedited + budget forcing 8k | en | 2 | 50.0 % | 1 % | 43 % | 80 % |

- **p50c edit (primary):** Δ(sv-think − en-think) = -22.8 [-28.2, -17.2] points, n = 200 problems → worse than English
- **unedited + budget forcing 8k:** Δ(sv-think − en-think) = -15.2 [-20.8, -9.8] points, n = 200 problems → worse than English
