-- Gains table on the held-out test set: split clients into 10 groups by predicted risk and
-- show how many of all defaulters each group catches, cumulatively (riskiest group first).
WITH ranked AS (
    SELECT
        "default",
        probability,
        NTILE(10) OVER (ORDER BY probability DESC) AS risk_decile
    FROM scores
),
by_decile AS (
    SELECT
        risk_decile,
        COUNT(*)              AS clients,
        SUM("default")        AS defaulters,
        MIN(probability)      AS min_prob,
        MAX(probability)      AS max_prob
    FROM ranked
    GROUP BY risk_decile
)
SELECT
    risk_decile,
    clients,
    defaulters,
    ROUND(100.0 * defaulters / clients, 1)                                          AS default_rate_pct,
    ROUND(100.0 * SUM(defaulters) OVER (ORDER BY risk_decile) / SUM(defaulters) OVER (), 1) AS cum_defaulters_caught_pct,
    ROUND(min_prob, 3)                                                              AS min_prob,
    ROUND(max_prob, 3)                                                              AS max_prob
FROM by_decile
ORDER BY risk_decile;
