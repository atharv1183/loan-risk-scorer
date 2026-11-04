-- Rank age bands by default rate and compare each with the portfolio average (window functions).
WITH bands AS (
    SELECT
        CASE
            WHEN age < 25 THEN '21-24'
            WHEN age < 30 THEN '25-29'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            ELSE               '60+'
        END AS age_band,
        "default"
    FROM credit
),
agg AS (
    SELECT age_band, COUNT(*) AS clients, AVG("default") AS rate
    FROM bands
    GROUP BY age_band
)
SELECT
    age_band,
    clients,
    ROUND(100.0 * rate, 1)                                                    AS default_rate_pct,
    RANK() OVER (ORDER BY rate DESC)                                          AS risk_rank,
    ROUND(100.0 * (rate - SUM(rate * clients) OVER () / SUM(clients) OVER ()), 1) AS vs_average_pct_points
FROM agg
ORDER BY age_band;
