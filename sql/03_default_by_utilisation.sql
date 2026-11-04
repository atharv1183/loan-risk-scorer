-- Credit utilisation (latest bill / limit) against default, with each band's share of all defaults.
WITH util AS (
    SELECT
        CASE
            WHEN bill_1 <= 0                  THEN '1. no balance'
            WHEN bill_1 / limit_bal < 0.30    THEN '2. under 30%'
            WHEN bill_1 / limit_bal < 0.70    THEN '3. 30-70%'
            WHEN bill_1 / limit_bal < 1.00    THEN '4. 70-100%'
            ELSE                                   '5. over limit'
        END AS utilisation_band,
        "default"
    FROM credit
)
SELECT
    utilisation_band,
    COUNT(*)                                                    AS clients,
    ROUND(100.0 * AVG("default"), 1)                            AS default_rate_pct,
    ROUND(100.0 * SUM("default") / SUM(SUM("default")) OVER (), 1) AS share_of_all_defaults_pct
FROM util
GROUP BY utilisation_band
ORDER BY utilisation_band;
