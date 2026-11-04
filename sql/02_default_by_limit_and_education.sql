-- Default rate by credit-limit band and education, keeping only segments big enough to trust.
WITH banded AS (
    SELECT
        CASE
            WHEN limit_bal <  50000 THEN '1. under 50k'
            WHEN limit_bal < 150000 THEN '2. 50k-150k'
            WHEN limit_bal < 300000 THEN '3. 150k-300k'
            ELSE                         '4. 300k+'
        END AS limit_band,
        CASE education
            WHEN 1 THEN 'graduate school'
            WHEN 2 THEN 'university'
            WHEN 3 THEN 'high school'
            ELSE        'other'
        END AS education,
        "default"
    FROM credit
)
SELECT
    limit_band,
    education,
    COUNT(*)                         AS clients,
    ROUND(100.0 * AVG("default"), 1) AS default_rate_pct
FROM banded
GROUP BY limit_band, education
HAVING COUNT(*) >= 300
ORDER BY limit_band, default_rate_pct DESC;
