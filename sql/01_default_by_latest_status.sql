-- How strongly does last month's repayment behaviour predict next month's default?
SELECT
    CASE
        WHEN pay_1 = -2 THEN 'a. card not used'
        WHEN pay_1 = -1 THEN 'b. paid in full'
        WHEN pay_1 = 0  THEN 'c. revolving, on time'
        WHEN pay_1 = 1  THEN 'd. 1 month late'
        WHEN pay_1 = 2  THEN 'e. 2 months late'
        ELSE                 'f. 3+ months late'
    END                                   AS latest_status,
    COUNT(*)                              AS clients,
    ROUND(100.0 * AVG("default"), 1)      AS default_rate_pct
FROM credit
GROUP BY 1
ORDER BY 1;
