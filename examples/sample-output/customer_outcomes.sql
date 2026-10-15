WITH transactions AS (
    SELECT customer_id, transaction_id, MAX(total_cents) AS total_cents
    FROM purchase_lines GROUP BY customer_id, transaction_id
), customer_totals AS (
    SELECT customer_id, COUNT(*) AS n_transactions, SUM(total_cents) AS spend_cents
    FROM transactions GROUP BY customer_id
)
SELECT a.customer_id, a.treatment,
       CASE WHEN c.n_transactions > 0 THEN 1 ELSE 0 END AS purchased,
       COALESCE(c.spend_cents, 0) AS spend_cents
FROM assignments a LEFT JOIN customer_totals c USING (customer_id)
ORDER BY a.customer_id;
