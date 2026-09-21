# Educational Dataset Dictionary

The full dataset is generated deterministically by the project notebook and written to `data/raw/support_tickets.csv`.

| Field | Meaning |
|---|---|
| `ticket_id` | unique synthetic ticket identifier |
| `text` | raw customer support message |
| `intent` | primary routing label: billing/refund/access/technical/account/delivery |
| `kb_article_id` | relevant support article used as retrieval ground truth |
| `channel` | chat, email or app |
| `priority` | low, medium or high |
| `country` | synthetic country code |
| `product` | synthetic product surface |
| `created_at` | synthetic event date |
| `amount` | exact-value ground truth for amount extraction |
| `order_id` | exact-value ground truth for order-id extraction |
| `email` | exact-value ground truth for email extraction |
| `event_date` | exact-value ground truth for date extraction |
| `secondary_intent` | controlled ambiguity field; non-empty rows contain an additional intent phrase |

## Dataset purpose

This is an **educational synthetic/curated dataset**. It is designed to expose the mechanics of a complete NLP lifecycle. It is not a public benchmark and its metrics must not be presented as evidence of production performance.
