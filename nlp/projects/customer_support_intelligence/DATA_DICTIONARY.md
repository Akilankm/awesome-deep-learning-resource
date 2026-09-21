# Educational Dataset Dictionary

The full ready-made dataset is committed at:

- `data/raw/support_tickets.csv`
- `data/raw/knowledge_base.csv`

The notebook **reads** these inputs and never generates them.

## Support tickets

| Field | Meaning |
|---|---|
| `ticket_id` | unique ticket identifier |
| `text` | raw customer support message |
| `intent` | primary routing label: billing/refund/access/technical/account/delivery |
| `kb_article_id` | relevant support article used as retrieval ground truth |
| `channel` | chat, email or app |
| `priority` | low, medium or high |
| `country` | educational country code |
| `product` | educational product surface |
| `created_at` | event date |
| `amount` | exact-value ground truth for amount extraction when present |
| `order_id` | exact-value ground truth for order-ID extraction when present |
| `email` | exact-value ground truth for email extraction when present |
| `event_date` | exact-value ground truth for date extraction when present |
| `secondary_intent` | optional second issue in an intentionally ambiguous ticket |

## Knowledge base

| Field | Meaning |
|---|---|
| `article_id` | support-article identifier |
| `intent` | broad article domain |
| `title` | article title |
| `content` | short support guidance used for retrieval |

## Dataset purpose

This is an **educational synthetic/curated dataset**. It deliberately includes ambiguity, duplicates, typos, metadata variation, entity-bearing rows and multi-intent tickets so students can practice realistic EDA, leakage control, error analysis and retrieval evaluation.

The construction recipe is not included because the learning goal is to infer patterns from evidence rather than read the pattern-generation rules.
