# Forgetting check: base vs r16 (adapter off vs on, greedy, same prompts)

| benchmark | n | base | r16 | Δ | right→wrong | wrong→right | McNemar p |
|---|---|---|---|---|---|---|---|
| mmlu | 285 | 65.6% | 39.6% | -26.0 pt | 82 | 8 | 0.00 |
| gsm8k | 100 | 92.0% | 85.0% | -7.0 pt | 8 | 1 | 0.04 |

Format probe: answers that came back as JSON instead of prose: base 0/8, r16 0/8. JSON-like answers on the benchmark items: base 0, r16 130.

## Diagnosis

```
{
 "mmlu_diagnosis": {
  "format_r16": {
   "prose": 155,
   "json_with_answer": 42,
   "json_without_answer": 88
  },
  "acc_r16_reading_json_answer": 0.4807017543859649,
  "items_r16_gave_a_letter": 192,
  "acc_base_on_those": 0.6666666666666666,
  "acc_r16_on_those": 0.7135416666666666
 },
 "gsm8k_diagnosis": {
  "no_final_answer_line_base": 3,
  "no_final_answer_line_r16": 8,
  "lost_items_without_final_answer_line": 7,
  "lost_items": 8
 }
}
```

## Format probe, side by side

### probe-code

- **base:** Sure! Here's a Python function that checks whether a string is a palindrome, ignoring case and spaces:  ```python def is_palindrome(s):     """     Returns True if the string is a palindrome, ignoring case and spaces.          Args:         s (str): The string to check.          Returns:         bool: True if the string is a palindrome, False otherwise.     """     # Remove spaces and convert to lowercase     cleaned = s.replace(" ", "").lower()          # Check if the cleaned string reads the s…
- **r16:** ```python def is_palindrome(s):     """     Returns True if a string is a palindrome, ignoring case and spaces.          Args:         s (str): The string to check              Returns:         bool: True if the string is a palindrome, False otherwise     """     # Remove spaces and convert to lowercase     cleaned = s.replace(" ", "").lower()          # Check if the cleaned string reads the same forwards and backwards     return cleaned == cleaned[::-1]

### probe-contract_question

- **base:** Yes, the supplier can terminate the agreement early by providing ninety (90) days' prior written notice to the other party.
- **r16:** Yes, the supplier can terminate the agreement early for convenience upon ninety (90) days' prior written notice to the other party.

### probe-email

- **base:** Subject: Request to Reschedule Delivery – One Week Later    Dear [Supplier's Name],    I hope this message finds you well.    I’m writing to kindly request a one-week delay in the delivery of the order originally scheduled for next Tuesday. This change would allow us to better align the receipt with our internal planning and ensure a smooth integration into our operations.    We understand that this may require some adjustment on your end, and we sincerely appreciate your flexibility and support…
- **r16:** Subject: Request to Reschedule Delivery to Next Tuesday    Dear [Supplier's Name],    I hope you're well. I'm writing to kindly request a one-week delay in the delivery of our order, which was originally scheduled for next Tuesday.    Due to unforeseen internal scheduling changes, we would appreciate it if the delivery could be moved to the following Tuesday instead.    Please let me know if this adjustment works for you or if there’s any additional information you need. I’m happy to discuss fur…

### probe-explain_legal

- **base:** A limitation-of-liability clause in a contract sets a maximum amount of money one party agrees to pay if they are sued for damages. It means the other party can't claim more than that amount, even if the loss is large. This protects the company from being held responsible for huge financial losses in case something goes wrong.
- **r16:** A limitation-of-liability clause in a contract sets a maximum amount of money that one party can be required to pay if they are sued for damages. It's like a cap on how much you can be held responsible for losses. This clause is often used to protect businesses from being sued for huge amounts of money in case of a mistake or accident.

### probe-reasoning

- **base:** Let's solve the problem step by step.  We are told:  - A bat and a ball cost $1.10 in total. - The bat costs $1.00 more than the ball.  Let the cost of the **ball** be **x** dollars.   Then the cost of the **bat** is **x + 1.00** dollars.  Together, they cost:  $$ x + (x + 1.00) = 1.10 $$  Simplify:  $$ 2x + 1.00 = 1.10 $$  Subtract 1.00 from both sides:  $$ 2x = 0.10 $$  Divide by 2:  $$ x = 0.05 $$  So, the **ball costs $0.05** (5 cents).  ✅ Check: - Ball: $0.05 - Bat: $0.05 + $1.00 = $1.05 - …
- **r16:** Let's solve this step by step:  We are told:  - A bat and a ball cost $1.10 in total. - The bat costs $1.00 more than the ball.  Let the cost of the ball be **x** dollars.  Then the cost of the bat is **x + 1.00** dollars.  Together, they cost:  > x + (x + 1.00) = 1.10  Simplify:  > 2x + 1.00 = 1.10  Subtract 1.00 from both sides:  > 2x = 0.10  Divide by 2:  > x = 0.05  So, the ball costs **$0.05**.  ✅ Check:   Ball = $0.05   Bat = $1.00 + $0.05 = $1.05   Total = $0.05 + $1.05 = $1.10 → Correct.…

### probe-summarise

- **base:** Photosynthesis is the process by which green plants, algae, and some bacteria use sunlight to convert carbon dioxide and water into glucose, releasing oxygen as a by-product. This process occurs in chloroplasts containing chlorophyll and is the main source of oxygen in Earth's atmosphere.
- **r16:** Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy, using sunlight to turn carbon dioxide and water into glucose and releasing oxygen as a by-product. This process occurs primarily in chloroplasts, which contain chlorophyll, and is the main source of oxygen in Earth's atmosphere.

### probe-supplier_risks

- **base:** Relying on a single supplier for a critical component poses several significant risks. Here are three key risks:  1. **Supply Chain Disruption**: A single point of failure—such as natural disasters, transportation issues, political instability, or supplier bankruptcy—can lead to sudden shortages or complete interruptions in the supply of the critical component, disrupting production and operations.  2. **Loss of Control and Flexibility**: Dependence on one supplier limits a company’s ability to …
- **r16:** Relying on a single supplier for a critical component poses several significant risks. Here are three key ones:  1. **Supply Chain Disruption**: A single point of failure—such as a natural disaster, political instability, labor strikes, or factory closure—can halt production if the supplier is unable to deliver the component. This can lead to significant downtime, missed deadlines, and financial losses.  2. **Price Volatility and Lack of Negotiation Power**: If there's no competition, the suppli…

### probe-translate

- **base:** La réunion a été déplacée à l'après-midi jeudi.
- **r16:** "La réunion a été déplacée à l'après-midi jeudi."

