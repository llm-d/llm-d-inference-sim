# Response Generation Logic

The simulator determines the content and length of its responses based on the configured **mode** and the specific parameters of each request.

## Echo Mode
In this mode, the simulator acts as a loopback mechanism.
- **Response Content:** Mirrors the echoed text of the input request.
  - For `/v1/completions`: The echoed text is the `prompt` field.
  - For `/v1/chat/completions`: The echoed text is the content of the last message in the `messages` list.
- **Maximum length:** `max_tokens` / `max_completion_tokens` capped by `<context_window> - <input_length>`, or `<context_window> - <input_length>` if neither is set, where `<input_length>` is the token count of the full input (for chat, all messages).
- **Truncation:** If the echoed text is longer than the maximum length, the response is truncated to the first maximum length tokens.
- **`ignore_eos`:** If `true`, `max_tokens` is set, and the maximum length is longer than the echoed text, the echoed text is padded up to the maximum length with predefined text (see [Predefined Text](#predefined-text-default)). Otherwise it has no effect.
- **`min_tokens`:** If the response is still shorter than `min_tokens`, it is padded up to `min_tokens` with predefined text. Negative values and a value above the maximum length return `400 Bad Request`.
- **Response length:** The echoed text's token count, after truncation or padding as described above.
- **Finish reason:** `LENGTH` if the response length reaches the maximum length, otherwise `STOP`.
- **Context window:** The request is rejected with `400 Bad Request` if `<input_length> >= <context_window>`. The response length is still bounded by the remaining context window.

## Random Mode
In this mode, the simulator generates synthetic responses. The length and content depend on the request parameters.

**Context window:** The request is rejected with `400 Bad Request` if `<input_length> >= <context_window>`. The response length is still bounded by the remaining context window; see below.

### Response Length Calculation
`min_tokens` sets the lower bound for generated chat-completion response length. A value of zero is valid and preserves the default lower bound of one token. Negative values, a minimum greater than the requested maximum, and a minimum greater than the context-clamped maximum return `400 Bad Request`.

If `max_tokens` or `max_completion_tokens` is specified, the response length is sampled from a custom histogram with **six buckets** between the minimum and the effective maximum. If the specified value exceeds the room remaining in the context window (`<model_context_limit> - <input_length>`), that remaining room is used as the cap instead — the response never overflows the context window.

**Probability Distribution:**
| Bucket | Probability |
| --- | --- | 
| 1 | 20% | 
| 2 | 30% | 
| 3 | 20% | 
| 4 | 5% | 
| 5 | 10% | 
| 6 | 15% |

**Buckets size**
- Small Requests (≤ 120 tokens): All buckets are equal in size.
- Large Requests (> 120 tokens): Buckets 1, 2, 3, 5, and 6 are fixed at 20 tokens. Bucket 4 expands to cover the remaining range.


**Examples**

`max_tokens = 60`

| Bucket | Size | Tokens |
| --- | --- | --- |
|1|10|1-10|
|2|10|11-20|
|3|10|21-30|
|4|10|31-40|
|5|10|41-50|
|6|10|51-60|

`max_tokens = 200`

| Bucket | Size | Tokens |
| --- | --- | --- |
|1|20|1-20|
|2|20|21-40|
|3|20|41-60|
|**4**|**100**|**61-160**|
|5|20|161-180|
|6|20|181-200|

**Default Length:**
If no maximum length is specified, the length defaults to `<model_context_limit> - <input_length>`. In this specific case, unless the calculated maximum length is smaller than 20, the length is sampled from a Gaussian distribution (Mean=40, SD=20) within the range of the minimum and the maximum. If `min_tokens` is greater than 60 (Mean + SD), the mean is `40 + min_tokens`. If the calculated maximum length is smaller than 20, the length is chosen uniformly at random.

### Content Generation Source

#### Predefined Text (Default)
The simulator constructs responses by concatenating sentences from an internal list of predefined text.
- A random sentence is selected.
- If it exceeds the target length, it is trimmed.
- If it is too short, additional sentences are appended until the target length is met.

#### Dataset Responses (Optional)
If a valid SQLite dataset is provided, the simulator attempts to find a matching conversation:
- Hash Matching: the request prompt is tokenized and hashed and matched against the dataset.
- Selection:
  - If matches are found: A random match longer than the target length is selected and trimmed.
  - If `ignore_eos=true` and no match is long enough: The response is padded with random predefined text.
- Fallback: if the hash is not found, a random response from the dataset is selected (constrained by length).
- Responses shorter than `min_tokens` are padded with predefined text.

### Stop Logic
- `finish_reason`: Set to `LENGTH` if the response reaches the maximum allowed tokens; otherwise set to `STOP`.
- `ignore_eos`: If `true`, the generator forces the response to reach the exact `max_tokens` count, padding with extra content if necessary.
