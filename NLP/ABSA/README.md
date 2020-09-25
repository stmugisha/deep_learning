# Aspect-Based Sentiment Analysis (ABSA)

Aspect based sentiment analysis is an NLP technique that breaks down a sequence of text into its consitituent components (aspects), and then allocates a sentiment value (positive, negative, neutral, etc) to a specific aspect.

### Why ABSA?

* ABSA enables getting consumers' finegrained feedback from reviews about products.
    * Consider a review like `Their katogo was very tasty and sweet but I didn't enjoy their chicken as much.` It's both positive and negative but a solving it like normal sentiment analysis task will only predict either of the positive or negative sentiment hence there will be information loss.
    With ABSA, the aspects `katogo` and `chicken` can be extracted and assigned the respective sentiment values thus minimizing information loss. That way this restaurant owner can know that he needs to improve his chicken dishes much as his katogo is tasty.

